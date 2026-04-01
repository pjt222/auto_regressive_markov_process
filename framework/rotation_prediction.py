"""
Rotation Prediction Experiment: QuatBlock vs Givens vs Diagonal SSM on SO(3) Data
(Issue #50)

Tests whether the geometric (spinor) inductive bias helps on data with genuine
rotational structure. Generates synthetic sequences of 3D rotations as a random
walk on SO(3) with Ornstein-Uhlenbeck angular velocity, then trains each SSM
variant to predict the next rotation.

This is the domain-specific follow-up to issue #31, which showed rotation
hurts on language data. Rotation prediction is the ideal domain for QuatBlock.

Data generation:
  - Angular velocity follows OU process:
        omega *= (1 - gamma*dt); omega += sigma_omega * sqrt(dt) * noise
  - Rotation integrated via quaternion exponential map:
        q_{t+1} = q_t * exp(omega * dt / 2)

Models:
  1. QuatBlock SSM: D/3 blocks of SU(2) sandwich product (native SO(3))
  2. GivensSSM: D/2 paired rotations (SO(2) ablation -- rotation w/o SO(3) match)
  3. DiagonalSSM: per-dimension scalar decay (Mamba-style baseline, no rotation)
  4. PascalSSM: per-dim decay + rotation (optional "best of both worlds")

Usage:
    cd framework && python rotation_prediction.py [--d_model 12] [--epochs 50]
                                                   [--seeds 0,1,2]
"""

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.ssm_cells import QuatBlockSSM, GivensSSM, DiagonalSSM


# ============================================================================
# 1. PascalSSM (pure-Python, no CUDA required)
# ============================================================================

class PascalSSM(nn.Module):
    """Pascal grade-hierarchy SSM: per-dim decay (grade 1) + rotation (grade 2).

    Combines the per-dimension flexibility of DiagonalSSM with the rotation
    structure of QuatBlockSSM. This is the "best of both worlds" baseline.

    h_{t+1} = diag(lambda_t) * rotate(h_t, bivector_t) + injection_t
    """

    def __init__(self, d_model, block_size=3):
        super().__init__()
        assert d_model % block_size == 0, f"d_model={d_model} not divisible by block_size={block_size}"
        self.d_model = d_model
        self.block_size = block_size
        self.n_blocks = d_model // block_size

        self.bivector_proj = nn.Linear(d_model, self.n_blocks * 3)
        self.decay_proj = nn.Linear(d_model, d_model)  # per-dimension (grade 1)
        self.input_proj = nn.Linear(d_model, d_model)

        nn.init.constant_(self.decay_proj.bias, 1.4)
        nn.init.zeros_(self.bivector_proj.bias)

    def forward(self, x, h_prev):
        """
        x: (batch, d_model)
        h_prev: (batch, d_model)
        returns: h_next (batch, d_model)
        """
        batch = x.shape[0]

        bivectors = self.bivector_proj(x).view(batch, self.n_blocks, 3)
        decay = torch.sigmoid(self.decay_proj(x))  # (batch, d_model) per-dim
        injection = self.input_proj(x)

        # Rotate state blocks
        h_blocks = h_prev.view(batch, self.n_blocks, 3)
        rotated = self._quat_rotate_blocks(h_blocks, bivectors)

        # Per-dimension decay + inject
        h_next = decay * rotated.view(batch, self.d_model) + injection
        return h_next

    def step_precomputed(self, bivectors, decay, injection, h_prev):
        """Recurrence step with pre-computed projections."""
        h_blocks = h_prev.view(-1, self.n_blocks, 3)
        rotated = self._quat_rotate_blocks(h_blocks, bivectors)
        return decay * rotated.view(-1, self.d_model) + injection

    def _quat_rotate_blocks(self, h_blocks, bivectors):
        """Rodrigues rotation via quaternion sandwich product (same as QuatBlockSSM)."""
        norm_b = torch.norm(bivectors, dim=-1, keepdim=True).clamp(min=1e-8)
        half_angle = norm_b / 2

        cos_ha = torch.cos(half_angle)
        sin_ha_over_norm = torch.sin(half_angle) / norm_b

        w = cos_ha.squeeze(-1)
        qx = sin_ha_over_norm[..., 0] * bivectors[..., 2]
        qy = -sin_ha_over_norm[..., 0] * bivectors[..., 1]
        qz = sin_ha_over_norm[..., 0] * bivectors[..., 0]

        vx, vy, vz = h_blocks[..., 0], h_blocks[..., 1], h_blocks[..., 2]

        tx = 2 * (qy * vz - qz * vy)
        ty = 2 * (qz * vx - qx * vz)
        tz = 2 * (qx * vy - qy * vx)

        rx = vx + w * tx + (qy * tz - qz * ty)
        ry = vy + w * ty + (qz * tx - qx * tz)
        rz = vz + w * tz + (qx * ty - qy * tx)

        return torch.stack([rx, ry, rz], dim=-1)


# ============================================================================
# 2. Dataset: random walk on SO(3) with OU angular velocity
# ============================================================================

def quat_multiply(q1, q2):
    """Hamilton product of two quaternions [w, x, y, z].

    q1, q2: (..., 4) numpy arrays
    returns: q1 * q2 as (..., 4) numpy array
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def axis_angle_to_quaternion_batch(rotation_vectors):
    """Convert batch of axis-angle vectors to unit quaternions [w, x, y, z].

    rotation_vectors: (N, 3) rotation vectors (axis * angle)
    returns: (N, 4) unit quaternions
    """
    angles = np.linalg.norm(rotation_vectors, axis=-1, keepdims=True)  # (N, 1)
    half_angles = angles / 2.0

    # Safe division: where angle ≈ 0, axis is irrelevant (identity rotation)
    safe_angles = np.where(angles > 1e-10, angles, 1.0)
    axes = rotation_vectors / safe_angles  # (N, 3)

    cos_half = np.cos(half_angles)          # (N, 1)
    sin_half = np.sin(half_angles)          # (N, 1)

    # Where angle ≈ 0, return identity quaternion
    w = np.where(angles > 1e-10, cos_half, 1.0)        # (N, 1)
    xyz = np.where(angles > 1e-10, sin_half * axes, 0.0)  # (N, 3)

    return np.concatenate([w, xyz], axis=-1)  # (N, 4)


def generate_rotation_sequences_batch(n_sequences, seq_len, gamma=1.0, dt=0.05,
                                      sigma_omega=0.3, seed=0):
    """Generate N rotation sequences in parallel as random walks on SO(3).

    Vectorized across all sequences — only loops over seq_len (typically 128),
    not over n_sequences (typically 10000). ~100x faster than scalar version.

    Angular velocity follows an Ornstein-Uhlenbeck process:
        omega_t = omega_{t-1} * (1 - gamma * dt) + sigma_omega * sqrt(dt) * noise
    Rotation is integrated via quaternion exponential map:
        q_{t+1} = q_t * exp(omega_t * dt / 2)

    Returns: (n_sequences, seq_len, 4) float32 quaternions [w,x,y,z], unit-norm
    """
    rng = np.random.default_rng(seed)

    quaternions = np.zeros((n_sequences, seq_len, 4), dtype=np.float64)
    q_current = np.zeros((n_sequences, 4))
    q_current[:, 0] = 1.0  # identity
    omega = np.zeros((n_sequences, 3))
    sqrt_dt = np.sqrt(dt)
    decay_factor = 1.0 - gamma * dt

    for t in range(seq_len):
        # OU update for angular velocity — all sequences at once
        noise = rng.normal(0, 1, size=(n_sequences, 3))
        omega = omega * decay_factor + sigma_omega * sqrt_dt * noise

        # Incremental rotation quaternions — vectorized
        delta_q = axis_angle_to_quaternion_batch(omega * dt)  # (N, 4)

        # Compose: q_current = q_current * delta_q — vectorized via broadcasting
        q_current = quat_multiply(q_current, delta_q)

        # Re-normalize (numerical drift)
        norms = np.linalg.norm(q_current, axis=-1, keepdims=True)
        q_current = q_current / (norms + 1e-12)

        # Consistent hemisphere: w >= 0
        flip_mask = q_current[:, 0:1] < 0
        q_current = np.where(flip_mask, -q_current, q_current)

        quaternions[:, t] = q_current

    return quaternions.astype(np.float32)


class RotationSequenceDataset(Dataset):
    """Dataset of SO(3) rotation sequences for next-step prediction.

    Generates random walks on SO(3) with OU angular velocity.
    Each sample is (input_quats, target_quats) where target is shifted by 1.
    """

    def __init__(self, n_sequences, seq_len, gamma=1.0, dt=0.05,
                 sigma_omega=0.3, seed=0):
        self.n_sequences = n_sequences
        self.seq_len = seq_len

        self.data = torch.from_numpy(generate_rotation_sequences_batch(
            n_sequences, seq_len, gamma=gamma, dt=dt,
            sigma_omega=sigma_omega, seed=seed,
        ))

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        sequence = self.data[idx]
        input_quats = sequence[:-1]   # (seq_len-1, 4)
        target_quats = sequence[1:]   # (seq_len-1, 4)
        return input_quats, target_quats


# ============================================================================
# 3. Model: SO3PredictionModel wrapping an SSM cell
# ============================================================================

class SO3PredictionModel(nn.Module):
    """Predicts next rotation in a sequence using an SSM cell.

    Architecture:
        quaternion (4) -> input_proj (d_model) -> SSM recurrence -> output_proj (4)
        -> L2 normalize to unit quaternion
    """

    def __init__(self, ssm_cell, d_model):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(4, d_model, bias=False)
        self.ssm_cell = ssm_cell
        self.output_proj = nn.Linear(d_model, 4, bias=False)

        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def forward(self, quat_sequence):
        """
        quat_sequence: (batch, seq_len, 4) -- input quaternions
        returns: predicted_quats (batch, seq_len, 4) -- unit quaternions
        """
        batch, seq_len, _ = quat_sequence.shape
        device = quat_sequence.device

        hidden_input = self.input_proj(quat_sequence)  # (batch, seq_len, d_model)

        # Precompute all input-dependent projections as batched matmuls
        cell = self.ssm_cell
        h = torch.zeros(batch, self.d_model, device=device, dtype=quat_sequence.dtype)
        hidden_output = torch.empty(batch, seq_len, self.d_model,
                                    device=device, dtype=quat_sequence.dtype)

        if hasattr(cell, 'step_precomputed'):
            if isinstance(cell, QuatBlockSSM):
                bivectors_all = cell.bivector_proj(hidden_input).view(
                    batch, seq_len, cell.n_blocks, 3)
                decay_all = torch.sigmoid(cell.decay_proj(hidden_input))
                injection_all = cell.input_proj(hidden_input)
                for t in range(seq_len):
                    h = cell.step_precomputed(
                        bivectors_all[:, t], decay_all[:, t],
                        injection_all[:, t], h)
                    hidden_output[:, t] = h
            elif isinstance(cell, GivensSSM):
                angles_all = cell.angle_proj(hidden_input)
                decay_all = torch.sigmoid(cell.decay_proj(hidden_input))
                injection_all = cell.input_proj(hidden_input)
                for t in range(seq_len):
                    h = cell.step_precomputed(
                        angles_all[:, t], decay_all[:, t],
                        injection_all[:, t], h)
                    hidden_output[:, t] = h
            elif isinstance(cell, PascalSSM):
                bivectors_all = cell.bivector_proj(hidden_input).view(
                    batch, seq_len, cell.n_blocks, 3)
                decay_all = torch.sigmoid(cell.decay_proj(hidden_input))
                injection_all = cell.input_proj(hidden_input)
                for t in range(seq_len):
                    h = cell.step_precomputed(
                        bivectors_all[:, t], decay_all[:, t],
                        injection_all[:, t], h)
                    hidden_output[:, t] = h
            elif isinstance(cell, DiagonalSSM):
                decay_all = torch.sigmoid(cell.decay_proj(hidden_input))
                injection_all = cell.input_proj(hidden_input)
                for t in range(seq_len):
                    h = cell.step_precomputed(
                        decay_all[:, t], injection_all[:, t], h)
                    hidden_output[:, t] = h
            else:
                for t in range(seq_len):
                    h = cell(hidden_input[:, t], h)
                    hidden_output[:, t] = h
        else:
            for t in range(seq_len):
                h = cell(hidden_input[:, t], h)
                hidden_output[:, t] = h

        predicted_raw = self.output_proj(hidden_output)  # (batch, seq_len, 4)

        # L2 normalize to unit quaternion
        predicted_quats = predicted_raw / (
            torch.norm(predicted_raw, dim=-1, keepdim=True).clamp(min=1e-8)
        )
        return predicted_quats

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_ssm_parameters(self):
        return sum(p.numel() for p in self.ssm_cell.parameters() if p.requires_grad)


# ============================================================================
# 4. Loss: quaternion geodesic distance
# ============================================================================

def quaternion_geodesic_loss(predicted, target):
    """Geodesic distance on SO(3) as training loss.

    Loss = 2 * arccos(|<q_pred, q_target>|)

    This is the true geodesic distance (rotation angle between orientations).
    Using |dot| handles antipodal equivalence (q and -q are the same rotation).

    predicted: (batch, seq_len, 4)
    target: (batch, seq_len, 4)
    returns: scalar loss (mean geodesic distance in radians)
    """
    dot_product = torch.sum(predicted * target, dim=-1)  # (batch, seq_len)
    abs_dot = dot_product.abs().clamp(max=1.0 - 1e-7)  # clamp for acos stability
    geodesic_distance = 2.0 * torch.acos(abs_dot)
    return geodesic_distance.mean()


def quaternion_geodesic_degrees(predicted, target):
    """Mean geodesic error in degrees (reporting metric).

    predicted: (batch, seq_len, 4)
    target: (batch, seq_len, 4)
    returns: mean angle in degrees
    """
    dot_product = torch.sum(predicted * target, dim=-1).abs()
    dot_product = dot_product.clamp(max=1.0 - 1e-7)
    angle_rad = 2.0 * torch.acos(dot_product)
    angle_deg = angle_rad * (180.0 / math.pi)
    return angle_deg.mean().item()


# ============================================================================
# 5. Training and evaluation
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for input_quats, target_quats in dataloader:
        input_quats = input_quats.to(device)
        target_quats = target_quats.to(device)

        predicted_quats = model(input_quats)
        loss = quaternion_geodesic_loss(predicted_quats, target_quats)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size = input_quats.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_degrees = 0.0
    total_samples = 0

    for input_quats, target_quats in dataloader:
        input_quats = input_quats.to(device)
        target_quats = target_quats.to(device)

        predicted_quats = model(input_quats)
        loss = quaternion_geodesic_loss(predicted_quats, target_quats)
        degrees = quaternion_geodesic_degrees(predicted_quats, target_quats)

        batch_size = input_quats.shape[0]
        total_loss += loss.item() * batch_size
        total_degrees += degrees * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_degrees / total_samples


# ============================================================================
# 6. Run one model variant
# ============================================================================

def make_ssm_cell(ssm_type, d_model):
    """Create an SSM cell by type name."""
    if ssm_type == "quatblock":
        return QuatBlockSSM(d_model, block_size=3)
    elif ssm_type == "givens":
        return GivensSSM(d_model)
    elif ssm_type == "diagonal":
        return DiagonalSSM(d_model)
    elif ssm_type == "pascal":
        return PascalSSM(d_model, block_size=3)
    else:
        raise ValueError(f"Unknown ssm_type: {ssm_type}")


def run_single_seed(ssm_type, args, seed, device):
    """Train and evaluate one SSM type with one seed. Returns result dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Datasets: train=n_sequences, val=1000, test=1000
    n_val = 1000
    n_test = 1000
    train_dataset = RotationSequenceDataset(
        n_sequences=args.n_sequences, seq_len=args.seq_len,
        gamma=args.gamma, dt=args.dt, sigma_omega=args.sigma_omega,
        seed=seed,
    )
    val_dataset = RotationSequenceDataset(
        n_sequences=n_val, seq_len=args.seq_len,
        gamma=args.gamma, dt=args.dt, sigma_omega=args.sigma_omega,
        seed=seed + 10000,
    )
    test_dataset = RotationSequenceDataset(
        n_sequences=n_test, seq_len=args.seq_len,
        gamma=args.gamma, dt=args.dt, sigma_omega=args.sigma_omega,
        seed=seed + 20000,
    )

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        pin_memory=use_cuda, num_workers=2 if use_cuda else 0,
        persistent_workers=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
        pin_memory=use_cuda,
    )

    # Model
    ssm_cell = make_ssm_cell(ssm_type, args.d_model)
    model = SO3PredictionModel(ssm_cell, args.d_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1,
    )

    best_val_loss = float("inf")
    best_val_degrees = float("inf")
    epochs_without_improvement = 0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        val_loss, val_degrees = evaluate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_degrees = val_degrees
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d} | train_loss={train_loss:.6f} | "
                  f"val_loss={val_loss:.6f} | val_deg={val_degrees:.2f}",
                  flush=True)

        # Early stopping
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"    early stop at epoch {epoch} (patience={args.patience})",
                  flush=True)
            break

    # Evaluate on test set using best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    test_loss, test_degrees = evaluate(model, test_loader, device)

    return {
        "ssm_type": ssm_type,
        "seed": seed,
        "total_params": model.count_parameters(),
        "ssm_params": model.count_ssm_parameters(),
        "best_val_loss": best_val_loss,
        "best_val_degrees": best_val_degrees,
        "test_loss": test_loss,
        "test_degrees": test_degrees,
    }


# ============================================================================
# 7. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SO(3) Rotation Prediction Experiment (Issue #50)")
    parser.add_argument("--d_model", type=int, default=12,
                        help="Hidden dimension (must be divisible by 3 for QuatBlock, "
                             "by 2 for Givens)")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Length of each rotation sequence")
    parser.add_argument("--n_sequences", type=int, default=10000,
                        help="Number of training sequences")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (0 to disable)")
    # OU process parameters
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="OU reversion rate for angular velocity")
    parser.add_argument("--dt", type=float, default=0.05,
                        help="Integration time step")
    parser.add_argument("--sigma_omega", type=float, default=0.3,
                        help="OU diffusion coefficient (rotation noise scale). "
                             "Sweep: {0.1, 0.3, 1.0}")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated random seeds (full run: 0,1,2,3,4)")
    parser.add_argument("--models", nargs="+",
                        default=["quatblock", "givens", "diagonal"],
                        help="SSM types to compare (available: quatblock, givens, "
                             "diagonal, pascal)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*70}", flush=True)
    print("SO(3) Rotation Prediction Experiment (Issue #50)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Config: d_model={args.d_model}, seq_len={args.seq_len}, "
          f"n_sequences={args.n_sequences}", flush=True)
    print(f"        epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}, patience={args.patience}", flush=True)
    print(f"        OU: gamma={args.gamma}, dt={args.dt}, "
          f"sigma_omega={args.sigma_omega}", flush=True)
    print(f"        seeds={seeds}, models={args.models}", flush=True)

    # Collect all results: ssm_type -> list of per-seed results
    all_results = {ssm_type: [] for ssm_type in args.models}

    for ssm_type in args.models:
        print(f"\n{'='*70}", flush=True)
        print(f"Model: {ssm_type}", flush=True)
        print(f"{'='*70}", flush=True)

        for seed in seeds:
            print(f"  seed={seed}", flush=True)
            t0 = time.time()
            result = run_single_seed(ssm_type, args, seed, device)
            elapsed = time.time() - t0
            result["time"] = elapsed
            all_results[ssm_type].append(result)
            print(f"    DONE | val_deg={result['best_val_degrees']:.2f} | "
                  f"test_deg={result['test_degrees']:.2f} | "
                  f"time={elapsed:.1f}s", flush=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ========================================================================
    # Summary table
    # ========================================================================
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY: Mean Geodesic Error (degrees), averaged over seeds", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"{'Model':>12} | {'Params':>8} | {'SSM Params':>11} | "
          f"{'Val (deg)':>14} | {'Test (deg)':>14}", flush=True)
    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*11}-+-{'-'*14}-+-{'-'*14}", flush=True)

    summary = {}
    for ssm_type in args.models:
        results_list = all_results[ssm_type]
        mean_val_deg = np.mean([r["best_val_degrees"] for r in results_list])
        std_val_deg = np.std([r["best_val_degrees"] for r in results_list])
        mean_test_deg = np.mean([r["test_degrees"] for r in results_list])
        std_test_deg = np.std([r["test_degrees"] for r in results_list])
        total_params = results_list[0]["total_params"]
        ssm_params = results_list[0]["ssm_params"]

        summary[ssm_type] = {
            "mean_val_deg": mean_val_deg, "std_val_deg": std_val_deg,
            "mean_test_deg": mean_test_deg, "std_test_deg": std_test_deg,
            "total_params": total_params, "ssm_params": ssm_params,
        }

        print(f"{ssm_type:>12} | {total_params:>8,} | {ssm_params:>11,} | "
              f"{mean_val_deg:>5.2f}+/-{std_val_deg:>4.2f} | "
              f"{mean_test_deg:>5.2f}+/-{std_test_deg:>4.2f}", flush=True)

    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*11}-+-{'-'*14}-+-{'-'*14}", flush=True)

    # Pairwise comparison vs diagonal baseline
    if "diagonal" in summary and len(args.models) >= 2:
        baseline_deg = summary["diagonal"]["mean_test_deg"]
        print(f"\nComparison vs Diagonal (test mean={baseline_deg:.2f} deg):",
              flush=True)
        for ssm_type in args.models:
            if ssm_type != "diagonal":
                delta = summary[ssm_type]["mean_test_deg"] - baseline_deg
                pct = delta / baseline_deg * 100
                direction = "better" if delta < 0 else "worse"
                print(f"  {ssm_type:>12}: {delta:+.2f} deg ({pct:+.1f}%, {direction})",
                      flush=True)

    # Per-seed detail
    print(f"\nPer-seed test results (geodesic degrees):", flush=True)
    header = f"{'Seed':>6}"
    for ssm_type in args.models:
        header += f" | {ssm_type:>12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for i, seed in enumerate(seeds):
        row = f"{seed:>6}"
        for ssm_type in args.models:
            deg = all_results[ssm_type][i]["test_degrees"]
            row += f" | {deg:>12.2f}"
        print(row, flush=True)

    print(f"\n{'='*70}", flush=True)
    print("DONE", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
