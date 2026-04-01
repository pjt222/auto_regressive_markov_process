"""
Gated Rotation SSM Experiment (Issue #51)

Tests whether a gated SSM that learns when to apply rotation vs diagonal mode
can adapt to the domain: opening the gate on rotation data (where rotation helps)
and closing it on language data (where rotation hurts, per issue #31).

The critical diagnostic is **mean gate activation**:
  - Near 0 on language  -> model learned to avoid rotation (correct)
  - Near 1 on rotation  -> model learned to use rotation (correct)

Models compared:
  1. GatedRotation: sigmoid gate blends rotation and diagonal paths
  2. QuatBlock: always rotates (Spin(3) sandwich product)
  3. Diagonal: never rotates (per-dim scalar decay)

Usage:
    cd framework && python gated_rotation.py --task rotation --d_model 12 --epochs 30 --seeds 0,1,2
    cd framework && python gated_rotation.py --task language --d_model 192 --epochs 10 --seeds 0
"""

import argparse
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.ssm_cells import (
    QuatBlockSSM, GivensSSM, DiagonalSSM, GatedRotationSSM, SSMLanguageModel,
)
from rotation_prediction import (
    RotationSequenceDataset, SO3PredictionModel, PascalSSM,
    quaternion_geodesic_loss, quaternion_geodesic_degrees,
    make_ssm_cell,
)


# ============================================================================
# 1. SO3PredictionModel with gate tracking for GatedRotationSSM
# ============================================================================

class SO3PredictionModelGated(nn.Module):
    """Wraps an SSM cell for rotation prediction, with gate activation tracking.

    Identical to SO3PredictionModel but tracks mean gate activation per forward
    pass when the cell is GatedRotationSSM.
    """

    def __init__(self, ssm_cell, d_model):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(4, d_model, bias=False)
        self.ssm_cell = ssm_cell
        self.output_proj = nn.Linear(d_model, 4, bias=False)
        self.last_mean_gate = 0.0  # updated each forward pass

        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def forward(self, quat_sequence):
        """
        quat_sequence: (batch, seq_len, 4)
        returns: predicted_quats (batch, seq_len, 4)
        """
        batch, seq_len, _ = quat_sequence.shape
        device = quat_sequence.device

        hidden_input = self.input_proj(quat_sequence)  # (batch, seq_len, d_model)
        cell = self.ssm_cell
        h = torch.zeros(batch, self.d_model, device=device, dtype=quat_sequence.dtype)
        hidden_output = torch.empty(batch, seq_len, self.d_model,
                                    device=device, dtype=quat_sequence.dtype)

        if isinstance(cell, GatedRotationSSM):
            # Precompute projections for gated cell
            bivectors_all = cell.bivector_proj(hidden_input).view(
                batch, seq_len, cell.n_blocks, 3)
            gate_all = torch.sigmoid(cell.gate_proj(hidden_input))  # (B, T, n_blocks)
            rot_decay_all = torch.sigmoid(cell.rot_decay_proj(hidden_input))  # (B, T, 1)
            diag_decay_all = torch.sigmoid(cell.diag_decay_proj(hidden_input))  # (B, T, D)
            injection_all = cell.input_proj(hidden_input)  # (B, T, D)

            for t in range(seq_len):
                h = cell.step_precomputed(
                    bivectors_all[:, t], gate_all[:, t], rot_decay_all[:, t],
                    diag_decay_all[:, t], injection_all[:, t], h)
                hidden_output[:, t] = h

            # Track mean gate activation (detach to avoid graph retention)
            self.last_mean_gate = gate_all.detach().mean().item()

        elif isinstance(cell, QuatBlockSSM):
            bivectors_all = cell.bivector_proj(hidden_input).view(
                batch, seq_len, cell.n_blocks, 3)
            decay_all = torch.sigmoid(cell.decay_proj(hidden_input))
            injection_all = cell.input_proj(hidden_input)
            for t in range(seq_len):
                h = cell.step_precomputed(
                    bivectors_all[:, t], decay_all[:, t], injection_all[:, t], h)
                hidden_output[:, t] = h
            self.last_mean_gate = 1.0  # always rotates

        elif isinstance(cell, DiagonalSSM):
            decay_all = torch.sigmoid(cell.decay_proj(hidden_input))
            injection_all = cell.input_proj(hidden_input)
            for t in range(seq_len):
                h = cell.step_precomputed(
                    decay_all[:, t], injection_all[:, t], h)
                hidden_output[:, t] = h
            self.last_mean_gate = 0.0  # never rotates

        else:
            # Generic fallback (PascalSSM, GivensSSM, etc.)
            for t in range(seq_len):
                h = cell(hidden_input[:, t], h)
                hidden_output[:, t] = h
            self.last_mean_gate = float("nan")

        predicted_raw = self.output_proj(hidden_output)
        predicted_quats = predicted_raw / (
            torch.norm(predicted_raw, dim=-1, keepdim=True).clamp(min=1e-8)
        )
        return predicted_quats

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_ssm_parameters(self):
        return sum(p.numel() for p in self.ssm_cell.parameters() if p.requires_grad)


# ============================================================================
# 2. Gate-tracking wrapper for SSMLanguageModel
# ============================================================================

def compute_mean_gate_language(model):
    """Extract mean gate activation from an SSMLanguageModel with gated layers.

    Returns the mean gate bias sigmoid value as a proxy (no forward pass needed)
    when called outside training. During training, use the hook-based tracker.
    """
    gate_values = []
    for layer in model.layers:
        if isinstance(layer.ssm, GatedRotationSSM):
            # Use the gate bias as a stable proxy for the learned gate preference
            gate_bias = layer.ssm.gate_proj.bias.detach()
            gate_values.append(torch.sigmoid(gate_bias).mean().item())
    if gate_values:
        return sum(gate_values) / len(gate_values)
    return float("nan")


# ============================================================================
# 3. Task A: Rotation prediction
# ============================================================================

def make_cell(ssm_type, d_model):
    """Create SSM cell, supporting 'gated' in addition to the standard types."""
    if ssm_type == "gated":
        return GatedRotationSSM(d_model, block_size=3)
    return make_ssm_cell(ssm_type, d_model)


def run_rotation_task(args):
    """Test A: rotation prediction on SO(3) data."""
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_types = args.models

    print(f"{'='*70}", flush=True)
    print("Task A: Rotation Prediction (SO(3) random walk)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Config: d_model={args.d_model}, seq_len={args.seq_len}, "
          f"epochs={args.epochs}", flush=True)
    print(f"Seeds: {seeds}, Models: {model_types}", flush=True)

    all_results = {m: [] for m in model_types}

    for ssm_type in model_types:
        print(f"\n{'='*70}", flush=True)
        print(f"Model: {ssm_type}", flush=True)
        print(f"{'='*70}", flush=True)

        for seed in seeds:
            print(f"  seed={seed}", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.time()

            # Datasets
            train_dataset = RotationSequenceDataset(
                n_sequences=args.n_sequences, seq_len=args.seq_len,
                gamma=1.0, dt=0.05, sigma_omega=0.3, seed=seed)
            val_dataset = RotationSequenceDataset(
                n_sequences=1000, seq_len=args.seq_len,
                gamma=1.0, dt=0.05, sigma_omega=0.3, seed=seed + 10000)
            test_dataset = RotationSequenceDataset(
                n_sequences=1000, seq_len=args.seq_len,
                gamma=1.0, dt=0.05, sigma_omega=0.3, seed=seed + 20000)

            use_cuda = device.type == "cuda"
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                drop_last=True, pin_memory=use_cuda,
                num_workers=2 if use_cuda else 0,
                persistent_workers=use_cuda)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                drop_last=True, pin_memory=use_cuda)
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                drop_last=True, pin_memory=use_cuda)

            # Model
            cell = make_cell(ssm_type, args.d_model)
            model = SO3PredictionModelGated(cell, args.d_model).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                          weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-4)

            best_val_loss = float("inf")
            best_val_degrees = float("inf")
            best_state_dict = None
            epochs_without_improvement = 0
            gate_history = []

            for epoch in range(1, args.epochs + 1):
                # Train
                model.train()
                epoch_loss = 0.0
                epoch_samples = 0
                epoch_gate_sum = 0.0
                epoch_gate_count = 0

                for input_quats, target_quats in train_loader:
                    input_quats = input_quats.to(device)
                    target_quats = target_quats.to(device)

                    predicted = model(input_quats)
                    loss = quaternion_geodesic_loss(predicted, target_quats)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    batch_size = input_quats.shape[0]
                    epoch_loss += loss.item() * batch_size
                    epoch_samples += batch_size
                    epoch_gate_sum += model.last_mean_gate
                    epoch_gate_count += 1

                scheduler.step()
                mean_gate = epoch_gate_sum / max(epoch_gate_count, 1)
                gate_history.append(mean_gate)

                # Validate
                model.eval()
                val_loss = 0.0
                val_deg = 0.0
                val_samples = 0
                with torch.no_grad():
                    for input_quats, target_quats in val_loader:
                        input_quats = input_quats.to(device)
                        target_quats = target_quats.to(device)
                        predicted = model(input_quats)
                        vl = quaternion_geodesic_loss(predicted, target_quats)
                        vd = quaternion_geodesic_degrees(predicted, target_quats)
                        bs = input_quats.shape[0]
                        val_loss += vl.item() * bs
                        val_deg += vd * bs
                        val_samples += bs

                val_loss /= val_samples
                val_deg /= val_samples

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_degrees = val_deg
                    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epoch % 10 == 0 or epoch == 1:
                    print(f"    epoch {epoch:3d} | train_loss={epoch_loss/epoch_samples:.6f} | "
                          f"val_deg={val_deg:.2f} | gate={mean_gate:.4f}",
                          flush=True)

                if args.patience > 0 and epochs_without_improvement >= args.patience:
                    print(f"    early stop at epoch {epoch}", flush=True)
                    break

            # Test
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)
            model.eval()
            test_loss = 0.0
            test_deg = 0.0
            test_samples = 0
            with torch.no_grad():
                for input_quats, target_quats in test_loader:
                    input_quats = input_quats.to(device)
                    target_quats = target_quats.to(device)
                    predicted = model(input_quats)
                    tl = quaternion_geodesic_loss(predicted, target_quats)
                    td = quaternion_geodesic_degrees(predicted, target_quats)
                    bs = input_quats.shape[0]
                    test_loss += tl.item() * bs
                    test_deg += td * bs
                    test_samples += bs

            test_loss /= test_samples
            test_deg /= test_samples

            elapsed = time.time() - t0
            result = {
                "ssm_type": ssm_type,
                "seed": seed,
                "total_params": model.count_parameters(),
                "ssm_params": model.count_ssm_parameters(),
                "best_val_degrees": best_val_degrees,
                "test_degrees": test_deg,
                "final_gate": gate_history[-1] if gate_history else float("nan"),
                "time": elapsed,
            }
            all_results[ssm_type].append(result)
            print(f"    DONE | val_deg={best_val_degrees:.2f} | "
                  f"test_deg={test_deg:.2f} | gate={result['final_gate']:.4f} | "
                  f"time={elapsed:.1f}s", flush=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY: Rotation Prediction", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Model':>12} | {'Params':>8} | {'SSM Params':>11} | "
          f"{'Test (deg)':>14} | {'Final Gate':>11}", flush=True)
    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*11}-+-{'-'*14}-+-{'-'*11}", flush=True)

    for ssm_type in model_types:
        results_list = all_results[ssm_type]
        mean_test_deg = np.mean([r["test_degrees"] for r in results_list])
        std_test_deg = np.std([r["test_degrees"] for r in results_list])
        mean_gate = np.mean([r["final_gate"] for r in results_list])
        total_params = results_list[0]["total_params"]
        ssm_params = results_list[0]["ssm_params"]

        print(f"{ssm_type:>12} | {total_params:>8,} | {ssm_params:>11,} | "
              f"{mean_test_deg:>5.2f}+/-{std_test_deg:>4.2f} | "
              f"{mean_gate:>11.4f}", flush=True)

    print(f"\nKey: Final Gate near 1.0 = model uses rotation (expected for SO(3) data)",
          flush=True)
    print(f"{'='*70}", flush=True)


# ============================================================================
# 4. Task B: Language modeling
# ============================================================================

def run_language_task(args):
    """Test B: language modeling on WikiText-2."""
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_types = args.models

    print(f"{'='*70}", flush=True)
    print("Task B: Language Modeling (WikiText-2)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Config: d_model={args.d_model}, n_layers={args.n_layers}, "
          f"seq_len={args.seq_len}, epochs={args.epochs}", flush=True)
    print(f"Seeds: {seeds}, Models: {model_types}", flush=True)

    # Load tokenizer and data once
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Import WikiTextDataset from train_language
    sys.path.insert(0, "/mnt/d/dev/p/auto_regressive_markov_process/framework")
    from train_language import WikiTextDataset

    print("\nLoading WikiText-2...", flush=True)
    train_dataset = WikiTextDataset("train", seq_len=args.seq_len,
                                    tokenizer=tokenizer)
    val_dataset = WikiTextDataset("validation", seq_len=args.seq_len,
                                  tokenizer=tokenizer)
    print(f"  Train: {len(train_dataset)} sequences", flush=True)
    print(f"  Val:   {len(val_dataset)} sequences", flush=True)

    vocab_size = tokenizer.vocab_size
    all_results = {m: [] for m in model_types}

    for ssm_type in model_types:
        for seed in seeds:
            print(f"\n{'='*70}", flush=True)
            print(f"Model: {ssm_type}, seed={seed}", flush=True)
            print(f"{'='*70}", flush=True)

            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.time()

            model = SSMLanguageModel(
                vocab_size=vocab_size,
                d_model=args.d_model,
                n_layers=args.n_layers,
                ssm_type=ssm_type,
            ).to(device)

            total_params = model.count_parameters()
            ssm_params = model.count_ssm_parameters()
            print(f"  Total params: {total_params:,}", flush=True)
            print(f"  SSM params:   {ssm_params:,}", flush=True)

            use_cuda = device.type == "cuda"
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                drop_last=True, pin_memory=use_cuda,
                num_workers=2 if use_cuda else 0,
                persistent_workers=use_cuda)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                drop_last=True, pin_memory=use_cuda)

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          weight_decay=0.01)
            total_steps = len(train_loader) * args.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

            best_val_ppl = float("inf")
            gate_history = []

            for epoch in range(1, args.epochs + 1):
                # Train
                model.train()
                epoch_loss = 0.0
                epoch_tokens = 0

                for batch_idx, (input_ids, labels) in enumerate(train_loader):
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)

                    logits = model(input_ids)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), labels.view(-1))

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    batch_tokens = labels.numel()
                    epoch_loss += loss.item() * batch_tokens
                    epoch_tokens += batch_tokens

                train_loss = epoch_loss / epoch_tokens
                train_ppl = math.exp(min(train_loss, 20))

                # Gate tracking
                mean_gate = compute_mean_gate_language(model)
                gate_history.append(mean_gate)

                # Validate
                model.eval()
                val_loss = 0.0
                val_tokens = 0
                with torch.no_grad():
                    for input_ids, labels in val_loader:
                        input_ids = input_ids.to(device)
                        labels = labels.to(device)
                        logits = model(input_ids)
                        loss = nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)), labels.view(-1))
                        val_tokens += labels.numel()
                        val_loss += loss.item() * labels.numel()

                val_loss /= val_tokens
                val_ppl = math.exp(min(val_loss, 20))

                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl

                print(f"  epoch {epoch} | train_ppl={train_ppl:.1f} | "
                      f"val_ppl={val_ppl:.1f} | best={best_val_ppl:.1f} | "
                      f"gate={mean_gate:.4f}", flush=True)

            elapsed = time.time() - t0
            result = {
                "ssm_type": ssm_type,
                "seed": seed,
                "total_params": total_params,
                "ssm_params": ssm_params,
                "best_val_ppl": best_val_ppl,
                "final_gate": gate_history[-1] if gate_history else float("nan"),
                "time": elapsed,
            }
            all_results[ssm_type].append(result)
            print(f"  DONE | best_ppl={best_val_ppl:.1f} | "
                  f"gate={result['final_gate']:.4f} | time={elapsed:.0f}s",
                  flush=True)

            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY: Language Modeling (WikiText-2)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Model':>12} | {'Params':>13} | {'SSM Params':>11} | "
          f"{'Best PPL':>9} | {'Final Gate':>11}", flush=True)
    print(f"{'-'*12}-+-{'-'*13}-+-{'-'*11}-+-{'-'*9}-+-{'-'*11}", flush=True)

    for ssm_type in model_types:
        results_list = all_results[ssm_type]
        mean_ppl = np.mean([r["best_val_ppl"] for r in results_list])
        mean_gate = np.mean([r["final_gate"] for r in results_list])
        total_params = results_list[0]["total_params"]
        ssm_params = results_list[0]["ssm_params"]

        print(f"{ssm_type:>12} | {total_params:>13,} | {ssm_params:>11,} | "
              f"{mean_ppl:>9.1f} | {mean_gate:>11.4f}", flush=True)

    print(f"\nKey: Final Gate near 0.0 = model avoids rotation (expected for language)",
          flush=True)
    print(f"{'='*70}", flush=True)


# ============================================================================
# 5. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gated Rotation SSM Experiment (Issue #51)")
    parser.add_argument("--task", type=str, default="rotation",
                        choices=["rotation", "language"],
                        help="Which task to run (default: rotation)")
    parser.add_argument("--d_model", type=int, default=12,
                        help="Hidden dimension (must be divisible by 3)")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of SSM layers (language task only)")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Sequence length")
    parser.add_argument("--n_sequences", type=int, default=10000,
                        help="Training sequences (rotation task only)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience, 0 to disable (rotation only)")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated random seeds")
    parser.add_argument("--models", nargs="+",
                        default=["gated", "quatblock", "diagonal"],
                        help="SSM types to compare")

    args = parser.parse_args()

    if args.task == "rotation":
        run_rotation_task(args)
    elif args.task == "language":
        # Override defaults for language task if user didn't set them
        if args.d_model == 12:
            print("Note: Using d_model=192 for language task (override with --d_model)",
                  flush=True)
            args.d_model = 192
        if args.seq_len == 128:
            args.seq_len = 256
        if args.batch_size == 64:
            args.batch_size = 8
        run_language_task(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
