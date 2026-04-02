"""
SSM Cell Implementations for Language Modeling (Issue #31)

Three SSM cells with matched architecture, differing only in the transition:
  1. QuatBlockSSM: factored quaternion (Spin(3)) rotation per block of 3 dims
  2. GivensSSM: block-diagonal Givens (pairs of dims) rotation
  3. DiagonalSSM: per-dimension scalar decay (Mamba-style baseline)

All cells implement: h_{t+1} = transition(h_t, x_t) + injection(x_t)
where the transition structure is the independent variable.
"""

import torch
import torch.nn as nn
import math


class QuatBlockSSM(nn.Module):
    """Factored quaternion SSM: D/3 blocks of SU(2) sandwich product.

    h_{t+1} = decay * (q_t h_t q_t^{-1}) + B_t x_t

    Parameters:
        bivector_proj: input -> 3 bivector params per block (for quaternion)
        decay_proj: input -> scalar decay
        input_proj: input -> injection vector
    """

    def __init__(self, d_model, block_size=3):
        super().__init__()
        assert d_model % block_size == 0, f"d_model={d_model} not divisible by block_size={block_size}"
        self.d_model = d_model
        self.block_size = block_size
        self.n_blocks = d_model // block_size

        self.bivector_proj = nn.Linear(d_model, self.n_blocks * 3)
        self.decay_proj = nn.Linear(d_model, 1)
        self.input_proj = nn.Linear(d_model, d_model)

        # Initialize decay bias so sigmoid ≈ 0.8
        nn.init.constant_(self.decay_proj.bias, 1.4)
        nn.init.zeros_(self.bivector_proj.bias)

    def forward(self, x, h_prev):
        """
        x: (batch, d_model)
        h_prev: (batch, d_model)
        returns: h_next (batch, d_model)
        """
        batch = x.shape[0]

        # Input-dependent parameters
        bivectors = self.bivector_proj(x).view(batch, self.n_blocks, 3)
        decay = torch.sigmoid(self.decay_proj(x))  # (batch, 1)
        injection = self.input_proj(x)  # (batch, d_model)

        # Reshape state into blocks of 3
        h_blocks = h_prev.view(batch, self.n_blocks, 3)

        # Quaternion rotation per block
        rotated = self._quat_rotate_blocks(h_blocks, bivectors)

        # Decay + inject
        h_next = decay * rotated.view(batch, self.d_model) + injection
        return h_next

    def step_precomputed(self, bivectors, decay, injection, h_prev):
        """Recurrence step with pre-computed projections (no Linear calls).

        bivectors: (batch, n_blocks, 3)
        decay: (batch, 1)
        injection: (batch, d_model)
        h_prev: (batch, d_model)
        """
        h_blocks = h_prev.view(-1, self.n_blocks, 3)
        rotated = self._quat_rotate_blocks(h_blocks, bivectors)
        return decay * rotated.view(-1, self.d_model) + injection

    def _quat_rotate_blocks(self, h_blocks, bivectors):
        """Batched quaternion sandwich product for all blocks simultaneously.

        h_blocks: (batch, n_blocks, 3)
        bivectors: (batch, n_blocks, 3) — [alpha, beta, gamma] per block
        returns: rotated (batch, n_blocks, 3)
        """
        # Bivector norm
        norm_b = torch.norm(bivectors, dim=-1, keepdim=True).clamp(min=1e-8)
        half_angle = norm_b / 2

        # Quaternion: q = cos(|B|/2) + sin(|B|/2) * B/|B|
        cos_ha = torch.cos(half_angle)  # (batch, n_blocks, 1)
        sin_ha_over_norm = torch.sin(half_angle) / norm_b  # (batch, n_blocks, 1)

        w = cos_ha.squeeze(-1)  # (batch, n_blocks)
        # Convention: e23 -> qx, -e13 -> qy, e12 -> qz
        qx = sin_ha_over_norm[..., 0] * bivectors[..., 2]   # gamma
        qy = -sin_ha_over_norm[..., 0] * bivectors[..., 1]  # -beta
        qz = sin_ha_over_norm[..., 0] * bivectors[..., 0]   # alpha

        # Rodrigues rotation: v' = v + w*t + cross(q_vec, t)
        vx, vy, vz = h_blocks[..., 0], h_blocks[..., 1], h_blocks[..., 2]

        tx = 2 * (qy * vz - qz * vy)
        ty = 2 * (qz * vx - qx * vz)
        tz = 2 * (qx * vy - qy * vx)

        rx = vx + w * tx + (qy * tz - qz * ty)
        ry = vy + w * ty + (qz * tx - qx * tz)
        rz = vz + w * tz + (qx * ty - qy * tx)

        return torch.stack([rx, ry, rz], dim=-1)


class GivensSSM(nn.Module):
    """Givens rotation SSM: D/2 paired rotations.

    h_{t+1} = decay * R(theta_t) h_t + B_t x_t

    where R is block-diagonal Givens rotation on pairs (0,1), (2,3), ...
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.n_pairs = d_model // 2

        self.angle_proj = nn.Linear(d_model, self.n_pairs)
        self.decay_proj = nn.Linear(d_model, 1)
        self.input_proj = nn.Linear(d_model, d_model)

        nn.init.constant_(self.decay_proj.bias, 1.4)
        nn.init.zeros_(self.angle_proj.bias)

    def forward(self, x, h_prev):
        batch = x.shape[0]

        angles = self.angle_proj(x)  # (batch, n_pairs)
        decay = torch.sigmoid(self.decay_proj(x))  # (batch, 1)
        injection = self.input_proj(x)

        # Givens rotation
        rotated = self._rotate(h_prev, angles)
        h_next = decay * rotated + injection
        return h_next

    def _rotate(self, h_prev, angles):
        """Apply Givens rotation to state."""
        batch = h_prev.shape[0]
        h_pairs = h_prev.view(batch, self.n_pairs, 2)
        cos_a = torch.cos(angles).unsqueeze(-1)
        sin_a = torch.sin(angles).unsqueeze(-1)
        a = h_pairs[..., 0:1]
        b = h_pairs[..., 1:2]
        rotated_a = cos_a * a - sin_a * b
        rotated_b = sin_a * a + cos_a * b
        return torch.cat([rotated_a, rotated_b], dim=-1).view(batch, self.d_model)

    def step_precomputed(self, angles, decay, injection, h_prev):
        """Recurrence step with pre-computed projections."""
        rotated = self._rotate(h_prev, angles)
        return decay * rotated + injection


class DiagonalSSM(nn.Module):
    """Diagonal SSM: per-dimension scalar decay (Mamba-style).

    h_{t+1} = diag(lambda_t) * h_t + B_t x_t
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.decay_proj = nn.Linear(d_model, d_model)
        self.input_proj = nn.Linear(d_model, d_model)

        nn.init.constant_(self.decay_proj.bias, 1.4)

    def forward(self, x, h_prev):
        decay = torch.sigmoid(self.decay_proj(x))  # (batch, d_model)
        injection = self.input_proj(x)
        h_next = decay * h_prev + injection
        return h_next

    def step_precomputed(self, decay, injection, h_prev):
        """Recurrence step with pre-computed projections."""
        return decay * h_prev + injection


class GatedRotationSSM(nn.Module):
    """Gated rotation SSM: learns when to rotate vs use diagonal decay.

    gate_t = sigmoid(gate_proj(x_t))           # per-block gate
    h_rotated = rot_decay * quat_rotate(h_t, bivec(x_t))  # rotation path
    h_diagonal = diag_decay(x_t) * h_t         # diagonal path
    h_{t+1} = gate_t * h_rotated + (1 - gate_t) * h_diagonal + injection(x_t)

    The gate learns which path is useful for the domain:
    - Near 0 on language data -> model avoids rotation (correct: rotation hurts)
    - Near 1 on rotation data -> model uses rotation (correct: rotation helps)

    Reuses QuatBlockSSM._quat_rotate_blocks() for the rotation math.
    """

    def __init__(self, d_model, block_size=3):
        super().__init__()
        assert d_model % block_size == 0, f"d_model={d_model} not divisible by block_size={block_size}"
        self.d_model = d_model
        self.block_size = block_size
        self.n_blocks = d_model // block_size

        # Rotation path: bivector params for quaternion rotation + scalar decay
        self.bivector_proj = nn.Linear(d_model, self.n_blocks * 3)
        self.rot_decay_proj = nn.Linear(d_model, 1)

        # Diagonal path: per-dimension decay
        self.diag_decay_proj = nn.Linear(d_model, d_model)

        # Gate: per-block decision (rotate or diagonal)
        self.gate_proj = nn.Linear(d_model, self.n_blocks)

        # Injection (shared)
        self.input_proj = nn.Linear(d_model, d_model)

        # Initialize gate bias to 0.0 so sigmoid = 0.5 (equal weight to both paths)
        nn.init.constant_(self.gate_proj.bias, 0.0)
        nn.init.constant_(self.rot_decay_proj.bias, 1.4)
        nn.init.constant_(self.diag_decay_proj.bias, 1.4)
        nn.init.zeros_(self.bivector_proj.bias)

    def forward(self, x, h_prev):
        """
        x: (batch, d_model)
        h_prev: (batch, d_model)
        returns: h_next (batch, d_model), gate_value (batch, n_blocks)
        """
        batch = x.shape[0]

        # Gate
        gate = torch.sigmoid(self.gate_proj(x))  # (batch, n_blocks)

        # Rotation path
        bivectors = self.bivector_proj(x).view(batch, self.n_blocks, 3)
        rot_decay = torch.sigmoid(self.rot_decay_proj(x))  # (batch, 1)
        h_blocks = h_prev.view(batch, self.n_blocks, 3)
        rotated = QuatBlockSSM._quat_rotate_blocks(None, h_blocks, bivectors)
        h_rotated = rot_decay * rotated.view(batch, self.d_model)  # (batch, d_model)

        # Diagonal path
        diag_decay = torch.sigmoid(self.diag_decay_proj(x))  # (batch, d_model)
        h_diagonal = diag_decay * h_prev  # (batch, d_model)

        # Blend: expand gate from per-block to per-dim for broadcasting
        # gate is (batch, n_blocks), expand to (batch, n_blocks, block_size) -> (batch, d_model)
        gate_expanded = gate.unsqueeze(-1).expand(-1, -1, self.block_size).reshape(batch, self.d_model)

        # Injection
        injection = self.input_proj(x)

        h_next = gate_expanded * h_rotated + (1 - gate_expanded) * h_diagonal + injection
        return h_next

    def forward_with_gate(self, x, h_prev):
        """Like forward() but also returns the gate activation for diagnostics."""
        batch = x.shape[0]

        gate = torch.sigmoid(self.gate_proj(x))  # (batch, n_blocks)

        bivectors = self.bivector_proj(x).view(batch, self.n_blocks, 3)
        rot_decay = torch.sigmoid(self.rot_decay_proj(x))
        h_blocks = h_prev.view(batch, self.n_blocks, 3)
        rotated = QuatBlockSSM._quat_rotate_blocks(None, h_blocks, bivectors)
        h_rotated = rot_decay * rotated.view(batch, self.d_model)

        diag_decay = torch.sigmoid(self.diag_decay_proj(x))
        h_diagonal = diag_decay * h_prev

        gate_expanded = gate.unsqueeze(-1).expand(-1, -1, self.block_size).reshape(batch, self.d_model)
        injection = self.input_proj(x)

        h_next = gate_expanded * h_rotated + (1 - gate_expanded) * h_diagonal + injection
        return h_next, gate

    def step_precomputed(self, bivectors, gate, rot_decay, diag_decay, injection, h_prev):
        """Recurrence step with pre-computed projections (no Linear calls).

        bivectors: (batch, n_blocks, 3)
        gate: (batch, n_blocks)
        rot_decay: (batch, 1)
        diag_decay: (batch, d_model)
        injection: (batch, d_model)
        h_prev: (batch, d_model)
        """
        batch = h_prev.shape[0]

        # Rotation path
        h_blocks = h_prev.view(batch, self.n_blocks, 3)
        rotated = QuatBlockSSM._quat_rotate_blocks(None, h_blocks, bivectors)
        h_rotated = rot_decay * rotated.view(batch, self.d_model)

        # Diagonal path
        h_diagonal = diag_decay * h_prev

        # Blend with per-block gate expanded to per-dim
        gate_expanded = gate.unsqueeze(-1).expand(-1, -1, self.block_size).reshape(batch, self.d_model)

        return gate_expanded * h_rotated + (1 - gate_expanded) * h_diagonal + injection


class SSMLayer(nn.Module):
    """Single SSM layer: projection → SSM cell → LayerNorm + residual."""

    def __init__(self, d_model, ssm_type="quatblock"):
        super().__init__()
        if ssm_type == "quatblock":
            self.ssm = QuatBlockSSM(d_model, block_size=3)
        elif ssm_type == "givens":
            self.ssm = GivensSSM(d_model)
        elif ssm_type == "diagonal":
            self.ssm = DiagonalSSM(d_model)
        elif ssm_type == "gated":
            self.ssm = GatedRotationSSM(d_model, block_size=3)
        else:
            raise ValueError(f"Unknown ssm_type: {ssm_type}")

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)"""
        batch, seq_len, d_model = x.shape
        h = torch.zeros(batch, d_model, device=x.device, dtype=x.dtype)
        out = torch.empty(batch, seq_len, d_model, device=x.device, dtype=x.dtype)

        # Precompute all input-dependent projections as batched matmuls
        ssm = self.ssm
        if isinstance(ssm, QuatBlockSSM):
            bivectors_all = ssm.bivector_proj(x).view(batch, seq_len, ssm.n_blocks, 3)
            decay_all = torch.sigmoid(ssm.decay_proj(x))  # (B, T, 1)
            injection_all = ssm.input_proj(x)  # (B, T, D)
            for t in range(seq_len):
                h = ssm.step_precomputed(
                    bivectors_all[:, t], decay_all[:, t], injection_all[:, t], h)
                out[:, t] = h
        elif isinstance(ssm, GivensSSM):
            angles_all = ssm.angle_proj(x)  # (B, T, n_pairs)
            decay_all = torch.sigmoid(ssm.decay_proj(x))  # (B, T, 1)
            injection_all = ssm.input_proj(x)  # (B, T, D)
            for t in range(seq_len):
                h = ssm.step_precomputed(
                    angles_all[:, t], decay_all[:, t], injection_all[:, t], h)
                out[:, t] = h
        elif isinstance(ssm, DiagonalSSM):
            decay_all = torch.sigmoid(ssm.decay_proj(x))  # (B, T, D)
            injection_all = ssm.input_proj(x)  # (B, T, D)
            for t in range(seq_len):
                h = ssm.step_precomputed(decay_all[:, t], injection_all[:, t], h)
                out[:, t] = h
        elif isinstance(ssm, GatedRotationSSM):
            bivectors_all = ssm.bivector_proj(x).view(batch, seq_len, ssm.n_blocks, 3)
            gate_all = torch.sigmoid(ssm.gate_proj(x))  # (B, T, n_blocks)
            rot_decay_all = torch.sigmoid(ssm.rot_decay_proj(x))  # (B, T, 1)
            diag_decay_all = torch.sigmoid(ssm.diag_decay_proj(x))  # (B, T, D)
            injection_all = ssm.input_proj(x)  # (B, T, D)
            for t in range(seq_len):
                h = ssm.step_precomputed(
                    bivectors_all[:, t], gate_all[:, t], rot_decay_all[:, t],
                    diag_decay_all[:, t], injection_all[:, t], h)
                out[:, t] = h
        else:
            # Fallback for unknown SSM types
            for t in range(seq_len):
                h = ssm(x[:, t], h)
                out[:, t] = h

        return self.norm(out) + x  # residual connection


class SSMLanguageModel(nn.Module):
    """Simple SSM-based language model for next-token prediction."""

    def __init__(self, vocab_size, d_model=768, n_layers=2, ssm_type="quatblock"):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SSMLayer(d_model, ssm_type) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights
        self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.output_proj:
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids):
        """input_ids: (batch, seq_len) -> logits (batch, seq_len, vocab_size)"""
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_ssm_parameters(self):
        """Count only the SSM cell parameters (excluding embedding/output)."""
        total = 0
        for layer in self.layers:
            total += sum(p.numel() for p in layer.ssm.parameters() if p.requires_grad)
        return total
