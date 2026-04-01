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
        h_pairs = h_prev.view(batch, self.n_pairs, 2)
        cos_a = torch.cos(angles).unsqueeze(-1)  # (batch, n_pairs, 1)
        sin_a = torch.sin(angles).unsqueeze(-1)

        a = h_pairs[..., 0:1]  # (batch, n_pairs, 1)
        b = h_pairs[..., 1:2]

        rotated_a = cos_a * a - sin_a * b
        rotated_b = sin_a * a + cos_a * b
        rotated = torch.cat([rotated_a, rotated_b], dim=-1).view(batch, self.d_model)

        h_next = decay * rotated + injection
        return h_next


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
        else:
            raise ValueError(f"Unknown ssm_type: {ssm_type}")

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)"""
        batch, seq_len, d_model = x.shape
        h = torch.zeros(batch, d_model, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            h = self.ssm(x[:, t], h)
            outputs.append(h)
        out = torch.stack(outputs, dim=1)
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
