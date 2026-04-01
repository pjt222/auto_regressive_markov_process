"""
CUDA-accelerated SSM cells using fused recurrence kernels.

The entire T-step sequential loop runs in a single CUDA kernel,
eliminating per-step kernel launch overhead. Parameters are projected
by standard PyTorch Linear layers, then the fused kernel handles the
recurrence.

Drop-in replacement for ssm_cells.py — same interface, ~50-100x faster.
"""

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# JIT-compile the CUDA extension
_csrc_dir = os.path.join(os.path.dirname(__file__), "..", "csrc")
_ssm_cuda = load(
    name="ssm_recurrence",
    sources=[os.path.join(_csrc_dir, "ssm_recurrence.cu")],
    verbose=False,
)


class QuatBlockSSMCuda(nn.Module):
    """Fused CUDA QuatBlock SSM. Same API as QuatBlockSSM."""

    def __init__(self, d_model, block_size=3):
        super().__init__()
        assert d_model % block_size == 0
        self.d_model = d_model
        self.block_size = block_size
        self.n_blocks = d_model // block_size

        self.bivector_proj = nn.Linear(d_model, self.n_blocks * 3)
        self.decay_proj = nn.Linear(d_model, 1)
        self.input_proj = nn.Linear(d_model, d_model)

        nn.init.constant_(self.decay_proj.bias, 1.4)
        nn.init.zeros_(self.bivector_proj.bias)

    def forward_sequence(self, x_seq):
        """Process full sequence at once via fused CUDA kernel.

        x_seq: (B, T, D) — input embeddings for all timesteps
        returns: states (B, T, D) — hidden states at all timesteps
        """
        B, T, D = x_seq.shape

        # Project all timesteps at once (batched matmuls)
        bivectors = self.bivector_proj(x_seq).view(B, T, self.n_blocks, 3)
        decays = torch.sigmoid(self.decay_proj(x_seq)).squeeze(-1)  # (B, T)
        injections = self.input_proj(x_seq)  # (B, T, D)

        # Fused CUDA kernel — entire T-step recurrence in one launch
        states = _ssm_cuda.quatblock_forward(
            bivectors.contiguous(),
            decays.contiguous(),
            injections.contiguous(),
        )
        return states


class GivensSSMCuda(nn.Module):
    """Fused CUDA Givens SSM."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.n_pairs = d_model // 2

        self.angle_proj = nn.Linear(d_model, self.n_pairs)
        self.decay_proj = nn.Linear(d_model, 1)
        self.input_proj = nn.Linear(d_model, d_model)

        nn.init.constant_(self.decay_proj.bias, 1.4)
        nn.init.zeros_(self.angle_proj.bias)

    def forward_sequence(self, x_seq):
        B, T, D = x_seq.shape

        angles = self.angle_proj(x_seq)  # (B, T, n_pairs)
        decays = torch.sigmoid(self.decay_proj(x_seq)).squeeze(-1)  # (B, T)
        injections = self.input_proj(x_seq)  # (B, T, D)

        states = _ssm_cuda.givens_forward(
            angles.contiguous(),
            decays.contiguous(),
            injections.contiguous(),
        )
        return states


class DiagonalSSMCuda(nn.Module):
    """Fused CUDA Diagonal SSM."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.decay_proj = nn.Linear(d_model, d_model)
        self.input_proj = nn.Linear(d_model, d_model)

        nn.init.constant_(self.decay_proj.bias, 1.4)

    def forward_sequence(self, x_seq):
        B, T, D = x_seq.shape

        decays = torch.sigmoid(self.decay_proj(x_seq))  # (B, T, D)
        injections = self.input_proj(x_seq)  # (B, T, D)

        states = _ssm_cuda.diagonal_forward(
            decays.contiguous(),
            injections.contiguous(),
        )
        return states


class PascalSSMCuda(nn.Module):
    """Pascal grade-hierarchy SSM: per-dim decay (grade 1) + rotation (grade 2).

    The key insight: QuatBlock failed because it used scalar decay (grade 0),
    stripping per-dimension flexibility. This model stacks grade 1 (per-dim
    decay, like Diagonal) WITH grade 2 (rotation, like QuatBlock).

    h_{t+1} = diag(lambda_t) * rotate(h_t, bivector_t) + injection_t
    """

    def __init__(self, d_model, block_size=3):
        super().__init__()
        assert d_model % block_size == 0
        self.d_model = d_model
        self.block_size = block_size
        self.n_blocks = d_model // block_size

        self.bivector_proj = nn.Linear(d_model, self.n_blocks * 3)
        self.decay_proj = nn.Linear(d_model, d_model)  # per-dimension! (grade 1)
        self.input_proj = nn.Linear(d_model, d_model)

        nn.init.constant_(self.decay_proj.bias, 1.4)
        nn.init.zeros_(self.bivector_proj.bias)

    def forward_sequence(self, x_seq):
        B, T, D = x_seq.shape

        bivectors = self.bivector_proj(x_seq).view(B, T, self.n_blocks, 3)
        decays = torch.sigmoid(self.decay_proj(x_seq))  # (B, T, D) per-dim
        injections = self.input_proj(x_seq)

        states = _ssm_cuda.pascal_forward(
            bivectors.contiguous(),
            decays.contiguous(),
            injections.contiguous(),
        )
        return states


class SSMLayerCuda(nn.Module):
    """SSM layer using fused CUDA kernels."""

    def __init__(self, d_model, ssm_type="quatblock"):
        super().__init__()
        if ssm_type == "quatblock":
            self.ssm = QuatBlockSSMCuda(d_model, block_size=3)
        elif ssm_type == "givens":
            self.ssm = GivensSSMCuda(d_model)
        elif ssm_type == "diagonal":
            self.ssm = DiagonalSSMCuda(d_model)
        elif ssm_type == "pascal":
            self.ssm = PascalSSMCuda(d_model, block_size=3)
        else:
            raise ValueError(f"Unknown ssm_type: {ssm_type}")

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, T, D) -> (B, T, D)"""
        states = self.ssm.forward_sequence(x)
        return self.norm(states) + x


class SSMLanguageModelCuda(nn.Module):
    """SSM language model using fused CUDA recurrence kernels."""

    def __init__(self, vocab_size, d_model=768, n_layers=2, ssm_type="quatblock"):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SSMLayerCuda(d_model, ssm_type) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.output_proj:
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_ssm_parameters(self):
        total = 0
        for layer in self.layers:
            total += sum(p.numel() for p in layer.ssm.parameters() if p.requires_grad)
        return total
