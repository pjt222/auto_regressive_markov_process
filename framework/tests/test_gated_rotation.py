"""
Tests for GatedRotationSSM cell (issue #51).

Verifies:
  - Gate initialization at 0.0 / sigmoid=0.5
  - Forward shape correctness
  - forward_with_gate returns gate activations
  - step_precomputed equivalence with forward
  - Gate learns toward 1.0 on rotation-friendly data
  - Gate learns toward 0.0 on diagonal-friendly data
  - Parameter count vs QuatBlockSSM

Usage:
    cd /mnt/d/dev/p/auto_regressive_markov_process
    python -m pytest framework/tests/test_gated_rotation.py -v
"""

import sys
import os

# Add project root so `framework.models.ssm_cells` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import pytest

from framework.models.ssm_cells import GatedRotationSSM, QuatBlockSSM


# ---------------------------------------------------------------------------
# 1. Gate init value
# ---------------------------------------------------------------------------


def test_gate_init_bias_is_zero():
    model = GatedRotationSSM(d_model=12)
    bias = model.gate_proj.bias.detach()
    assert torch.allclose(bias, torch.zeros_like(bias)), (
        f"Expected gate bias to be 0.0, got {bias}"
    )


def test_gate_init_sigmoid_is_half():
    model = GatedRotationSSM(d_model=12)
    gate_sigmoid = torch.sigmoid(model.gate_proj.bias.detach())
    expected = torch.full_like(gate_sigmoid, 0.5)
    assert torch.allclose(gate_sigmoid, expected, atol=1e-6), (
        f"Expected sigmoid(gate bias) = 0.5, got {gate_sigmoid}"
    )


# ---------------------------------------------------------------------------
# 2. Forward shape correctness
# ---------------------------------------------------------------------------


def test_forward_output_shape():
    d_model = 12
    batch = 4
    model = GatedRotationSSM(d_model=d_model)
    x = torch.randn(batch, d_model)
    h_prev = torch.randn(batch, d_model)

    h_next = model(x, h_prev)
    assert h_next.shape == (batch, d_model), (
        f"Expected shape ({batch}, {d_model}), got {h_next.shape}"
    )


# ---------------------------------------------------------------------------
# 3. forward_with_gate returns gate
# ---------------------------------------------------------------------------


def test_forward_with_gate_returns_tuple():
    d_model = 12
    batch = 4
    model = GatedRotationSSM(d_model=d_model)
    x = torch.randn(batch, d_model)
    h_prev = torch.randn(batch, d_model)

    result = model.forward_with_gate(x, h_prev)
    assert isinstance(result, tuple) and len(result) == 2, (
        "forward_with_gate should return (h_next, gate)"
    )

    h_next, gate = result
    assert h_next.shape == (batch, d_model), (
        f"h_next shape: expected ({batch}, {d_model}), got {h_next.shape}"
    )
    assert gate.shape == (batch, model.n_blocks), (
        f"gate shape: expected ({batch}, {model.n_blocks}), got {gate.shape}"
    )


# ---------------------------------------------------------------------------
# 4. step_precomputed equivalence with forward
# ---------------------------------------------------------------------------


def test_step_precomputed_matches_forward():
    d_model = 12
    batch = 4
    model = GatedRotationSSM(d_model=d_model)
    model.eval()

    x = torch.randn(batch, d_model)
    h_prev = torch.randn(batch, d_model)

    with torch.no_grad():
        # forward path
        h_forward = model(x, h_prev)

        # manually compute projections, then call step_precomputed
        bivectors = model.bivector_proj(x).view(batch, model.n_blocks, 3)
        gate = torch.sigmoid(model.gate_proj(x))
        rot_decay = torch.sigmoid(model.rot_decay_proj(x))
        diag_decay = torch.sigmoid(model.diag_decay_proj(x))
        injection = model.input_proj(x)

        h_step = model.step_precomputed(
            bivectors, gate, rot_decay, diag_decay, injection, h_prev
        )

    assert torch.allclose(h_forward, h_step, atol=1e-5), (
        f"Max diff: {(h_forward - h_step).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 5. Gate learns toward 1.0 on rotation-friendly data
# ---------------------------------------------------------------------------


def test_gate_learns_rotation():
    """On data with rotational structure, the gate should learn to prefer
    the rotation path (gate > 0.6 after training).

    Strategy: train on h_next = rotation(h_prev) with NO injection component.
    The rotation is a 90-degree rotation that mixes all 3 dims per block,
    which diagonal scaling fundamentally cannot represent. We freeze the
    injection bias to zero and use a larger learning rate for the gate."""
    torch.manual_seed(42)
    d_model = 6
    batch = 64
    model = GatedRotationSSM(d_model=d_model)

    # Give the gate a higher learning rate so it moves faster
    gate_params = list(model.gate_proj.parameters())
    other_params = [p for p in model.parameters()
                    if not any(p is gp for gp in gate_params)]
    optimizer = torch.optim.Adam([
        {"params": gate_params, "lr": 5e-2},
        {"params": other_params, "lr": 1e-2},
    ])

    # A 90-degree rotation in blocks of 3 that mixes all three dimensions.
    # This is a rotation around the (1,1,1) axis by ~120 degrees:
    # cyclic permutation (x,y,z) -> (z,x,y)
    def make_rotation_batch():
        h = torch.randn(batch, d_model)
        h_next = torch.empty_like(h)
        for b_idx in range(d_model // 3):
            s = b_idx * 3
            # Cyclic permutation: x->y, y->z, z->x
            h_next[:, s] = h[:, s + 2]
            h_next[:, s + 1] = h[:, s]
            h_next[:, s + 2] = h[:, s + 1]
        return h, h_next

    for step in range(300):
        h_prev, h_target = make_rotation_batch()
        h_pred = model(h_prev, h_prev)
        loss = ((h_pred - h_target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check gate activation on a test batch
    model.eval()
    with torch.no_grad():
        h_test, _ = make_rotation_batch()
        _, gate = model.forward_with_gate(h_test, h_test)
        mean_gate = gate.mean().item()

    assert mean_gate > 0.6, (
        f"Expected mean gate > 0.6 on rotation data, got {mean_gate:.3f}"
    )


# ---------------------------------------------------------------------------
# 6. Gate learns toward 0.0 on diagonal-friendly data
# ---------------------------------------------------------------------------


def test_gate_learns_diagonal():
    """On data where next state is a per-dim exponential decay of the previous
    state, the gate should learn to prefer the diagonal path (gate < 0.4)."""
    torch.manual_seed(123)
    d_model = 6
    batch = 64
    model = GatedRotationSSM(d_model=d_model)

    # Give the gate a higher learning rate
    gate_params = list(model.gate_proj.parameters())
    other_params = [p for p in model.parameters()
                    if not any(p is gp for gp in gate_params)]
    optimizer = torch.optim.Adam([
        {"params": gate_params, "lr": 5e-2},
        {"params": other_params, "lr": 1e-2},
    ])

    # Diagonal-friendly: h_next = fixed per-dim decay * h_prev (no rotation).
    # Highly asymmetric decay factors make this impossible for a uniform rotation
    # to approximate, pushing the gate toward diagonal.
    decay_factors = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.15, 0.85])

    def make_diagonal_batch():
        h = torch.randn(batch, d_model)
        h_next = decay_factors.unsqueeze(0) * h
        return h, h_next

    for step in range(300):
        h_prev, h_target = make_diagonal_batch()
        h_pred = model(h_prev, h_prev)
        loss = ((h_pred - h_target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check gate activation
    model.eval()
    with torch.no_grad():
        h_test, _ = make_diagonal_batch()
        _, gate = model.forward_with_gate(h_test, h_test)
        mean_gate = gate.mean().item()

    assert mean_gate < 0.4, (
        f"Expected mean gate < 0.4 on diagonal data, got {mean_gate:.3f}"
    )


# ---------------------------------------------------------------------------
# 7. Parameter count
# ---------------------------------------------------------------------------


def test_parameter_count_exceeds_quatblock():
    d_model = 12
    gated = GatedRotationSSM(d_model=d_model)
    quat = QuatBlockSSM(d_model=d_model, block_size=3)

    gated_params = sum(p.numel() for p in gated.parameters())
    quat_params = sum(p.numel() for p in quat.parameters())

    assert gated_params > quat_params, (
        f"GatedRotationSSM ({gated_params}) should have more params "
        f"than QuatBlockSSM ({quat_params})"
    )


def test_parameter_count_deterministic():
    d_model = 12
    count_a = sum(p.numel() for p in GatedRotationSSM(d_model=d_model).parameters())
    count_b = sum(p.numel() for p in GatedRotationSSM(d_model=d_model).parameters())
    assert count_a == count_b, (
        f"Parameter count should be deterministic: {count_a} vs {count_b}"
    )
