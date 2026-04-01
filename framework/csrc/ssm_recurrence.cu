/*
 * Fused SSM Recurrence Kernels (Issue #31)
 *
 * Fuses the entire sequence loop into a single CUDA kernel.
 * Grid: (batch, n_blocks)  — each thread block handles one (batch, block) pair
 * Loop: sequential over T timesteps within the kernel
 *
 * Three variants:
 *   1. quatblock_forward: quaternion sandwich product per 3D block
 *   2. givens_forward: Givens rotation per 2D pair
 *   3. diagonal_forward: per-dimension scalar decay
 *
 * All kernels read pre-computed per-step parameters (projected by PyTorch Linear
 * layers before kernel launch) and write output states.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// QuatBlock kernel: h_{t+1} = decay * quat_rotate(h_t) + injection
// ============================================================================

__global__ void quatblock_forward_kernel(
    const float* __restrict__ bivectors,    // (B, T, n_blocks, 3)
    const float* __restrict__ decays,       // (B, T, 1)
    const float* __restrict__ injections,   // (B, T, D)
    float* __restrict__ states,             // (B, T, D) output
    const int T,
    const int D,
    const int n_blocks
) {
    const int b = blockIdx.x;   // batch index
    const int blk = blockIdx.y; // block index
    const int block_start = blk * 3;

    // Local state for this block (3 floats)
    float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f;

    for (int t = 0; t < T; t++) {
        // Read bivector for this (batch, time, block)
        const int biv_offset = ((b * T + t) * n_blocks + blk) * 3;
        float alpha = bivectors[biv_offset + 0];
        float beta  = bivectors[biv_offset + 1];
        float gamma = bivectors[biv_offset + 2];

        // Bivector -> quaternion: q = cos(|B|/2) + sin(|B|/2) * B/|B|
        float norm_b = sqrtf(alpha * alpha + beta * beta + gamma * gamma);
        float w, qx, qy, qz;
        if (norm_b < 1e-8f) {
            w = 1.0f; qx = 0.0f; qy = 0.0f; qz = 0.0f;
        } else {
            float half_angle = norm_b * 0.5f;
            float cos_ha = cosf(half_angle);
            float sin_ha_over_n = sinf(half_angle) / norm_b;
            w  = cos_ha;
            qx = sin_ha_over_n * gamma;   // e23
            qy = -sin_ha_over_n * beta;   // -e13
            qz = sin_ha_over_n * alpha;   // e12
        }

        // Rodrigues rotation: v' = v + w*t + cross(q_vec, t)
        float tx = 2.0f * (qy * h2 - qz * h1);
        float ty = 2.0f * (qz * h0 - qx * h2);
        float tz = 2.0f * (qx * h1 - qy * h0);

        float rx = h0 + w * tx + (qy * tz - qz * ty);
        float ry = h1 + w * ty + (qz * tx - qx * tz);
        float rz = h2 + w * tz + (qx * ty - qy * tx);

        // Read decay (shared across all blocks for this batch, time)
        const int decay_offset = b * T + t;
        float decay = decays[decay_offset];

        // Read injection for this block's 3 dims
        const int inj_offset = (b * T + t) * D + block_start;
        float i0 = injections[inj_offset + 0];
        float i1 = injections[inj_offset + 1];
        float i2 = injections[inj_offset + 2];

        // h_{t+1} = decay * rotated + injection
        h0 = decay * rx + i0;
        h1 = decay * ry + i1;
        h2 = decay * rz + i2;

        // Write output state
        const int out_offset = (b * T + t) * D + block_start;
        states[out_offset + 0] = h0;
        states[out_offset + 1] = h1;
        states[out_offset + 2] = h2;
    }
}


// ============================================================================
// Givens kernel: h_{t+1} = decay * givens_rotate(h_t) + injection
// ============================================================================

__global__ void givens_forward_kernel(
    const float* __restrict__ angles,       // (B, T, n_pairs)
    const float* __restrict__ decays,       // (B, T, 1)
    const float* __restrict__ injections,   // (B, T, D)
    float* __restrict__ states,             // (B, T, D) output
    const int T,
    const int D,
    const int n_pairs
) {
    const int b = blockIdx.x;
    const int pair = blockIdx.y;
    const int dim_a = pair * 2;
    const int dim_b = pair * 2 + 1;

    float ha = 0.0f, hb = 0.0f;

    for (int t = 0; t < T; t++) {
        // Read angle
        const int angle_offset = (b * T + t) * n_pairs + pair;
        float theta = angles[angle_offset];
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        // Givens rotation
        float ra = cos_t * ha - sin_t * hb;
        float rb = sin_t * ha + cos_t * hb;

        // Decay + injection
        const int decay_offset = b * T + t;
        float decay = decays[decay_offset];

        const int inj_offset = (b * T + t) * D;
        float ia = injections[inj_offset + dim_a];
        float ib = injections[inj_offset + dim_b];

        ha = decay * ra + ia;
        hb = decay * rb + ib;

        // Write
        const int out_offset = (b * T + t) * D;
        states[out_offset + dim_a] = ha;
        states[out_offset + dim_b] = hb;
    }
}


// ============================================================================
// Diagonal kernel: h_{t+1} = diag(decay_t) * h_t + injection
// ============================================================================

__global__ void diagonal_forward_kernel(
    const float* __restrict__ decays,       // (B, T, D)
    const float* __restrict__ injections,   // (B, T, D)
    float* __restrict__ states,             // (B, T, D) output
    const int T,
    const int D
) {
    const int b = blockIdx.x;
    const int d = blockIdx.y;  // dimension index

    float h = 0.0f;

    for (int t = 0; t < T; t++) {
        const int offset = (b * T + t) * D + d;
        float decay = decays[offset];
        float inj = injections[offset];

        h = decay * h + inj;

        states[offset] = h;
    }
}


// ============================================================================
// Pascal kernel: per-dim decay (grade 1) + quaternion rotation (grade 2)
//   h_{t+1} = diag(decay_t) * quat_rotate(h_t) + injection
// ============================================================================

__global__ void pascal_forward_kernel(
    const float* __restrict__ bivectors,    // (B, T, n_blocks, 3)
    const float* __restrict__ decays,       // (B, T, D) — per-dimension!
    const float* __restrict__ injections,   // (B, T, D)
    float* __restrict__ states,             // (B, T, D) output
    const int T,
    const int D,
    const int n_blocks
) {
    const int b = blockIdx.x;
    const int blk = blockIdx.y;
    const int block_start = blk * 3;

    float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f;

    for (int t = 0; t < T; t++) {
        // Bivector -> quaternion (same as quatblock)
        const int biv_offset = ((b * T + t) * n_blocks + blk) * 3;
        float alpha = bivectors[biv_offset + 0];
        float beta  = bivectors[biv_offset + 1];
        float gamma = bivectors[biv_offset + 2];

        float norm_b = sqrtf(alpha * alpha + beta * beta + gamma * gamma);
        float w, qx, qy, qz;
        if (norm_b < 1e-8f) {
            w = 1.0f; qx = 0.0f; qy = 0.0f; qz = 0.0f;
        } else {
            float half_angle = norm_b * 0.5f;
            float cos_ha = cosf(half_angle);
            float sin_ha_over_n = sinf(half_angle) / norm_b;
            w  = cos_ha;
            qx = sin_ha_over_n * gamma;
            qy = -sin_ha_over_n * beta;
            qz = sin_ha_over_n * alpha;
        }

        // Rodrigues rotation
        float tx = 2.0f * (qy * h2 - qz * h1);
        float ty = 2.0f * (qz * h0 - qx * h2);
        float tz = 2.0f * (qx * h1 - qy * h0);

        float rx = h0 + w * tx + (qy * tz - qz * ty);
        float ry = h1 + w * ty + (qz * tx - qx * tz);
        float rz = h2 + w * tz + (qx * ty - qy * tx);

        // Per-dimension decay (grade 1) — the key difference from quatblock
        const int decay_base = (b * T + t) * D + block_start;
        float d0 = decays[decay_base + 0];
        float d1 = decays[decay_base + 1];
        float d2 = decays[decay_base + 2];

        const int inj_offset = (b * T + t) * D + block_start;
        float i0 = injections[inj_offset + 0];
        float i1 = injections[inj_offset + 1];
        float i2 = injections[inj_offset + 2];

        // h_{t+1} = per_dim_decay ⊙ rotated + injection
        h0 = d0 * rx + i0;
        h1 = d1 * ry + i1;
        h2 = d2 * rz + i2;

        const int out_offset = (b * T + t) * D + block_start;
        states[out_offset + 0] = h0;
        states[out_offset + 1] = h1;
        states[out_offset + 2] = h2;
    }
}


// ============================================================================
// C++ wrapper functions (called from Python via torch extension)
// ============================================================================

torch::Tensor quatblock_forward(
    torch::Tensor bivectors,    // (B, T, n_blocks, 3)
    torch::Tensor decays,       // (B, T)
    torch::Tensor injections    // (B, T, D)
) {
    const int B = bivectors.size(0);
    const int T = bivectors.size(1);
    const int n_blocks = bivectors.size(2);
    const int D = n_blocks * 3;

    auto states = torch::zeros({B, T, D}, bivectors.options());

    dim3 grid(B, n_blocks);
    quatblock_forward_kernel<<<grid, 1>>>(
        bivectors.data_ptr<float>(),
        decays.data_ptr<float>(),
        injections.data_ptr<float>(),
        states.data_ptr<float>(),
        T, D, n_blocks
    );

    return states;
}

torch::Tensor givens_forward(
    torch::Tensor angles,       // (B, T, n_pairs)
    torch::Tensor decays,       // (B, T)
    torch::Tensor injections    // (B, T, D)
) {
    const int B = angles.size(0);
    const int T = angles.size(1);
    const int n_pairs = angles.size(2);
    const int D = n_pairs * 2;

    auto states = torch::zeros({B, T, D}, angles.options());

    dim3 grid(B, n_pairs);
    givens_forward_kernel<<<grid, 1>>>(
        angles.data_ptr<float>(),
        decays.data_ptr<float>(),
        injections.data_ptr<float>(),
        states.data_ptr<float>(),
        T, D, n_pairs
    );

    return states;
}

torch::Tensor diagonal_forward(
    torch::Tensor decays,       // (B, T, D)
    torch::Tensor injections    // (B, T, D)
) {
    const int B = decays.size(0);
    const int T = decays.size(1);
    const int D = decays.size(2);

    auto states = torch::zeros({B, T, D}, decays.options());

    dim3 grid(B, D);
    diagonal_forward_kernel<<<grid, 1>>>(
        decays.data_ptr<float>(),
        injections.data_ptr<float>(),
        states.data_ptr<float>(),
        T, D
    );

    return states;
}

torch::Tensor pascal_forward(
    torch::Tensor bivectors,    // (B, T, n_blocks, 3)
    torch::Tensor decays,       // (B, T, D) — per-dimension
    torch::Tensor injections    // (B, T, D)
) {
    const int B = bivectors.size(0);
    const int T = bivectors.size(1);
    const int n_blocks = bivectors.size(2);
    const int D = n_blocks * 3;

    auto states = torch::zeros({B, T, D}, bivectors.options());

    dim3 grid(B, n_blocks);
    pascal_forward_kernel<<<grid, 1>>>(
        bivectors.data_ptr<float>(),
        decays.data_ptr<float>(),
        injections.data_ptr<float>(),
        states.data_ptr<float>(),
        T, D, n_blocks
    );

    return states;
}


// ============================================================================
// Pybind11 module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quatblock_forward", &quatblock_forward,
          "Fused QuatBlock SSM forward (CUDA)");
    m.def("givens_forward", &givens_forward,
          "Fused Givens SSM forward (CUDA)");
    m.def("diagonal_forward", &diagonal_forward,
          "Fused Diagonal SSM forward (CUDA)");
    m.def("pascal_forward", &pascal_forward,
          "Fused Pascal (grade1+grade2) SSM forward (CUDA)");
}
