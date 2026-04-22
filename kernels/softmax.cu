#include "cuda_utils.h"
#include "tensor.h"
#include <cstdio>

// Softmax forward, row-wise over a 2D tensor.
// x, y: (B, C). Softmax applied along the last dim (C).
// One thread per row. Each thread: finds max, computes exp & sum, normalizes.
__global__ void softmax_forward_kernel(
    const float* x, float* y, int B, int C)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const float* row_in = x + b * C;
    float* row_out = y + b * C;

    // 1. Find max for numerical stability
    float m = row_in[0];
    for (int i = 1; i < C; ++i) {
        if (row_in[i] > m) m = row_in[i];
    }

    // 2. Compute exp(x - max) and accumulate the sum
    float s = 0.f;
    for (int i = 0; i < C; ++i) {
        float e = expf(row_in[i] - m);
        row_out[i] = e;
        s += e;
    }

    // 3. Normalize (multiply is faster than divide)
    float inv_s = 1.f / s;
    for (int i = 0; i < C; ++i) {
        row_out[i] *= inv_s;
    }
}

void softmax_forward(Tensor& y, const Tensor& x) {
    if (x.ndim() != 2 || y.ndim() != 2) {
        std::fprintf(stderr, "softmax_forward: inputs must be 2D\n");
        std::abort();
    }
    if (x.numel != y.numel || x.size(0) != y.size(0) || x.size(1) != y.size(1)) {
        std::fprintf(stderr, "softmax_forward: shape mismatch\n");
        std::abort();
    }
    int B = x.size(0), C = x.size(1);

    int block = 128;
    int grid = (B + block - 1) / block;
    softmax_forward_kernel<<<grid, block>>>(x.data, y.data, B, C);
    CUDA_CHECK_KERNEL();
}

// Causal-masked row-wise softmax.
// Input/output shape: (B, T, T) — one softmax per (batch, row).
// For row i: only attend to columns j <= i. Position (i, j) with j > i gets 0.
//
// One thread per (batch, row). Each thread loops over the row's T columns.
__global__ void causal_softmax_kernel(
    const float* S, float* P, int B, int T)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = B * T;
    if (tid >= total_rows) return;

    int b = tid / T;
    int i = tid % T;                    // row index — attends to [0..i]
    const float* row_in  = S + (b * T + i) * T;
    float* row_out       = P + (b * T + i) * T;

    // 1. Max over the valid (unmasked) positions.
    float m = row_in[0];                // position 0 is always valid (i >= 0)
    for (int j = 1; j <= i; ++j) {
        if (row_in[j] > m) m = row_in[j];
    }

    // 2. Exp and sum over valid positions; zero out masked positions.
    float s = 0.f;
    for (int j = 0; j <= i; ++j) {
        float e = expf(row_in[j] - m);
        row_out[j] = e;
        s += e;
    }
    for (int j = i + 1; j < T; ++j) {
        row_out[j] = 0.f;
    }

    // 3. Normalize valid positions.
    float inv_s = 1.f / s;
    for (int j = 0; j <= i; ++j) {
        row_out[j] *= inv_s;
    }
}

void causal_softmax(Tensor& P, const Tensor& S) {
    if (S.ndim() != 3) {
        std::fprintf(stderr, "causal_softmax: S must be 3D (B, T, T)\n");
        std::abort();
    }
    int B = S.size(0), T = S.size(1);
    if (S.size(2) != T) {
        std::fprintf(stderr, "causal_softmax: S must be (B, T, T), got (%d, %d, %d)\n",
                     B, T, S.size(2));
        std::abort();
    }

    int total = B * T;
    int block = 128;
    int grid = (total + block - 1) / block;
    causal_softmax_kernel<<<grid, block>>>(S.data, P.data, B, T);
    CUDA_CHECK_KERNEL();
}

// Row-wise softmax backward:
//   dS[b, i, j] = P[b, i, j] * (dP[b, i, j] - sum_k P[b, i, k] * dP[b, i, k])
//
// Shapes: P, dP, dS are all (B, T, T).
// One thread per (batch, row).
//
// Masked positions: if P[b, i, j] = 0 for j > i (causal), then that term
// contributes zero to the sum AND multiplies to zero in the final output.
// So masked gradients stay zero automatically. No special handling needed.
__global__ void softmax_backward_rowwise_kernel(
    const float* P, const float* dP, float* dS, int B, int T)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = B * T;
    if (tid >= total_rows) return;

    int b = tid / T;
    int i = tid % T;
    const float* P_row  = P  + (b * T + i) * T;
    const float* dP_row = dP + (b * T + i) * T;
    float* dS_row       = dS + (b * T + i) * T;

    // Compute scalar s = sum_k P[i, k] * dP[i, k]
    float s = 0.f;
    for (int j = 0; j < T; ++j) {
        s += P_row[j] * dP_row[j];
    }

    // Elementwise: dS[i, j] = P[i, j] * (dP[i, j] - s)
    for (int j = 0; j < T; ++j) {
        dS_row[j] = P_row[j] * (dP_row[j] - s);
    }
}

void softmax_backward_rowwise(
    Tensor& dS, const Tensor& P, const Tensor& dP)
{
    if (P.ndim() != 3 || dP.ndim() != 3 || dS.ndim() != 3) {
        std::fprintf(stderr, "softmax_backward_rowwise: expected 3D tensors\n");
        std::abort();
    }
    int B = P.size(0), T = P.size(1);
    if (P.size(2) != T) {
        std::fprintf(stderr, "softmax_backward_rowwise: P must be (B, T, T)\n");
        std::abort();
    }

    int total = B * T;
    int block = 128;
    int grid = (total + block - 1) / block;
    softmax_backward_rowwise_kernel<<<grid, block>>>(
        P.data, dP.data, dS.data, B, T);
    CUDA_CHECK_KERNEL();
}