#include "cuda_utils.h"
#include "tensor.h"
#include <cstdio>
#include <cmath>

// LayerNorm forward, applied over the last dim.
// x, y: (..., D). Treat as a 2D tensor (N, D) where N = numel / D.
// One thread per row.
//
// Caches rstd = 1/sqrt(var + eps) for backward — one scalar per row.
__global__ void layernorm_forward_kernel(
    const float* x, const float* gamma, const float* beta,
    float* y, float* rstd,
    int N, int D, float eps)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* row_in  = x + n * D;
    float* row_out       = y + n * D;

    // Compute mean
    float mean = 0.f;
    for (int i = 0; i < D; ++i) mean += row_in[i];
    mean /= (float)D;

    // Compute variance
    float var = 0.f;
    for (int i = 0; i < D; ++i) {
        float d = row_in[i] - mean;
        var += d * d;
    }
    var /= (float)D;

    float inv_std = 1.f / std::sqrt(var + eps);
    rstd[n] = inv_std;

    // Normalize and apply affine: y = gamma * (x - mean) * inv_std + beta
    for (int i = 0; i < D; ++i) {
        float x_hat = (row_in[i] - mean) * inv_std;
        row_out[i] = gamma[i] * x_hat + beta[i];
    }
}

void layernorm_forward(
    Tensor& y, Tensor& rstd,
    const Tensor& x,
    const Tensor& gamma, const Tensor& beta,
    float eps = 1e-5f)
{
    int D = x.shape.back();
    int N = (int)(x.numel / D);
    if (gamma.size(0) != D || beta.size(0) != D) {
        std::fprintf(stderr, "layernorm_forward: gamma/beta must be (D,)=%d\n", D);
        std::abort();
    }
    if (rstd.numel != (size_t)N) {
        std::fprintf(stderr, "layernorm_forward: rstd must have numel=%d\n", N);
        std::abort();
    }
    int block = 128;
    int grid = (N + block - 1) / block;
    layernorm_forward_kernel<<<grid, block>>>(
        x.data, gamma.data, beta.data, y.data, rstd.data, N, D, eps);
    CUDA_CHECK_KERNEL();
}

// LayerNorm backward.
//
// Inputs:
//   dy:    (N, D)  upstream gradient
//   x:     (N, D)  forward input
//   gamma: (D,)
//   rstd:  (N,)    cached 1/sqrt(var+eps) per row
// Outputs:
//   dx:     (N, D)
//   dgamma: (D,)   accumulated across all rows
//   dbeta:  (D,)   accumulated across all rows
//
// Kernel 1: per-row, computes dx and accumulates into dgamma/dbeta via atomicAdd.
// (Atomic accumulation is simple and correct; for D=256 and N=1000, collision
// rate is low enough to be fine.)
__global__ void layernorm_backward_kernel(
    const float* dy, const float* x,
    const float* gamma, const float* rstd,
    float* dx, float* dgamma, float* dbeta,
    int N, int D)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* dy_row = dy + n * D;
    const float* x_row  = x  + n * D;
    float*       dx_row = dx + n * D;
    float s = rstd[n];

    // Recompute mean and x_hat for this row.
    // (We don't cache x_hat — recomputing costs one mean pass + D multiplies.)
    float mean = 0.f;
    for (int i = 0; i < D; ++i) mean += x_row[i];
    mean /= (float)D;

    // Compute dx_hat[i] = gamma[i] * dy[i], and the two reductions:
    //   m1 = (1/D) sum_j dx_hat[j]
    //   m2 = (1/D) sum_j x_hat[j] * dx_hat[j]
    float m1 = 0.f, m2 = 0.f;
    for (int i = 0; i < D; ++i) {
        float x_hat = (x_row[i] - mean) * s;
        float dx_hat = gamma[i] * dy_row[i];
        m1 += dx_hat;
        m2 += x_hat * dx_hat;
    }
    m1 /= (float)D;
    m2 /= (float)D;

    // Compute dx[i] = s * (dx_hat[i] - m1 - x_hat[i] * m2)
    // Also accumulate dgamma[i] += dy[i] * x_hat[i] and dbeta[i] += dy[i].
    for (int i = 0; i < D; ++i) {
        float x_hat = (x_row[i] - mean) * s;
        float dx_hat = gamma[i] * dy_row[i];
        dx_row[i] = s * (dx_hat - m1 - x_hat * m2);
        atomicAdd(&dgamma[i], dy_row[i] * x_hat);
        atomicAdd(&dbeta[i],  dy_row[i]);
    }
}

void layernorm_backward(
    Tensor& dx, Tensor& dgamma, Tensor& dbeta,
    const Tensor& dy, const Tensor& x,
    const Tensor& gamma, const Tensor& rstd)
{
    int D = x.shape.back();
    int N = (int)(x.numel / D);

    // Zero dgamma/dbeta before accumulation (atomicAdd is additive).
    CUDA_CHECK(cudaMemset(dgamma.data, 0, D * sizeof(float)));
    CUDA_CHECK(cudaMemset(dbeta.data,  0, D * sizeof(float)));

    int block = 128;
    int grid = (N + block - 1) / block;
    layernorm_backward_kernel<<<grid, block>>>(
        dy.data, x.data, gamma.data, rstd.data,
        dx.data, dgamma.data, dbeta.data, N, D);
    CUDA_CHECK_KERNEL();
}