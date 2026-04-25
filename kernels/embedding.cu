#include "cuda_utils.h"
#include "tensor.h"

#include <cstdio>
#include <cmath>

// Token embedding lookup.
//   tokens: (B, T) device int* (integer token IDs in [0, V))
//   E: (V, D) — embedding table
//   y: (B, T, D) output
__global__ void embedding_forward_kernel(
    const int* tokens, const float* E,
    float* y,
    int B, int T, int D, int V)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * D;
    if (tid >= total) return;

    int d = tid % D;
    int t = (tid / D) % T;
    int b = tid / (D * T);

    int token_id = tokens[b * T + t];
    if (token_id < 0 || token_id >= V) {
        // Out-of-bounds token — write a sentinel value the host can detect.
        y[tid] = -__int_as_float(0x7f800000);;  // -inf
        return;
    }
    y[tid] = E[token_id * D + d];
}

void embedding_forward(
    Tensor& y, const int* tokens_device,
    const Tensor& E, int B, int T)
{
    int V = E.size(0), D = E.size(1);
    if (y.size(0) != B || y.size(1) != T || y.size(2) != D) {
        std::fprintf(stderr, "embedding_forward: y shape mismatch\n");
        std::abort();
    }
    int total = B * T * D;
    int block = 256;
    int grid = (total + block - 1) / block;
    embedding_forward_kernel<<<grid, block>>>(
        tokens_device, E.data, y.data, B, T, D, V);
    CUDA_CHECK_KERNEL();
}

// Backward: scatter-add.
//   dy: (B, T, D)  upstream gradient
//   tokens: (B, T)
//   dE: (V, D) accumulated. *Must be zeroed by caller before this call.*
__global__ void embedding_backward_kernel(
    const float* dy, const int* tokens,
    float* dE,
    int B, int T, int D, int V)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * D;
    if (tid >= total) return;

    int d = tid % D;
    int t = (tid / D) % T;
    int b = tid / (D * T);

    int token_id = tokens[b * T + t];
    if (token_id < 0 || token_id >= V) return;

    atomicAdd(&dE[token_id * D + d], dy[tid]);
}

void embedding_backward(
    Tensor& dE,
    const Tensor& dy, const int* tokens_device,
    int B, int T)
{
    int V = dE.size(0), D = dE.size(1);
    // Zero dE before scatter-add.
    CUDA_CHECK(cudaMemset(dE.data, 0, dE.numel * sizeof(float)));

    int total = B * T * D;
    int block = 256;
    int grid = (total + block - 1) / block;
    embedding_backward_kernel<<<grid, block>>>(
        dy.data, tokens_device, dE.data, B, T, D, V);
    CUDA_CHECK_KERNEL();
}

// Positional embedding: y[b, t, :] += P[t, :]   (in-place addition)
// Or as a fresh write: y[b, t, :] = P[t, :]  (without the +=)
//
// We'll do the in-place add version since the typical use is:
//   y_token = embedding_forward(tokens, E)
//   y_token += positional_forward(P, T)   // residual-style add
//
// Two kernels: positional_add_forward (in-place add) and positional_add_backward.

__global__ void positional_add_forward_kernel(
    float* y,           // (B, T, D) — modified in place
    const float* P,     // (T_max, D)
    int B, int T, int D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * D;
    if (tid >= total) return;

    int d = tid % D;
    int t = (tid / D) % T;
    // b implicit; not needed for the add since P doesn't depend on b
    y[tid] += P[t * D + d];
}

void positional_add_forward(
    Tensor& y, const Tensor& P, int B, int T)
{
    int D = P.size(1);
    if (P.size(0) < T) {
        std::fprintf(stderr, "positional_add_forward: P has only %d rows, need >= %d\n",
                     P.size(0), T);
        std::abort();
    }
    int total = B * T * D;
    int block = 256;
    int grid = (total + block - 1) / block;
    positional_add_forward_kernel<<<grid, block>>>(
        y.data, P.data, B, T, D);
    CUDA_CHECK_KERNEL();
}

// Backward: dP[t, :] = sum over b of dy[b, t, :]
__global__ void positional_add_backward_kernel(
    const float* dy, float* dP,
    int B, int T, int D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * D;
    if (tid >= total) return;

    int d = tid % D;
    int t = (tid / D) % T;
    atomicAdd(&dP[t * D + d], dy[tid]);
}

void positional_add_backward(
    Tensor& dP, const Tensor& dy, int B, int T)
{
    int D = dP.size(1);
    // Zero dP before scatter-add.
    CUDA_CHECK(cudaMemset(dP.data, 0, dP.numel * sizeof(float)));

    int total = B * T * D;
    int block = 256;
    int grid = (total + block - 1) / block;
    positional_add_backward_kernel<<<grid, block>>>(
        dy.data, dP.data, B, T, D);
    CUDA_CHECK_KERNEL();
}