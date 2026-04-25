#include "cuda_utils.h"
#include "tensor.h"
#include <cstdio>

// The universal elementwise launch pattern.
// Each thread handles one element; total threads = numel.
__global__ void relu_forward_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] > 0.f ? x[i] : 0.f;
}

void relu_forward(Tensor& y, const Tensor& x) {
    if (y.numel != x.numel) {
        std::fprintf(stderr, "relu_forward: shape mismatch (%zu vs %zu)\n",
                     y.numel, x.numel);
        std::abort();
    }
    int n = (int)x.numel;
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_forward_kernel<<<grid, block>>>(x.data, y.data, n);
    CUDA_CHECK_KERNEL();
}

// bias_add: y[b, d] = x[b, d] + bias[d], broadcasting bias across the batch.
// Equivalent shape logic for any (..., D) tensor where bias is (D,):
// we index bias by (flat_index % D).
__global__ void bias_add_kernel(const float* x, const float* bias,
                                float* y, int n, int feat_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] + bias[i % feat_dim];
}

void bias_add(Tensor& y, const Tensor& x, const Tensor& bias) {
    if (bias.ndim() != 1) {
        std::fprintf(stderr, "bias_add: bias must be 1D, got %dD\n", bias.ndim());
        std::abort();
    }
    int feat = bias.size(0);
    if ((int)(x.numel % feat) != 0) {
        std::fprintf(stderr, "bias_add: x.numel (%zu) not divisible by feat_dim (%d)\n",
                     x.numel, feat);
        std::abort();
    }
    int n = (int)x.numel;
    int block = 256;
    int grid = (n + block - 1) / block;
    bias_add_kernel<<<grid, block>>>(x.data, bias.data, y.data, n, feat);
    CUDA_CHECK_KERNEL();
}

// ReLU backward: dL/dx = dL/dy * (x > 0), elementwise.
// Needs x (or y — same mask) from forward to compute the gradient mask.
__global__ void relu_backward_kernel(
    const float* dy, const float* x, float* dx, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dx[i] = x[i] > 0.f ? dy[i] : 0.f;
}

void relu_backward(Tensor& dx, const Tensor& dy, const Tensor& x) {
    if (dx.numel != dy.numel || dx.numel != x.numel) {
        std::fprintf(stderr, "relu_backward: shape mismatch\n");
        std::abort();
    }
    int n = (int)x.numel;
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_backward_kernel<<<grid, block>>>(dy.data, x.data, dx.data, n);
    CUDA_CHECK_KERNEL();
}

// Per-row argmax: preds[b] = argmax_c logits[b, c]
// One thread per row. Serial scan over C classes.
__global__ void argmax_row_kernel(
    const float* logits, int* preds, int B, int C)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const float* row = logits + b * C;
    int best = 0;
    float best_v = row[0];
    for (int c = 1; c < C; ++c) {
        if (row[c] > best_v) { best_v = row[c]; best = c; }
    }
    preds[b] = best;
}

void argmax_row(int* preds_device, const Tensor& logits) {
    int B = logits.size(0);
    int C = logits.size(1);
    int block = 128;
    int grid = (B + block - 1) / block;
    argmax_row_kernel<<<grid, block>>>(logits.data, preds_device, B, C);
    CUDA_CHECK_KERNEL();
}

// GELU forward (exact via erf):
//   y = 0.5 * x * (1 + erff(x / sqrt(2)))
__global__ void gelu_forward_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    y[i] = 0.5f * xi * (1.f + erff(xi * 0.7071067811865475f));
}

void gelu_forward(Tensor& y, const Tensor& x) {
    if (y.numel != x.numel) {
        std::fprintf(stderr, "gelu_forward: shape mismatch\n");
        std::abort();
    }
    int n = (int)x.numel;
    int block = 256;
    int grid = (n + block - 1) / block;
    gelu_forward_kernel<<<grid, block>>>(x.data, y.data, n);
    CUDA_CHECK_KERNEL();
}

// GELU backward:
//   dx = dy * (Phi(x) + x * phi(x))
//      = dy * (0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2)/sqrt(2*pi))
__global__ void gelu_backward_kernel(
    const float* dy, const float* x, float* dx, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    // Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    float phi_x = 0.5f * (1.f + erff(xi * 0.7071067811865475f));
    // pdf(x) = exp(-x^2/2) / sqrt(2*pi)
    float pdf_x = expf(-0.5f * xi * xi) * 0.3989422804014327f;
    dx[i] = dy[i] * (phi_x + xi * pdf_x);
}

void gelu_backward(Tensor& dx, const Tensor& dy, const Tensor& x) {
    if (dx.numel != x.numel || dy.numel != x.numel) {
        std::fprintf(stderr, "gelu_backward: shape mismatch\n");
        std::abort();
    }
    int n = (int)x.numel;
    int block = 256;
    int grid = (n + block - 1) / block;
    gelu_backward_kernel<<<grid, block>>>(dy.data, x.data, dx.data, n);
    CUDA_CHECK_KERNEL();
}

// Elementwise add: y = a + b. (Used for residual connections.)
__global__ void add_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

void add(Tensor& y, const Tensor& a, const Tensor& b) {
    if (a.numel != b.numel || a.numel != y.numel) {
        std::fprintf(stderr, "add: shape mismatch\n");
        std::abort();
    }
    int n = (int)a.numel;
    int block = 256;
    int grid = (n + block - 1) / block;
    add_kernel<<<grid, block>>>(a.data, b.data, y.data, n);
    CUDA_CHECK_KERNEL();
}