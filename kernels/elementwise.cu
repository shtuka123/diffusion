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