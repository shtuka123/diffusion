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