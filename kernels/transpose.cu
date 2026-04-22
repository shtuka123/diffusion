#include "cuda_utils.h"
#include "tensor.h"

// Transpose dims 1 and 2 of a 4D tensor:
// (d0, d1, d2, d3) -> (d0, d2, d1, d3)
// Used for attention: (B, T, H, dk) <-> (B, H, T, dk).
__global__ void transpose_12_kernel(
    const float* x, float* y,
    int d0, int d1, int d2, int d3)
{
    // One thread per output element.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2 * d3;
    if (tid >= total) return;

    // Decompose tid into output coordinates (i0, i2, i1, i3)
    // (remember: output is d0, d2, d1, d3 — dim1 and dim2 swapped)
    int i3 = tid % d3;
    int i1 = (tid / d3) % d1;
    int i2 = (tid / (d3 * d1)) % d2;
    int i0 = tid / (d3 * d1 * d2);

    // Input offset: (i0, i1, i2, i3) in original layout
    int src = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
    y[tid] = x[src];
}

void transpose_12(Tensor& y, const Tensor& x) {
    if (x.ndim() != 4 || y.ndim() != 4) {
        std::fprintf(stderr, "transpose_12: expected 4D tensors\n");
        std::abort();
    }
    int d0 = x.size(0), d1 = x.size(1), d2 = x.size(2), d3 = x.size(3);
    if (y.size(0) != d0 || y.size(1) != d2 || y.size(2) != d1 || y.size(3) != d3) {
        std::fprintf(stderr, "transpose_12: output shape mismatch\n");
        std::abort();
    }
    int total = d0 * d1 * d2 * d3;
    int block = 256;
    int grid = (total + block - 1) / block;
    transpose_12_kernel<<<grid, block>>>(x.data, y.data, d0, d1, d2, d3);
    CUDA_CHECK_KERNEL();
}