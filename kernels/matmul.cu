#include "cuda_utils.h"
#include "tensor.h"

// Naive matmul: one thread per output element.
// A: (M, K) row-major
// B: (K, N) row-major
// C: (M, N) row-major
__global__ void matmul_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check — the grid may be slightly larger than M x N
    if (m >= M || n >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[k * N + n];
    }
    C[m * N + n] = acc;
}

// Host-side wrapper. Validates shapes, sets grid/block dims, launches.
void matmul(Tensor& C, const Tensor& A, const Tensor& B) {
    // Shape checks — fail loud if something's wrong
    if (A.ndim() != 2 || B.ndim() != 2 || C.ndim() != 2) {
        std::fprintf(stderr, "matmul: all tensors must be 2D\n");
        std::abort();
    }
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    if (B.size(0) != K) {
        std::fprintf(stderr, "matmul: A is (%d, %d), B is (%d, %d) -- inner dims don't match\n",
                     M, K, B.size(0), N);
        std::abort();
    }
    if (C.size(0) != M || C.size(1) != N) {
        std::fprintf(stderr, "matmul: C must be (%d, %d), got (%d, %d)\n",
                     M, N, C.size(0), C.size(1));
        std::abort();
    }

    // 16x16 = 256 threads per block (a safe occupancy choice)
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    matmul_naive_kernel<<<grid, block>>>(A.data, B.data, C.data, M, K, N);
    CUDA_CHECK_KERNEL();
}