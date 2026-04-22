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

// C = A @ B^T
//   A: (M, K), B: (N, K) — B is treated as if transposed, so the effective
//   second operand has shape (K, N) and the product is (M, N).
//   C[m, n] = sum_k A[m, k] * B[n, k]
__global__ void matmul_at_b_t_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[n * K + k];   // note the B indexing: [n, k], not [k, n]
    }
    C[m * N + n] = acc;
}

// Host wrapper. A: (M, K), B: (N, K). Output C: (M, N).
void matmul_nt(Tensor& C, const Tensor& A, const Tensor& B) {
    int M = A.size(0), K = A.size(1);
    int N = B.size(0);
    if (B.size(1) != K) {
        std::fprintf(stderr, "matmul_nt: inner dims mismatch (A=%d, B=%d)\n",
                     K, B.size(1));
        std::abort();
    }
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_at_b_t_kernel<<<grid, block>>>(A.data, B.data, C.data, M, K, N);
    CUDA_CHECK_KERNEL();
}

// Raw-pointer entry points used by ops that need to matmul on a sub-slice
// of a larger buffer (e.g., per-batch in attention) without constructing
// views into Tensor.

void matmul_raw(float* C, const float* A, const float* B,
                int M, int K, int N)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_naive_kernel<<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK_KERNEL();
}

// A @ B^T with A: (M, K), B: (N, K), output C: (M, N)
void matmul_nt_raw(float* C, const float* A, const float* B,
                   int M, int K, int N)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_at_b_t_kernel<<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK_KERNEL();
}

// C = A^T @ B
//   A: (K, M), B: (K, N) — A is treated as if transposed, so the effective
//   first operand has shape (M, K) and the product is (M, N).
//   C[m, n] = sum_k A[k, m] * B[k, n]
__global__ void matmul_a_t_b_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += A[k * M + m] * B[k * N + n];
    }
    C[m * N + n] = acc;
}

// A^T @ B. A: (K, M), B: (K, N). Output C: (M, N).
void matmul_tn_raw(float* C, const float* A, const float* B,
                   int M, int K, int N)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_a_t_b_kernel<<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK_KERNEL();
}