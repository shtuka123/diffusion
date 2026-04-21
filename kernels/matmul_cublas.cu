#include "cuda_utils.h"
#include "tensor.h"
#include <cublas_v2.h>
#include <cstdio>

// Same CHECK pattern but for cuBLAS, which has its own error type.
#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t status = (call);                                     \
    if (status != CUBLAS_STATUS_SUCCESS) {                              \
        std::fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n",      \
                     __FILE__, __LINE__, (int)status);                  \
        std::abort();                                                   \
    }                                                                   \
} while (0)

// Singleton handle. cuBLAS handle creation is expensive; reuse one.
static cublasHandle_t g_handle = nullptr;
static cublasHandle_t get_handle() {
    if (!g_handle) CUBLAS_CHECK(cublasCreate(&g_handle));
    return g_handle;
}

// Row-major C = A @ B using a column-major library.
// Trick: cuBLAS sees our row-major (M,K) as column-major (K,M) which is A^T.
// So we ask cuBLAS for (B^T) @ (A^T) = (A @ B)^T, and the column-major output
// of (A@B)^T has the SAME bytes as the row-major output of A@B.
// Concretely: just call sgemm with B and A swapped, no transpose flags.
void matmul_cublas(Tensor& C, const Tensor& A, const Tensor& B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);

    const float alpha = 1.f, beta = 0.f;
    auto h = get_handle();
    CUBLAS_CHECK(cublasSgemm(
        h,
        CUBLAS_OP_N, CUBLAS_OP_N,   // no transpose
        N, M, K,                    // dims swapped: cuBLAS sees (N,K) @ (K,M)
        &alpha,
        B.data, N,                  // B with leading dim = N
        A.data, K,                  // A with leading dim = K
        &beta,
        C.data, N                   // C with leading dim = N
    ));
}