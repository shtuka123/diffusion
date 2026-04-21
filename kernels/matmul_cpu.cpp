#include "tensor.h"
#include <vector>

// Pure CPU matmul, same algorithm as the naive CUDA kernel.
// Operates entirely on host-side data.
void matmul_cpu(std::vector<float>& C,
                const std::vector<float>& A,
                const std::vector<float>& B,
                int M, int K, int N)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}