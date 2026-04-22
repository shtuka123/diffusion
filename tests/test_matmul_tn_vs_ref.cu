#include "tensor.h"
#include "tensor_io.h"
#include "matmul.h"
#include "cuda_utils.h"
#include <cstdio>

int main() {
    std::printf("=== matmul_tn (A^T @ B) vs PyTorch ===\n");

    Tensor A = load_tensor("data/ref/matmul_tn_A.bin");  // (K, M)
    Tensor B = load_tensor("data/ref/matmul_tn_B.bin");  // (K, N)
    Tensor expected_C = load_tensor("data/ref/matmul_tn_C.bin");  // (M, N)

    int K = A.size(0), M = A.size(1);
    int N = B.size(1);

    Tensor C({M, N});
    matmul_tn_raw(C.data, A.data, B.data, M, K, N);

    float err = max_abs_diff(C, expected_C);
    std::printf("  shapes: K=%d M=%d N=%d\n", K, M, N);
    std::printf("  max abs error: %.3e\n", err);
    if (err < 1e-4f) {
        std::printf("  PASS\n");
        return 0;
    } else {
        std::fprintf(stderr, "  FAIL\n");
        return 1;
    }
}