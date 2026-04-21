#include "tensor.h"
#include "tensor_io.h"
#include "matmul.h"
#include <cstdio>

int main() {
    std::printf("Loading reference tensors...\n");
    Tensor A = load_tensor("data/ref/matmul_A.bin");
    Tensor B = load_tensor("data/ref/matmul_B.bin");
    Tensor expected = load_tensor("data/ref/matmul_C.bin");
    std::printf("  A shape = %s\n", A.shape_str().c_str());
    std::printf("  B shape = %s\n", B.shape_str().c_str());
    std::printf("  expected C shape = %s\n", expected.shape_str().c_str());

    Tensor C({A.size(0), B.size(1)});
    matmul(C, A, B);

    float err = max_abs_diff(C, expected);
    std::printf("\nMax absolute error: %.3e\n", err);

    // For float32 matmul at K=256, ~1e-4 is normal.
    constexpr float TOL = 1e-3f;
    if (err < TOL) {
        std::printf("PASS (under tolerance %.0e)\n", TOL);
        return 0;
    } else {
        std::fprintf(stderr, "FAIL: error %.3e exceeds tolerance %.0e\n", err, TOL);
        return 1;
    }
}