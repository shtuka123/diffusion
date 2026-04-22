#include "tensor.h"
#include "tensor_io.h"
#include "layernorm.h"
#include <cstdio>

int main() {
    std::printf("=== LayerNorm forward vs PyTorch ===\n");

    Tensor x     = load_tensor("data/ref/ln_x.bin");
    Tensor gamma = load_tensor("data/ref/ln_gamma.bin");
    Tensor beta  = load_tensor("data/ref/ln_beta.bin");
    Tensor expected_y = load_tensor("data/ref/ln_y.bin");

    int B = x.size(0), T = x.size(1), D = x.size(2);
    int N = B * T;

    Tensor y(x.shape);
    Tensor rstd({N});

    layernorm_forward(y, rstd, x, gamma, beta);

    float err = max_abs_diff(y, expected_y);
    std::printf("  shape: (%d, %d, %d), N=%d, D=%d\n", B, T, D, N, D);
    std::printf("  max abs error: %.3e\n", err);

    if (err < 1e-5f) {
        std::printf("  PASS\n");
        return 0;
    } else {
        std::fprintf(stderr, "  FAIL\n");
        return 1;
    }
}