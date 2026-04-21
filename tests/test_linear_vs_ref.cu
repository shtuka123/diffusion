#include "tensor.h"
#include "tensor_io.h"
#include "linear.h"
#include <cstdio>

int main() {
    std::printf("=== Linear forward vs PyTorch ===\n");

    Tensor x = load_tensor("data/ref/linear_x.bin");
    Tensor W = load_tensor("data/ref/linear_W.bin");
    Tensor b = load_tensor("data/ref/linear_b.bin");
    Tensor expected_y = load_tensor("data/ref/linear_y.bin");

    int B = x.size(0), D_out = W.size(1);
    Tensor y({B, D_out});
    linear_forward(y, x, W, b);

    float err = max_abs_diff(y, expected_y);
    std::printf("  max abs error: %.3e\n", err);
    if (err < 1e-4f) {
        std::printf("  PASS\n");
        return 0;
    } else {
        std::fprintf(stderr, "  FAIL\n");
        return 1;
    }
}