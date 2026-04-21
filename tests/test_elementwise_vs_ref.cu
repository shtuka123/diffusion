#include "tensor.h"
#include "tensor_io.h"
#include "elementwise.h"
#include <cstdio>

int test_relu() {
    std::printf("=== ReLU ===\n");
    Tensor x = load_tensor("data/ref/relu_x.bin");
    Tensor expected = load_tensor("data/ref/relu_y.bin");
    Tensor y(x.shape);

    relu_forward(y, x);

    float err = max_abs_diff(y, expected);
    std::printf("  max abs error: %.3e\n", err);
    if (err < 1e-6f) {
        std::printf("  PASS\n");
        return 0;
    } else {
        std::printf("  FAIL\n");
        return 1;
    }
}

int test_bias_add() {
    std::printf("=== bias_add ===\n");
    Tensor x = load_tensor("data/ref/bias_add_x.bin");
    Tensor bias = load_tensor("data/ref/bias_add_bias.bin");
    Tensor expected = load_tensor("data/ref/bias_add_y.bin");
    Tensor y(x.shape);

    bias_add(y, x, bias);

    float err = max_abs_diff(y, expected);
    std::printf("  max abs error: %.3e\n", err);
    if (err < 1e-6f) {
        std::printf("  PASS\n");
        return 0;
    } else {
        std::printf("  FAIL\n");
        return 1;
    }
}

int main() {
    int fails = 0;
    fails += test_relu();
    fails += test_bias_add();
    if (fails == 0) {
        std::printf("\nAll elementwise tests passed.\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n%d test(s) failed.\n", fails);
        return 1;
    }
}