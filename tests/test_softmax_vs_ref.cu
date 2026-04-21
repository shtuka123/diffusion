#include "tensor.h"
#include "tensor_io.h"
#include "softmax.h"
#include <cstdio>
#include <cmath>
#include <string>

int test_case(const std::string& name, const std::string& x_path,
              const std::string& y_path, float tol = 1e-5f) {
    std::printf("=== %s ===\n", name.c_str());
    Tensor x = load_tensor(x_path);
    Tensor expected = load_tensor(y_path);
    Tensor y(x.shape);

    softmax_forward(y, x);

    // Check for NaN or inf in output
    auto h = y.to_host();
    for (size_t i = 0; i < h.size(); ++i) {
        if (std::isnan(h[i]) || std::isinf(h[i])) {
            std::fprintf(stderr, "  FAIL: NaN or inf at index %zu\n", i);
            return 1;
        }
    }

    // Check each row sums to ~1.0 (invariant of softmax)
    int B = y.size(0), C = y.size(1);
    for (int b = 0; b < B; ++b) {
        float sum = 0.f;
        for (int c = 0; c < C; ++c) sum += h[b * C + c];
        if (std::fabs(sum - 1.0f) > 1e-4f) {
            std::fprintf(stderr, "  FAIL: row %d sum=%f, expected 1.0\n", b, sum);
            return 1;
        }
    }

    float err = max_abs_diff(y, expected);
    std::printf("  max abs error vs PyTorch: %.3e\n", err);
    if (err < tol) {
        std::printf("  PASS\n");
        return 0;
    } else {
        std::fprintf(stderr, "  FAIL (tol=%.0e)\n", tol);
        return 1;
    }
}

int main() {
    int fails = 0;
    fails += test_case("normal logits",  "data/ref/softmax_normal_x.bin",
                                         "data/ref/softmax_normal_y.bin");
    fails += test_case("large logits",   "data/ref/softmax_large_x.bin",
                                         "data/ref/softmax_large_y.bin");
    fails += test_case("extreme logits", "data/ref/softmax_extreme_x.bin",
                                         "data/ref/softmax_extreme_y.bin");
    fails += test_case("wider class dim","data/ref/softmax_wide_x.bin",
                                         "data/ref/softmax_wide_y.bin");

    if (fails == 0) {
        std::printf("\nAll softmax tests passed.\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n%d test(s) failed.\n", fails);
        return 1;
    }
}