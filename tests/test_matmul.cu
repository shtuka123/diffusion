#include "matmul.h"
#include "tensor.h"
#include <cstdio>
#include <cmath>

static bool close_enough(float a, float b, float tol = 1e-4f) {
    return std::abs(a - b) < tol;
}

int main() {
    // ===== Test 1: Hand-computed 2x2 * 2x2 =====
    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    // Expected C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //            = [[19, 22], [43, 50]]
    {
        Tensor A({2, 2}); A.from_host({1, 2, 3, 4});
        Tensor B({2, 2}); B.from_host({5, 6, 7, 8});
        Tensor C({2, 2});
        matmul(C, A, B);
        auto h = C.to_host();
        std::vector<float> expected = {19, 22, 43, 50};
        for (int i = 0; i < 4; ++i) {
            if (!close_enough(h[i], expected[i])) {
                std::fprintf(stderr, "FAIL test 1: C[%d] = %f, expected %f\n",
                             i, h[i], expected[i]);
                return 1;
            }
        }
        std::printf("Test 1 (hand-computed 2x2): PASS\n");
    }

    // ===== Test 2: Multiply by identity =====
    // A @ I = A. Spot-check on a 3x3.
    {
        Tensor A({3, 3}); A.from_host({1, 2, 3, 4, 5, 6, 7, 8, 9});
        Tensor I({3, 3}); I.from_host({1, 0, 0, 0, 1, 0, 0, 0, 1});
        Tensor C({3, 3});
        matmul(C, A, I);
        auto h = C.to_host();
        auto a = A.to_host();
        for (size_t i = 0; i < h.size(); ++i) {
            if (!close_enough(h[i], a[i])) {
                std::fprintf(stderr, "FAIL test 2: C[%zu] = %f, expected %f\n",
                             i, h[i], a[i]);
                return 1;
            }
        }
        std::printf("Test 2 (A @ I = A): PASS\n");
    }

    // ===== Test 3: Non-square shapes =====
    // (3 x 4) @ (4 x 5) should produce (3 x 5)
    {
        Tensor A = Tensor::randn({3, 4}, 1);
        Tensor B = Tensor::randn({4, 5}, 2);
        Tensor C({3, 5});
        matmul(C, A, B);
        // Spot-check C[0, 0] = sum_k A[0, k] * B[k, 0]
        auto a = A.to_host();
        auto b = B.to_host();
        auto c = C.to_host();
        float expected = 0.f;
        for (int k = 0; k < 4; ++k) expected += a[0 * 4 + k] * b[k * 5 + 0];
        if (!close_enough(c[0], expected)) {
            std::fprintf(stderr, "FAIL test 3: C[0,0] = %f, expected %f\n",
                         c[0], expected);
            return 1;
        }
        std::printf("Test 3 (non-square 3x4 @ 4x5): PASS  (C[0,0] = %f)\n", c[0]);
    }

    // ===== Test 4: Larger size, just check it doesn't crash =====
    {
        Tensor A = Tensor::randn({128, 256}, 3);
        Tensor B = Tensor::randn({256, 64}, 4);
        Tensor C({128, 64});
        matmul(C, A, B);
        auto c = C.to_host();
        // Sanity: not all zeros, no NaN, no inf
        bool all_zero = true, any_nan = false;
        for (float v : c) {
            if (v != 0.f) all_zero = false;
            if (std::isnan(v) || std::isinf(v)) any_nan = true;
        }
        if (all_zero || any_nan) {
            std::fprintf(stderr, "FAIL test 4: result has all-zero=%d, nan/inf=%d\n",
                         all_zero, any_nan);
            return 1;
        }
        std::printf("Test 4 (128x256 @ 256x64): PASS  (C[0,0] = %f)\n", c[0]);
    }

    std::printf("\nAll matmul tests passed.\n");
    return 0;
}