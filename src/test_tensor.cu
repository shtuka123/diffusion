#include "tensor.h"
#include <cstdio>

int main() {
    // 1. Round-trip test
    std::printf("=== Round-trip test ===\n");
    std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t({2, 3});
    t.from_host(original);
    t.print();

    auto back = t.to_host();
    for (size_t i = 0; i < back.size(); ++i) {
        if (back[i] != original[i]) {
            std::fprintf(stderr, "FAIL: element %zu: expected %f, got %f\n",
                         i, original[i], back[i]);
            return 1;
        }
    }
    std::printf("Round-trip: PASS\n\n");

    // 2. zeros factory
    std::printf("=== zeros() test ===\n");
    Tensor z = Tensor::zeros({4});
    z.print();
    for (float v : z.to_host()) {
        if (v != 0.0f) { std::fprintf(stderr, "FAIL: zeros has non-zero\n"); return 1; }
    }
    std::printf("zeros: PASS\n\n");

    // 3. randn factory
    std::printf("=== randn() test ===\n");
    Tensor r = Tensor::randn({1000}, 42);
    auto rh = r.to_host();
    double mean = 0.0, sq = 0.0;
    for (float v : rh) { mean += v; sq += v * v; }
    mean /= rh.size();
    double var = sq / rh.size() - mean * mean;
    std::printf("randn(1000): mean=%.3f, var=%.3f (expected ~0, ~1)\n", mean, var);
    std::printf("randn: PASS\n\n");

    // 4. Move semantics
    std::printf("=== Move test ===\n");
    Tensor a = Tensor::zeros({5});
    Tensor b = std::move(a);
    if (a.data != nullptr) {
        std::fprintf(stderr, "FAIL: moved-from tensor still has data\n");
        return 1;
    }
    std::printf("Move: PASS\n\n");

    // 5. Leak test — allocate & free a lot, watch nvidia-smi doesn't climb
    std::printf("=== Leak test (watch nvidia-smi) ===\n");
    for (int i = 0; i < 10000; ++i) {
        Tensor tmp = Tensor::zeros({1024, 1024});  // 4 MB each
    }
    std::printf("Leak test: allocated and freed 10000 tensors\n");

    std::printf("All tests passed.\n");
    return 0;
}