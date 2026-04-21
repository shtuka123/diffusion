#include "tensor.h"
#include "grad_check.h"
#include "cross_entropy.h"
#include "cuda_utils.h"
#include <cstdio>
#include <vector>

int main() {
    std::printf("=== Cross-entropy backward: grad check ===\n");

    // Tiny: B=2, C=3
    int B = 2, C = 3;
    Tensor logits({B, C});
    logits.from_host({0.5f, -1.2f, 2.1f, -0.8f, 1.5f, 0.3f});

    std::vector<int> h_labels = {2, 0};  // row 0 true = class 2, row 1 true = class 0
    int* d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_labels, B * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), B * sizeof(int),
                          cudaMemcpyHostToDevice));

    Tensor per_row({(int)B});
    Tensor loss({1});

    // Forward closure: takes logits, returns scalar mean CE loss
    auto forward = [&](Tensor& x_in) -> float {
        cross_entropy_forward(loss, per_row, x_in, d_labels, B, C);
        auto h = loss.to_host();
        return h[0];
    };

    // Analytic gradient via our backward kernel
    Tensor dlogits({B, C});
    cross_entropy_backward(dlogits, logits, d_labels, B, C);

    float err = grad_check(forward, logits, dlogits, 1e-3f, true);
    std::printf("\nMax relative error: %.3e\n", err);

    cudaFree(d_labels);

    if (err < 1e-2f) {
        std::printf("PASS\n");
        return 0;
    } else {
        std::fprintf(stderr, "FAIL\n");
        return 1;
    }
}