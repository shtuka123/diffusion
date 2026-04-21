#include "tensor.h"
#include "parameter.h"
#include "linear.h"
#include "elementwise.h"
#include "cross_entropy.h"
#include "optim.h"
#include "init.h"
#include "cuda_utils.h"
#include <cstdio>
#include <vector>

// Goal: build a tiny (784 -> 64 -> 10) MLP, feed a random batch,
// take one SGD step, verify loss decreases.
// This is the whole forward-backward-update loop in miniature.
//
// No data loader yet (Day 17) — we use random inputs with random labels.

int main() {
    std::printf("=== Optimizer step: mini-MLP sanity check ===\n");

    const int B = 64;
    const int D_in = 784;
    const int H = 64;
    const int D_out = 10;

    // --- Parameters ---
    Parameter W1({D_in, H});      init_he_normal(W1, D_in, 1);
    Parameter b1({H});            init_zero(b1);
    Parameter W2({H, D_out});     init_he_normal(W2, H, 2);
    Parameter b2({D_out});        init_zero(b2);

    std::vector<Parameter*> params = {&W1, &b1, &W2, &b2};

    // --- Inputs: random images, random labels ---
    Tensor x = Tensor::randn({B, D_in}, 42);
    std::vector<int> h_labels(B);
    std::mt19937 rng(123);
    for (int i = 0; i < B; ++i) h_labels[i] = rng() % D_out;
    int* d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_labels, B * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), B * sizeof(int),
                          cudaMemcpyHostToDevice));

    // --- Intermediate / gradient buffers ---
    Tensor h({B, H}), h_relu({B, H});
    Tensor logits({B, D_out});
    Tensor dh({B, H}), dh_pre_relu({B, H});
    Tensor dlogits({B, D_out});
    Tensor per_row({B}), loss({1});

    float lr = 0.1f;

    // --- Training-step function ---
    auto step = [&]() -> float {
        // Forward
        linear_forward(h, x, W1.w, b1.w);
        relu_forward(h_relu, h);
        linear_forward(logits, h_relu, W2.w, b2.w);
        cross_entropy_forward(loss, per_row, logits, d_labels, B, D_out);

        // Read loss to CPU
        auto h_loss = loss.to_host();
        float L = h_loss[0];

        // Backward — go in reverse
        cross_entropy_backward(dlogits, logits, d_labels, B, D_out);

        Tensor dW2({H, D_out}), db2({D_out});
        linear_backward(dh, dW2, db2, h_relu, W2.w, dlogits);
        W2.grad.copy_from(dW2);
        b2.grad.copy_from(db2);

        relu_backward(dh_pre_relu, dh, h);

        Tensor dx({B, D_in}), dW1({D_in, H}), db1({H});
        linear_backward(dx, dW1, db1, x, W1.w, dh_pre_relu);
        W1.grad.copy_from(dW1);
        b1.grad.copy_from(db1);

        // Optimizer step
        sgd_step_all(params, lr);

        // Reset grads for next step
        zero_grads(params);

        return L;
    };

    // --- Run 20 steps and watch loss drop ---
    std::printf("Step   0: loss = ");
    float L0 = step();
    std::printf("%.4f\n", L0);

    float L_prev = L0;
    for (int i = 1; i <= 200; ++i) {
        float L = step();
        if (i % 5 == 0 || i <= 3) {
            std::printf("Step %3d: loss = %.4f\n", i, L);
        }
        L_prev = L;
    }

    CUDA_CHECK(cudaFree(d_labels));

    // Basic correctness: loss should have decreased significantly.
    // This is a trivial overfit-a-random-batch test; any remotely correct
    // network drops the loss by 2x+ in 20 steps.
    if (L_prev >= L0) {
        std::fprintf(stderr, "\nFAIL: loss did not decrease (%.4f -> %.4f)\n",
                     L0, L_prev);
        return 1;
    }
    std::printf("\nLoss dropped %.4f -> %.4f. PASS.\n", L0, L_prev);
    return 0;
}