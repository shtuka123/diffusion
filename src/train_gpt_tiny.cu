// Train GPT-tiny on tiny-shakespeare with SGD + cross-entropy.
// No new kernels.

#include "tensor.h"
#include "parameter.h"
#include "gpt_tiny.h"
#include "cross_entropy.h"
#include "optim.h"
#include "shakespeare.h"
#include "cuda_utils.h"

#include <cstdio>
#include <vector>
#include <random>

int main() {
    // ----- Config -----
    GPTTinyConfig cfg;
    cfg.vocab_size  = 65;       // tiny-shakespeare's char vocab
    cfg.max_seq_len = 64;
    cfg.d_model     = 128;
    cfg.n_layers    = 4;
    cfg.n_heads     = 4;
    cfg.d_mlp       = 512;

    int B = 16;
    int T = 64;

    int n_steps = 1000;
    float lr = 0.1f;
    int log_every = 50;
    uint64_t seed = 42;

    // ----- Data -----
    Shakespeare data("data/tinyshakespeare.txt");
    if (data.vocab_size != cfg.vocab_size) {
        std::fprintf(stderr, "vocab mismatch: data=%d, cfg=%d\n",
                     data.vocab_size, cfg.vocab_size);
        return 1;
    }

    // ----- Model -----
    GPTTiny model;
    model.build(cfg, B, T, seed);
    std::printf("Model built: %zu parameters (~%.2f MB)\n\n",
                model.param_count(), model.param_count() * 4.0 / (1024*1024));

    auto params = model.params();

    // ----- Buffers -----
    int* d_x = nullptr;
    int* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, B * T * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, B * T * sizeof(int)));

    Tensor per_row({B * T});
    Tensor loss({1});
    Tensor dlogits({B, T, cfg.vocab_size});

    std::mt19937 rng(seed + 99);

    // ----- Training loop -----
    std::printf("Training for %d steps, B=%d T=%d lr=%.3f\n", n_steps, B, T, lr);
    std::printf("---------------------------------------------\n");

    float running_loss = 0.f;
    int running_count = 0;

    for (int step = 0; step < n_steps; ++step) {
        data.get_random_batch(B, T, d_x, d_y, rng);

        // ----- Forward -----
        model.forward(d_x, B, T);

        // ----- Loss: CE between logits and shifted targets -----
        // Treat logits as (B*T, V) for cross-entropy.
        model.logits.shape = {B * T, cfg.vocab_size};
        cross_entropy_forward(loss, per_row, model.logits, d_y, B * T, cfg.vocab_size);

        auto h_loss = loss.to_host();
        running_loss += h_loss[0];
        running_count++;

        // ----- Backward -----
        dlogits.shape = {B * T, cfg.vocab_size};
        cross_entropy_backward(dlogits, model.logits, d_y, B * T, cfg.vocab_size);
        model.logits.shape = {B, T, cfg.vocab_size};
        dlogits.shape = {B, T, cfg.vocab_size};

        model.backward(dlogits, d_x, B, T);

        // ----- Optimizer step -----
        sgd_step_all(params, lr);
        zero_grads(params);

        // ----- Report -----
        if ((step + 1) % log_every == 0) {
            float avg = running_loss / running_count;
            std::printf("  step %4d / %d  loss %.4f\n", step + 1, n_steps, avg);
            running_loss = 0.f;
            running_count = 0;
        }
    }

    std::printf("\nTraining complete.\n");
    model.save("checkpoints/gpt_tiny.bin");
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}