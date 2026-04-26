// Train GPT-tiny on tiny-shakespeare with SGD + cross-entropy.
// No new kernels.

#include "tensor.h"
#include "parameter.h"
#include "gpt_tiny.h"
#include "cross_entropy.h"
#include "optim.h"
#include "shakespeare.h"
#include "cuda_utils.h"
#include "scheduler.h"
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

    int n_steps = 2000;
    AdamState adam;
    adam.beta1 = 0.9f;
    adam.beta2 = 0.999f;
    adam.eps = 1e-8f;
    int log_every = 50;
    uint64_t seed = 42;

    // LR schedule
    float lr_max = 3e-3f;     // peak after warmup
    float lr_min = 1e-4f;     // floor for long tail
    int warmup_steps = std::min(100, n_steps / 10);  // ~10% warmup

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
    std::printf("Training for %d steps, B=%d T=%d lr=%.3f\n", n_steps, B, T, adam.lr);
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
        adam.lr = warmup_cosine_lr(step, warmup_steps, n_steps, lr_max, lr_min);
        adam_step_all(params, adam);
        zero_grads(params);

        // ----- Report -----
        int n_steps = 2000;   // was 1000
// ...
if ((step + 1) % log_every == 0) {
    float avg = running_loss / running_count;
    std::printf("  step %4d / %d  lr=%.5f  loss %.4f\n",
                step + 1, n_steps, adam.lr, avg);
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