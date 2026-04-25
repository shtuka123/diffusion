#include "gpt_tiny.h"
#include "cuda_utils.h"
#include "cross_entropy.h"
#include "tensor.h"
#include <cstdio>
#include <vector>
#include <random>

int main() {
    std::printf("=== GPT-tiny smoke test ===\n");

    // Build a small model
    GPTTinyConfig cfg;
    cfg.vocab_size  = 65;
    cfg.max_seq_len = 32;
    cfg.d_model     = 64;
    cfg.n_layers    = 2;
    cfg.n_heads     = 4;
    cfg.d_mlp       = 256;

    int B = 2, T = 16;

    GPTTiny model;
    model.build(cfg, B, T, 42);
    std::printf("  parameters: %zu (~%.2f MB)\n",
                model.param_count(),
                model.param_count() * 4.0 / (1024*1024));

    // Random input tokens
    std::vector<int> tokens(B * T);
    std::mt19937 rng(123);
    for (int& t : tokens) t = rng() % cfg.vocab_size;
    int* d_tokens = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tokens, B * T * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, tokens.data(), B * T * sizeof(int),
                          cudaMemcpyHostToDevice));

    // ----- Forward -----
    model.forward(d_tokens, B, T);
    std::printf("  forward complete. logits shape: (%d, %d, %d)\n",
                model.logits.size(0), model.logits.size(1), model.logits.size(2));

    // Sanity check on logit magnitudes — should be O(1) right after init.
    auto h_logits = model.logits.to_host();
    float lo = h_logits[0], hi = h_logits[0];
    for (float v : h_logits) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    std::printf("  logit range: [%.3f, %.3f]\n", lo, hi);
    if (std::isnan(lo) || std::isnan(hi)) {
        std::fprintf(stderr, "FAIL: NaN in logits\n");
        return 1;
    }
    if (lo < -100.f || hi > 100.f) {
        std::fprintf(stderr, "WARN: logit magnitudes seem large\n");
    }

    // ----- Compute CE loss -----
    Tensor per_row({B * T});
    Tensor loss({1});
    // Treat logits (B, T, V) as (B*T, V) for cross_entropy_forward.
    int V = cfg.vocab_size;
    model.logits.shape = {B * T, V};
    cross_entropy_forward(loss, per_row, model.logits, d_tokens, B * T, V);
    auto h_loss = loss.to_host();
    std::printf("  cross-entropy loss: %.4f (expected ~%.4f for uniform)\n",
                h_loss[0], std::log((float)V));

    // For GPT, we'd predict tokens[t+1] from tokens[t], but with random labels
    // any value near log(V) is fine here — we're just checking the pipeline.
    model.logits.shape = {B, T, V};

    // ----- Backward -----
    Tensor dlogits({B, T, V});
    {
        // dlogits = (softmax(logits) - one_hot(tokens)) / (B*T)
        // We have a cross_entropy_backward for this exactly.
        dlogits.shape = {B * T, V};
        model.logits.shape = {B * T, V};
        cross_entropy_backward(dlogits, model.logits, d_tokens, B * T, V);
        dlogits.shape = {B, T, V};
        model.logits.shape = {B, T, V};
    }

    model.backward(dlogits, d_tokens, B, T);

    // Sanity check: every parameter's grad should be non-zero (roughly).
    auto params = model.params();
    int zero_grads = 0;
    for (auto* p : params) {
        auto h = p->grad.to_host();
        bool all_zero = true;
        for (float v : h) {
            if (v != 0.f && !std::isnan(v)) { all_zero = false; break; }
        }
        if (all_zero) ++zero_grads;
    }
    std::printf("  parameters with all-zero gradients: %d / %zu\n",
                zero_grads, params.size());

    cudaFree(d_tokens);

    if (zero_grads > 0) {
        std::fprintf(stderr, "FAIL: some parameters have no gradient\n");
        return 1;
    }

    std::printf("\nGPT-tiny smoke test PASS.\n");
    return 0;
}