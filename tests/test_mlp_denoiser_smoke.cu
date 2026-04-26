#include "mlp_denoiser.h"
#include "diffusion.h"
#include "tensor.h"
#include "cuda_utils.h"

#include <cstdio>
#include <vector>
#include <random>

int main() {
    std::printf("=== MLP denoiser smoke test ===\n");

    MLPDenoiserConfig cfg;
    cfg.input_dim = 784;
    cfg.time_dim = 128;
    cfg.hidden_dim = 256;
    cfg.n_layers = 4;

    int B = 16;

    MLPDenoiser model;
    model.build(cfg, B, 42);
    std::printf("  parameters: %zu (~%.2f MB)\n",
                model.param_count(),
                model.param_count() * 4.0 / (1024 * 1024));

    // Random input: noisy "images" and timesteps
    Tensor x_t = Tensor::randn({B, cfg.input_dim}, 1);
    std::vector<int> h_ts(B);
    std::mt19937 rng(123);
    for (int& t : h_ts) t = rng() % 1000;
    int* d_ts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ts, B * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ts, h_ts.data(), B * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Forward
    model.forward(x_t, d_ts, nullptr, B);
    std::printf("  forward complete. output shape: (%d, %d)\n",
                model.output.size(0), model.output.size(1));

    // Sanity: output should be O(1)-magnitude (network is randomly initialized,
    // so it produces random output of similar variance to its input).
    auto h_out = model.output.to_host();
    float lo = h_out[0], hi = h_out[0];
    for (float v : h_out) {
        if (std::isnan(v)) { std::fprintf(stderr, "FAIL: NaN in output\n"); return 1; }
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    std::printf("  output range: [%.3f, %.3f]\n", lo, hi);

    // MSE loss test: noise target = random
    Tensor target = Tensor::randn({B, cfg.input_dim}, 99);
    Tensor loss({1});
    mse_loss_forward(loss, model.output, target);
    std::printf("  initial MSE loss: %.4f\n", loss.to_host()[0]);

    // Backward
    Tensor dy({B, cfg.input_dim});
    mse_loss_backward(dy, model.output, target);
    model.backward(dy, x_t, d_ts, nullptr, B);

    // Verify all parameter gradients are non-zero.
    auto params = model.params();
    int zero_grads = 0;
    for (auto* p : params) {
        auto h = p->grad.to_host();
        bool all_zero = true;
        for (float v : h) {
            if (std::isnan(v)) {
                std::fprintf(stderr, "FAIL: NaN in gradient\n");
                cudaFree(d_ts); return 1;
            }
            if (v != 0.f) { all_zero = false; break; }
        }
        if (all_zero) ++zero_grads;
    }
    std::printf("  parameters with all-zero gradients: %d / %zu\n",
                zero_grads, params.size());

    cudaFree(d_ts);

    if (zero_grads > 0) {
        std::fprintf(stderr, "FAIL\n");
        return 1;
    }
    std::printf("\nMLP denoiser smoke test PASS.\n");
    return 0;
}