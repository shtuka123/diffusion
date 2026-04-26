// DDIM sampler with class conditioning + classifier-free guidance.

#include "tensor.h"
#include "mlp_denoiser.h"
#include "noise_schedule.h"
#include "diffusion.h"
#include "cuda_utils.h"

#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

int main(int argc, char** argv) {
    int n_samples = 8;
    int T_diffusion = 1000;
    int n_ddim_steps = 50;
    uint64_t seed = 1234;
    int target_class = -1;       // -1 = unconditional
    float cfg_scale = 1.0f;       // 1.0 = no guidance; >1 amplifies class adherence
    std::string ckpt_path = "checkpoints/diffusion_mlp.bin";

    for (int i = 1; i + 1 < argc; i += 2) {
        std::string flag = argv[i];
        std::string val = argv[i + 1];
        if      (flag == "--n")     n_samples = std::stoi(val);
        else if (flag == "--steps") n_ddim_steps = std::stoi(val);
        else if (flag == "--seed")  seed = std::stoull(val);
        else if (flag == "--class") target_class = std::stoi(val);
        else if (flag == "--cfg")   cfg_scale = std::stof(val);
        else if (flag == "--ckpt")  ckpt_path = val;
    }

    MLPDenoiserConfig model_cfg;
    model_cfg.input_dim  = 784;
    model_cfg.time_dim   = 128;
    model_cfg.hidden_dim = 256;
    model_cfg.n_layers   = 4;
    model_cfg.num_classes = 10;

    int B = n_samples;

    MLPDenoiser model;
    model.build(model_cfg, B, 0);
    model.load(ckpt_path);

    NoiseSchedule sched;
    sched.build(T_diffusion);

    if (target_class >= 0) {
        std::printf("Conditioning on class: %d (cfg_scale=%.2f)\n",
                    target_class, cfg_scale);
    } else {
        std::printf("Unconditional generation (null class)\n");
    }

    bool use_cfg = (cfg_scale != 1.0f) && (target_class >= 0);

    std::vector<int> ts;
    ts.reserve(n_ddim_steps);
    for (int i = 0; i < n_ddim_steps; ++i) {
        int t = (T_diffusion - 1) - (i * (T_diffusion - 1) / (n_ddim_steps - 1));
        ts.push_back(t);
    }

    std::printf("Generating %d MNIST digits with DDIM (%d steps over %d-step schedule)%s...\n",
                n_samples, n_ddim_steps, T_diffusion,
                use_cfg ? ", with CFG" : "");
    std::printf("Timestep sequence: ");
    for (int i = 0; i < std::min(8, (int)ts.size()); ++i) std::printf("%d ", ts[i]);
    std::printf("...\n\n");

    Tensor x_t({B, 784});
    Tensor x_prev({B, 784});
    Tensor x_0_cond({B, 784});
    Tensor x_0_uncond({B, 784});
    Tensor x_0_guided({B, 784});

    int* d_timesteps = nullptr;
    int* d_labels = nullptr;
    int* d_labels_uncond = nullptr;
    CUDA_CHECK(cudaMalloc(&d_timesteps, B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels, B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels_uncond, B * sizeof(int)));

    {
        // Conditional labels (or null class for unconditional)
        std::vector<int> h_labels(B);
        int label_value = (target_class >= 0) ? target_class : model_cfg.num_classes;
        for (int b = 0; b < B; ++b) h_labels[b] = label_value;
        CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), B * sizeof(int),
                              cudaMemcpyHostToDevice));

        // Unconditional labels (always null class)
        std::vector<int> h_uncond(B, model_cfg.num_classes);
        CUDA_CHECK(cudaMemcpy(d_labels_uncond, h_uncond.data(), B * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    x_t = Tensor::randn({B, 784}, seed);

    for (int idx = 0; idx < (int)ts.size(); ++idx) {
        int t = ts[idx];
        int t_prev = (idx == (int)ts.size() - 1) ? -1 : ts[idx + 1];

        std::vector<int> h_t(B, t);
        CUDA_CHECK(cudaMemcpy(d_timesteps, h_t.data(), B * sizeof(int),
                              cudaMemcpyHostToDevice));

        if (use_cfg) {
            // Two forward passes: conditional + unconditional
            model.forward(x_t, d_timesteps, d_labels, B);
            x_0_cond.copy_from(model.output);

            model.forward(x_t, d_timesteps, d_labels_uncond, B);
            x_0_uncond.copy_from(model.output);

            // x_0_guided = x_0_uncond + cfg_scale * (x_0_cond - x_0_uncond)
            // (host-side combine — one-shot per step, ~0.05ms)
            auto h_uncond = x_0_uncond.to_host();
            auto h_cond = x_0_cond.to_host();
            std::vector<float> h_guided(B * 784);
            for (size_t i = 0; i < h_guided.size(); ++i) {
                h_guided[i] = h_uncond[i] + cfg_scale * (h_cond[i] - h_uncond[i]);
            }
            x_0_guided.from_host(h_guided);

            ddim_step(x_prev, x_t, x_0_guided, t, t_prev, sched);
        } else {
            // Single forward pass (no CFG)
            model.forward(x_t, d_timesteps, d_labels, B);
            ddim_step(x_prev, x_t, model.output, t, t_prev, sched);
        }

        x_t.copy_from(x_prev);

        if (idx == 0 || idx == n_ddim_steps / 4 || idx == n_ddim_steps / 2 ||
            idx == 3 * n_ddim_steps / 4 || idx == (int)ts.size() - 1)
        {
            std::printf("[step %d, t=%d] middle row of first 4 samples:\n", idx, t);
            auto h = x_t.to_host();
            const char* shades = " .:-=+*#%@";
            for (int b = 0; b < std::min(B, 4); ++b) {
                std::printf("  sample %d: ", b);
                for (int col = 0; col < 28; ++col) {
                    float v = (h[b * 784 + 14 * 28 + col] + 1.f) * 0.5f;
                    int idx2 = (int)std::max(0.f, std::min(9.f, v * 9.99f));
                    std::putchar(shades[idx2]);
                }
                std::putchar('\n');
            }
            std::puts("");
        }
    }

    auto h_final = x_t.to_host();
    std::printf("Final samples (full 28x28 grids):\n");
    for (int b = 0; b < std::min(B, 4); ++b) {
        std::printf("\nsample %d:\n", b);
        const char* shades = " .:-=+*#%@";
        for (int row = 0; row < 28; ++row) {
            std::printf("  ");
            for (int col = 0; col < 28; ++col) {
                float v = (h_final[b * 784 + row * 28 + col] + 1.f) * 0.5f;
                int idx2 = (int)std::max(0.f, std::min(9.f, v * 9.99f));
                std::putchar(shades[idx2]);
                std::putchar(shades[idx2]);
            }
            std::putchar('\n');
        }
    }

    float lo = h_final[0], hi = h_final[0];
    int nan_count = 0;
    for (float v : h_final) {
        if (std::isnan(v)) nan_count++;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    std::printf("\nFinal pixel range: [%.3f, %.3f], NaN count: %d\n",
                lo, hi, nan_count);

    cudaFree(d_timesteps);
    cudaFree(d_labels);
    cudaFree(d_labels_uncond);
    return 0;
}