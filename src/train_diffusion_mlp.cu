// Train the MLP denoiser on MNIST with diffusion-style training.
//
// Class-conditional with classifier-free guidance support: 10% of the time,
// we replace the label with the null class (num_classes). This trains the
// model to do both conditional and unconditional generation. At sample time,
// CFG combines them to amplify class adherence.

#include "tensor.h"
#include "parameter.h"
#include "mlp_denoiser.h"
#include "noise_schedule.h"
#include "diffusion.h"
#include "mnist.h"
#include "optim.h"
#include "scheduler.h"
#include "cuda_utils.h"

#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

int main() {
    MLPDenoiserConfig model_cfg;
    model_cfg.input_dim  = 784;
    model_cfg.time_dim   = 128;
    model_cfg.hidden_dim = 256;
    model_cfg.n_layers   = 4;
    model_cfg.num_classes = 10;

    int B = 128;
    int T_diffusion = 1000;
    int n_steps = 5000;
    int log_every = 100;
    int warmup = 200;
    float lr_max = 1e-3f;
    float lr_min = 1e-4f;
    float label_dropout_prob = 0.1f;     // for CFG
    uint64_t seed = 42;

    std::printf("Loading MNIST...\n");
    MNIST train("data/mnist/train-images-idx3-ubyte",
                "data/mnist/train-labels-idx1-ubyte");
    std::printf("  %d training samples\n\n", train.n_samples);

    NoiseSchedule sched;
    sched.build(T_diffusion);

    MLPDenoiser model;
    model.build(model_cfg, B, seed);
    std::printf("Model built: %zu parameters (~%.2f MB)\n\n",
                model.param_count(), model.param_count() * 4.0 / (1024 * 1024));

    auto params = model.params();

    Tensor x0({B, 784});
    Tensor eps({B, 784});
    Tensor x_t({B, 784});
    Tensor loss({1});
    Tensor dy({B, 784});

    int* d_timesteps = nullptr;
    int* d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_timesteps, B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels, B * sizeof(int)));

    std::mt19937 rng(seed + 99);
    std::uniform_int_distribution<int> t_dist(0, T_diffusion - 1);
    std::uniform_real_distribution<float> dropout_dist(0.0f, 1.0f);

    AdamState adam;
    adam.beta1 = 0.9f;
    adam.beta2 = 0.999f;
    adam.eps = 1e-8f;

    std::vector<float> h_x_norm(B * 784);
    std::vector<int>   h_timesteps(B);
    std::vector<int>   h_labels_dropout(B);

    std::printf("Training: B=%d, n_steps=%d, lr=%.4f→%.4f, label_dropout=%.2f\n",
                B, n_steps, lr_max, lr_min, label_dropout_prob);
    std::printf("---------------------------------------------\n");

    float running_loss = 0.f;
    int running_count = 0;

    for (int step = 0; step < n_steps; ++step) {
        train.get_random_batch(B, x0, d_labels, rng);

        // Apply label dropout for CFG: replace ~10% of labels with null class.
        CUDA_CHECK(cudaMemcpy(h_labels_dropout.data(), d_labels,
                              B * sizeof(int), cudaMemcpyDeviceToHost));
        for (int b = 0; b < B; ++b) {
            if (dropout_dist(rng) < label_dropout_prob) {
                h_labels_dropout[b] = model_cfg.num_classes;  // null class
            }
        }
        CUDA_CHECK(cudaMemcpy(d_labels, h_labels_dropout.data(),
                              B * sizeof(int), cudaMemcpyHostToDevice));

        auto h_x_raw = x0.to_host();
        for (size_t i = 0; i < h_x_raw.size(); ++i) {
            h_x_norm[i] = 2.0f * h_x_raw[i] - 1.0f;
        }
        x0.from_host(h_x_norm);

        for (int b = 0; b < B; ++b) h_timesteps[b] = t_dist(rng);
        CUDA_CHECK(cudaMemcpy(d_timesteps, h_timesteps.data(),
                              B * sizeof(int), cudaMemcpyHostToDevice));

        eps = Tensor::randn({B, 784}, seed + 1000 + step);

        q_sample(x_t, x0, eps, d_timesteps, sched);

        model.forward(x_t, d_timesteps, d_labels, B);

        mse_loss_forward(loss, model.output, x0);
        running_loss += loss.to_host()[0];
        running_count += 1;

        mse_loss_backward(dy, model.output, x0);
        model.backward(dy, x_t, d_timesteps, d_labels, B);

        adam.lr = warmup_cosine_lr(step, warmup, n_steps, lr_max, lr_min);
        adam_step_all(params, adam);
        zero_grads(params);

        if ((step + 1) % log_every == 0) {
            auto h_pred = model.output.to_host();
            auto h_x0 = x0.to_host();
            float lo_t_mse = 0, mid_t_mse = 0, hi_t_mse = 0;
            int lo_n = 0, mid_n = 0, hi_n = 0;
            for (int b = 0; b < B; ++b) {
                float sq = 0;
                for (int i = 0; i < 784; ++i) {
                    float d = h_pred[b*784+i] - h_x0[b*784+i];
                    sq += d*d;
                }
                sq /= 784.0f;
                int t = h_timesteps[b];
                if      (t < 100)  { lo_t_mse  += sq; lo_n++;  }
                else if (t < 700)  { mid_t_mse += sq; mid_n++; }
                else               { hi_t_mse  += sq; hi_n++;  }
            }
            float avg = running_loss / running_count;
            std::printf("  step %5d / %d  lr=%.5f  loss %.4f  | t<100: %.3f  t<700: %.3f  t>=700: %.3f\n",
                        step + 1, n_steps, adam.lr, avg,
                        lo_n  ? lo_t_mse  / lo_n  : 0,
                        mid_n ? mid_t_mse / mid_n : 0,
                        hi_n  ? hi_t_mse  / hi_n  : 0);
            running_loss = 0.f;
            running_count = 0;
        }
    }

    std::printf("\nTraining complete. Saving checkpoint...\n");
    model.save("checkpoints/diffusion_mlp.bin");

    cudaFree(d_timesteps);
    cudaFree(d_labels);
    return 0;
}