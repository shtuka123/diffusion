#pragma once
#include "tensor.h"
#include "parameter.h"
#include "noise_schedule.h"
#include <vector>
#include <cstdint>
#include <string>

struct MLPDenoiserConfig {
    int input_dim   = 784;
    int time_dim    = 128;
    int hidden_dim  = 256;
    int n_layers    = 4;
    int num_classes = 10;     // for class conditioning; 10 for MNIST
};

struct MLPDenoiser {
    MLPDenoiserConfig cfg;

    // Class embedding: (num_classes + 1, time_dim).
    // Last row is the "null" class for CFG (initialized to zero).
    Parameter class_emb;

    // Timestep MLP
    Parameter time_W1, time_b1;
    Parameter time_W2, time_b2;

    // Main MLP
    std::vector<Parameter> Ws;
    std::vector<Parameter> bs;

    // Forward-pass scratch
    Tensor t_emb_raw;        // (B, time_dim) — sinusoidal output (also used as scratch
                             // for the post-add value)
    Tensor c_emb;            // (B, time_dim) — looked-up class embedding
    Tensor t_emb_h;          // (B, time_dim) — pre-SiLU
    Tensor t_emb_h_silu;     // (B, time_dim) — post-SiLU
    Tensor t_emb;            // (B, time_dim) — final timestep+class embedding

    Tensor concat_input;     // (B, input_dim + time_dim)
    std::vector<Tensor> hidden_pre_silu;
    std::vector<Tensor> hidden_post_silu;
    Tensor output;           // (B, input_dim)

    void build(const MLPDenoiserConfig& config, int B, uint64_t seed = 42);

    // Forward: x_t (B, input_dim), timesteps (B,), labels (B,) → output (B, input_dim).
    // labels_device may be NULL — in that case all-null-class is used (unconditional).
    void forward(const Tensor& x_t,
                 const int* timesteps_device,
                 const int* labels_device,
                 int B);

    // Backward: dy (B, input_dim) is gradient at output. Fills all parameter grads.
    void backward(const Tensor& dy, const Tensor& x_t,
                  const int* timesteps_device,
                  const int* labels_device,
                  int B);

    std::vector<Parameter*> params();
    size_t param_count() const;

    void save(const std::string& path);
    void load(const std::string& path);
};