#pragma once
#include "tensor.h"
#include "parameter.h"
#include "transformer_block.h"
#include "init.h"
#include <vector>
#include <cstdint>
#include <random>

// Sample one token from logits[(B-1)*T + (T-1)] (the last position of the
// last batch element). Uses temperature and optional top-k.
//
// logits is the GPTTiny.logits tensor with shape (B, T, V).
// We sample for batch index 0 by default.
int sample_token(
    const Tensor& logits, int B, int T, int V,
    int batch_index, int seq_index,
    float temperature, int top_k,
    std::mt19937& rng);

struct GPTTinyConfig {
    int vocab_size  = 65;
    int max_seq_len = 64;
    int d_model     = 128;
    int n_layers    = 4;
    int n_heads     = 4;
    int d_mlp       = 512;   // typically 4 * d_model
};

// Parameters for one Transformer block.
struct BlockParams {
    Parameter gamma1, beta1;       // LN1
    Parameter W_Q, b_Q;
    Parameter W_K, b_K;
    Parameter W_V, b_V;
    Parameter W_O, b_O;            // MHA
    Parameter gamma2, beta2;       // LN2
    Parameter W1_mlp, b1_mlp;
    Parameter W2_mlp, b2_mlp;      // MLP
};

// All scratch / cache tensors for one block.
struct BlockCacheBundle {
    TransformerBlockCache cache;
};

struct GPTTiny {
    GPTTinyConfig cfg;

    // Embeddings
    Parameter E;     // (V, D)
    Parameter P;     // (T_max, D)

    // Per-layer parameters
    std::vector<BlockParams> blocks;

    // Final LN
    Parameter gamma_f, beta_f;

    // Output head: (D, V)
    Parameter W_head, b_head;

    // Cache: per-layer block caches + final LN cache + head intermediates
    std::vector<BlockCacheBundle> caches;
    Tensor x_emb;             // (B, T, D) — after token + pos embed
    Tensor block_outputs;     // scratch (B, T, D) — output of each block
                              // We'll need a per-block cache to remember inputs.
    std::vector<Tensor> block_inputs;  // (n_layers,) each (B, T, D), saved for backward
    Tensor x_postnorm;        // (B, T, D) — after final LN
    Tensor rstd_f;            // (B*T,)
    Tensor logits;            // (B, T, V)

    GPTTiny() = default;

    // Build the model: allocate all parameters, initialize them, allocate
    // caches sized for batch B and sequence T (must be <= cfg.max_seq_len).
    void build(const GPTTinyConfig& config, int B, int T, uint64_t seed = 42);

    // Forward: tokens (B, T) device int* -> logits (B, T, V) inside this->logits.
    // The output is also returned by reference for chaining.
    void forward(const int* tokens_device, int B, int T);

    // Backward: dlogits (B, T, V) -> all parameter gradients populated, plus dx_emb.
    // The "input" gradient flows back to dE and dP via embedding_backward.
    void backward(const Tensor& dlogits, const int* tokens_device, int B, int T);

    // Collect raw pointers to every Parameter for the optimizer.
    std::vector<Parameter*> params();

    // Total parameter count.
    size_t param_count() const;

    // Save all parameter weights to disk in a single binary file.
// Format: header (n_params, then per-tensor shape metadata), then concatenated data.
void save(const std::string& path);

// Load weights from a file. Model must already be built() with matching config.
void load(const std::string& path);
};