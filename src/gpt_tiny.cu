#include "gpt_tiny.h"
#include "transformer_block.h"
#include "embedding.h"
#include "layernorm.h"
#include "linear.h"
#include "matmul.h"
#include "elementwise.h"
#include "init.h"
#include "cuda_utils.h"
#include <fstream>
#include <string>

#include <cstdio>

#include <fstream>
#include <string>

#include <algorithm>
#include <cmath>

int sample_token(
    const Tensor& logits, int B, int T, int V,
    int batch_index, int seq_index,
    float temperature, int top_k,
    std::mt19937& rng)
{
    // Copy the (V,) slice of logits we need to host.
    std::vector<float> v(V);
    size_t offset = (size_t)batch_index * T * V + (size_t)seq_index * V;
    CUDA_CHECK(cudaMemcpy(v.data(), logits.data + offset,
                          V * sizeof(float), cudaMemcpyDeviceToHost));

    // Apply temperature scaling: divide logits by T (T = temperature here).
    if (temperature > 0.f) {
        float invT = 1.0f / temperature;
        for (int i = 0; i < V; ++i) v[i] *= invT;
    }

    // Optional top-k filter: keep only the k highest, set the rest to -inf.
    if (top_k > 0 && top_k < V) {
        std::vector<float> sorted = v;
        std::nth_element(sorted.begin(), sorted.begin() + (V - top_k),
                         sorted.end());
        float threshold = sorted[V - top_k];
        for (int i = 0; i < V; ++i) {
            if (v[i] < threshold) v[i] = -1e30f;
        }
    }

    // Numerically stable softmax.
    float m = v[0];
    for (int i = 1; i < V; ++i) if (v[i] > m) m = v[i];
    float sum = 0.f;
    for (int i = 0; i < V; ++i) {
        v[i] = std::exp(v[i] - m);
        sum += v[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < V; ++i) v[i] *= inv;

    // Sample: uniform [0, 1), walk the cumulative distribution.
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u = dist(rng);
    float cum = 0.f;
    for (int i = 0; i < V; ++i) {
        cum += v[i];
        if (u < cum) return i;
    }
    return V - 1;  // floating-point safety fallback
}

// File format (simple):
//   [4 bytes] magic 0x47505430  ('GPT0')
//   For each parameter, in the order returned by params():
//     [4 bytes] ndim (uint32)
//     [4 bytes × ndim] shape (uint32)
//     [4 bytes × numel] data (float32)

static const uint32_t GPT_SAVE_MAGIC = 0x47505430;  // 'GPT0'

void GPTTiny::save(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "GPTTiny::save: cannot open %s\n", path.c_str());
        std::abort();
    }
    f.write(reinterpret_cast<const char*>(&GPT_SAVE_MAGIC), 4);

    auto ps = params();
    for (auto* p : ps) {
        uint32_t ndim = (uint32_t)p->w.shape.size();
        f.write(reinterpret_cast<const char*>(&ndim), 4);
        for (int d : p->w.shape) {
            uint32_t dim = (uint32_t)d;
            f.write(reinterpret_cast<const char*>(&dim), 4);
        }
        auto host = p->w.to_host();
        f.write(reinterpret_cast<const char*>(host.data()),
                host.size() * sizeof(float));
    }
    std::printf("Saved model to %s (%zu params)\n", path.c_str(), ps.size());
}

void GPTTiny::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "GPTTiny::load: cannot open %s\n", path.c_str());
        std::abort();
    }
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != GPT_SAVE_MAGIC) {
        std::fprintf(stderr, "GPTTiny::load: bad magic 0x%x\n", magic);
        std::abort();
    }

    auto ps = params();
    for (auto* p : ps) {
        uint32_t ndim;
        f.read(reinterpret_cast<char*>(&ndim), 4);
        if (ndim != p->w.shape.size()) {
            std::fprintf(stderr, "GPTTiny::load: ndim mismatch (%u vs %zu)\n",
                         ndim, p->w.shape.size());
            std::abort();
        }
        for (size_t d = 0; d < ndim; ++d) {
            uint32_t dim;
            f.read(reinterpret_cast<char*>(&dim), 4);
            if ((int)dim != p->w.shape[d]) {
                std::fprintf(stderr, "GPTTiny::load: shape mismatch at dim %zu\n", d);
                std::abort();
            }
        }
        std::vector<float> host(p->w.numel);
        f.read(reinterpret_cast<char*>(host.data()), host.size() * sizeof(float));
        p->w.from_host(host);
    }
    std::printf("Loaded model from %s\n", path.c_str());
}

void GPTTiny::build(const GPTTinyConfig& config, int B, int T, uint64_t seed) {
    cfg = config;
    int D = cfg.d_model;
    int V = cfg.vocab_size;
    int H = cfg.n_heads;
    int d_k = D / H;
    int D_mlp = cfg.d_mlp;
    int N = cfg.n_layers;

    if (T > cfg.max_seq_len) {
        std::fprintf(stderr, "GPTTiny::build: T (%d) exceeds max_seq_len (%d)\n",
                     T, cfg.max_seq_len);
        std::abort();
    }

    // ----- Embeddings -----
    E = Parameter({V, D});
    P = Parameter({cfg.max_seq_len, D});
    init_normal(E, 0.02f, seed + 1);
    init_normal(P, 0.02f, seed + 2);

    // ----- Per-layer params + caches -----
    blocks.clear();
    blocks.reserve(N);
    caches.clear();
    caches.reserve(N);
    block_inputs.clear();
    block_inputs.reserve(N);

    for (int l = 0; l < N; ++l) {
        BlockParams bp;
        // LN1
        bp.gamma1 = Parameter({D});  init_normal(bp.gamma1, 1.0f, seed + 100*l + 1);  // ~1
        bp.beta1  = Parameter({D});  init_zero(bp.beta1);
        // Workaround: we want gamma1 to be all 1s, not random. Use init_zero then add 1.
        {
            std::vector<float> ones(D, 1.0f);
            bp.gamma1.w.from_host(ones);
        }

        // MHA (use He init for the projections)
        bp.W_Q = Parameter({D, D});  init_he_normal(bp.W_Q, D, seed + 100*l + 3);
        bp.b_Q = Parameter({D});     init_zero(bp.b_Q);
        bp.W_K = Parameter({D, D});  init_he_normal(bp.W_K, D, seed + 100*l + 4);
        bp.b_K = Parameter({D});     init_zero(bp.b_K);
        bp.W_V = Parameter({D, D});  init_he_normal(bp.W_V, D, seed + 100*l + 5);
        bp.b_V = Parameter({D});     init_zero(bp.b_V);
        bp.W_O = Parameter({D, D});  init_he_normal(bp.W_O, D, seed + 100*l + 6);
        bp.b_O = Parameter({D});     init_zero(bp.b_O);

        // LN2 — same gamma=1 trick
        bp.gamma2 = Parameter({D});
        bp.beta2  = Parameter({D});  init_zero(bp.beta2);
        {
            std::vector<float> ones(D, 1.0f);
            bp.gamma2.w.from_host(ones);
        }

        // MLP
        bp.W1_mlp = Parameter({D, D_mlp});      init_he_normal(bp.W1_mlp, D, seed + 100*l + 7);
        bp.b1_mlp = Parameter({D_mlp});         init_zero(bp.b1_mlp);
        bp.W2_mlp = Parameter({D_mlp, D});      init_he_normal(bp.W2_mlp, D_mlp, seed + 100*l + 8);
        bp.b2_mlp = Parameter({D});             init_zero(bp.b2_mlp);

        blocks.push_back(std::move(bp));

        // Cache for this block
        BlockCacheBundle bcb;
        bcb.cache.t1 = Tensor({B, T, D});
        bcb.cache.rstd1 = Tensor({B * T});
        bcb.cache.a = Tensor({B, T, D});
        bcb.cache.r1 = Tensor({B, T, D});
        bcb.cache.t2 = Tensor({B, T, D});
        bcb.cache.rstd2 = Tensor({B * T});
        bcb.cache.m = Tensor({B, T, D});
        bcb.cache.mha_cache.Q_raw = Tensor({B, T, D});
        bcb.cache.mha_cache.K_raw = Tensor({B, T, D});
        bcb.cache.mha_cache.V_raw = Tensor({B, T, D});
        bcb.cache.mha_cache.Q_heads = Tensor({B, H, T, d_k});
        bcb.cache.mha_cache.K_heads = Tensor({B, H, T, d_k});
        bcb.cache.mha_cache.V_heads = Tensor({B, H, T, d_k});
        bcb.cache.mha_cache.attn_out_heads = Tensor({B, H, T, d_k});
        bcb.cache.mha_cache.attn_out_flat  = Tensor({B, T, D});
        bcb.cache.mha_cache.P     = Tensor({B * H, T, T});
        bcb.cache.mha_cache.S_buf = Tensor({B * H, T, T});
        bcb.cache.mlp_cache.h_pre  = Tensor({B, T, D_mlp});
        bcb.cache.mlp_cache.h_post = Tensor({B, T, D_mlp});
        caches.push_back(std::move(bcb));

        // Block input cache (saved for backward)
        block_inputs.push_back(Tensor({B, T, D}));
    }

    // Final LN
    gamma_f = Parameter({D});
    beta_f  = Parameter({D});
    init_zero(beta_f);
    {
        std::vector<float> ones(D, 1.0f);
        gamma_f.w.from_host(ones);
    }

    // Output head
    W_head = Parameter({D, V});  init_he_normal(W_head, D, seed + 999);
    b_head = Parameter({V});     init_zero(b_head);

    // Forward-pass scratch tensors
    x_emb        = Tensor({B, T, D});
    block_outputs = Tensor({B, T, D});
    x_postnorm   = Tensor({B, T, D});
    rstd_f       = Tensor({B * T});
    logits       = Tensor({B, T, V});
}

void GPTTiny::forward(const int* tokens_device, int B, int T) {
    int D = cfg.d_model;
    int V = cfg.vocab_size;

    // ----- Embeddings -----
    embedding_forward(x_emb, tokens_device, E.w, B, T);
    positional_add_forward(x_emb, P.w, B, T);

    // ----- Blocks -----
    // Save the input to each block, since the block's backward needs it.
    Tensor* current = &x_emb;
    for (int l = 0; l < cfg.n_layers; ++l) {
        // Save block l's input (= output of previous step).
        block_inputs[l].copy_from(*current);

        const BlockParams& bp = blocks[l];
        TransformerBlockCache& cache = caches[l].cache;

        Tensor& output = (l == cfg.n_layers - 1) ? block_outputs : block_outputs;

        transformer_block_forward(
            output, cache, *current,
            bp.gamma1.w, bp.beta1.w,
            bp.W_Q.w, bp.b_Q.w, bp.W_K.w, bp.b_K.w,
            bp.W_V.w, bp.b_V.w, bp.W_O.w, bp.b_O.w,
            bp.gamma2.w, bp.beta2.w,
            bp.W1_mlp.w, bp.b1_mlp.w, bp.W2_mlp.w, bp.b2_mlp.w,
            cfg.n_heads);

        current = &block_outputs;
    }

    // ----- Final LN -----
    layernorm_forward(x_postnorm, rstd_f, *current, gamma_f.w, beta_f.w);

    // ----- Output projection: logits = x_postnorm @ W_head + b_head -----
    // Flatten (B, T, D) → (B*T, D) → matmul to (B*T, V)
    {
        x_postnorm.shape = {B * T, D};
        logits.shape = {B * T, V};
        matmul_raw(logits.data, x_postnorm.data, W_head.w.data, B * T, D, V);

        // bias_add: logits += b_head
        extern __global__ void bias_add_kernel(const float*, const float*, float*,
                                               int, int);
        int n = B * T * V;
        int blk = 256, grd = (n + blk - 1) / blk;
        bias_add_kernel<<<grd, blk>>>(logits.data, b_head.w.data, logits.data, n, V);
        CUDA_CHECK_KERNEL();

        x_postnorm.shape = {B, T, D};
        logits.shape = {B, T, V};
    }
}

void GPTTiny::backward(const Tensor& dlogits, const int* tokens_device, int B, int T) {
    int D = cfg.d_model;
    int V = cfg.vocab_size;

    // ----- Backward through output projection -----
    // logits = x_postnorm @ W_head + b_head
    // → dW_head = x_postnorm^T @ dlogits   shape (D, V)
    //   db_head = sum(dlogits, axis=batch)
    //   dx_postnorm = dlogits @ W_head^T   shape (B*T, D)
    Tensor dx_postnorm({B * T, D});
    {
        x_postnorm.shape = {B * T, D};
        Tensor& dl = const_cast<Tensor&>(dlogits);
        dl.shape = {B * T, V};

        linear_backward(dx_postnorm, W_head.grad, b_head.grad,
                        x_postnorm, W_head.w, dl);

        x_postnorm.shape = {B, T, D};
        dl.shape = {B, T, V};
    }
    dx_postnorm.shape = {B, T, D};

    // ----- Backward through final LN -----
    // x_postnorm = LN(blocks_output, gamma_f, beta_f)
    // → d_blocks_out = LN_backward(dx_postnorm, ...)
    Tensor d_blocks_out({B, T, D});
    layernorm_backward(d_blocks_out, gamma_f.grad, beta_f.grad,
                       dx_postnorm, block_outputs, gamma_f.w, rstd_f);

    // ----- Backward through blocks (reverse order) -----
    // We need each block's (1) cached intermediates and (2) the original input
    // to that block. We saved (1) in caches[l].cache and (2) in block_inputs[l].
    Tensor d_current({B, T, D});
    d_current.copy_from(d_blocks_out);

    for (int l = cfg.n_layers - 1; l >= 0; --l) {
        const BlockParams& bp = blocks[l];
        TransformerBlockCache& cache = caches[l].cache;

        Tensor d_prev({B, T, D});

        transformer_block_backward(
            d_prev,
            blocks[l].gamma1.grad, blocks[l].beta1.grad,
            blocks[l].W_Q.grad, blocks[l].b_Q.grad,
            blocks[l].W_K.grad, blocks[l].b_K.grad,
            blocks[l].W_V.grad, blocks[l].b_V.grad,
            blocks[l].W_O.grad, blocks[l].b_O.grad,
            blocks[l].gamma2.grad, blocks[l].beta2.grad,
            blocks[l].W1_mlp.grad, blocks[l].b1_mlp.grad,
            blocks[l].W2_mlp.grad, blocks[l].b2_mlp.grad,
            d_current, block_inputs[l], cache,
            bp.gamma1.w, bp.W_Q.w, bp.W_K.w, bp.W_V.w, bp.W_O.w,
            bp.gamma2.w, bp.W1_mlp.w, bp.W2_mlp.w,
            cfg.n_heads);

        d_current.copy_from(d_prev);
    }

    // d_current is now dx_emb — the gradient at the embedding output (B, T, D).

    // ----- Backward through embeddings -----
    // x_emb[b, t, :] = E[tokens[b, t], :] + P[t, :]
    // → dE: scatter-add at indices tokens[b,t]
    //   dP: scatter-add over batch dim
    embedding_backward(E.grad, d_current, tokens_device, B, T);
    positional_add_backward(P.grad, d_current, B, T);
}

std::vector<Parameter*> GPTTiny::params() {
    std::vector<Parameter*> out;
    out.push_back(&E);
    out.push_back(&P);
    for (auto& b : blocks) {
        out.push_back(&b.gamma1);  out.push_back(&b.beta1);
        out.push_back(&b.W_Q);     out.push_back(&b.b_Q);
        out.push_back(&b.W_K);     out.push_back(&b.b_K);
        out.push_back(&b.W_V);     out.push_back(&b.b_V);
        out.push_back(&b.W_O);     out.push_back(&b.b_O);
        out.push_back(&b.gamma2);  out.push_back(&b.beta2);
        out.push_back(&b.W1_mlp);  out.push_back(&b.b1_mlp);
        out.push_back(&b.W2_mlp);  out.push_back(&b.b2_mlp);
    }
    out.push_back(&gamma_f);  out.push_back(&beta_f);
    out.push_back(&W_head);   out.push_back(&b_head);
    return out;
}

size_t GPTTiny::param_count() const {
    size_t total = E.w.numel + P.w.numel;
    for (const auto& b : blocks) {
        total += b.gamma1.w.numel + b.beta1.w.numel;
        total += b.W_Q.w.numel + b.b_Q.w.numel;
        total += b.W_K.w.numel + b.b_K.w.numel;
        total += b.W_V.w.numel + b.b_V.w.numel;
        total += b.W_O.w.numel + b.b_O.w.numel;
        total += b.gamma2.w.numel + b.beta2.w.numel;
        total += b.W1_mlp.w.numel + b.b1_mlp.w.numel;
        total += b.W2_mlp.w.numel + b.b2_mlp.w.numel;
    }
    total += gamma_f.w.numel + beta_f.w.numel;
    total += W_head.w.numel + b_head.w.numel;
    return total;
}