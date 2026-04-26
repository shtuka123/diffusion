#include "mlp_denoiser.h"
#include "linear.h"
#include "elementwise.h"
#include "matmul.h"
#include "init.h"
#include "timestep_embed.h"
#include "embedding.h"
#include "cuda_utils.h"

#include <fstream>
#include <cstdio>
#include <cstdint>

void MLPDenoiser::build(const MLPDenoiserConfig& config, int B, uint64_t seed) {
    cfg = config;

    int td = cfg.time_dim;

    // Class embedding (num_classes + 1 rows; last row is null class, zero-init)
    class_emb = Parameter({cfg.num_classes + 1, td});
    init_normal(class_emb, 0.02f, seed + 50);
    {
        auto h = class_emb.w.to_host();
        for (int i = 0; i < td; ++i) {
            h[cfg.num_classes * td + i] = 0.0f;
        }
        class_emb.w.from_host(h);
    }

    // Timestep MLP
    time_W1 = Parameter({td, td});  init_he_normal(time_W1, td, seed + 1);
    time_b1 = Parameter({td});       init_zero(time_b1);
    time_W2 = Parameter({td, td});  init_he_normal(time_W2, td, seed + 2);
    time_b2 = Parameter({td});       init_zero(time_b2);

    int input_dim = cfg.input_dim;
    int hd = cfg.hidden_dim;
    int N = cfg.n_layers;

    Ws.clear();
    bs.clear();
    Ws.reserve(N);
    bs.reserve(N);

    for (int l = 0; l < N; ++l) {
        int in_dim, out_dim;
        if (l == 0) {
            in_dim = input_dim + td;
            out_dim = hd;
        } else if (l == N - 1) {
            in_dim = hd;
            out_dim = input_dim;
        } else {
            in_dim = hd;
            out_dim = hd;
        }
        Parameter W({in_dim, out_dim}); init_he_normal(W, in_dim, seed + 100 * l + 3);
        Parameter b({out_dim});         init_zero(b);
        Ws.push_back(std::move(W));
        bs.push_back(std::move(b));
    }

    // Scratch
    t_emb_raw    = Tensor({B, td});
    c_emb        = Tensor({B, td});
    t_emb_h      = Tensor({B, td});
    t_emb_h_silu = Tensor({B, td});
    t_emb        = Tensor({B, td});

    concat_input = Tensor({B, input_dim + td});

    hidden_pre_silu.clear();
    hidden_post_silu.clear();
    hidden_pre_silu.reserve(N);
    hidden_post_silu.reserve(N);
    for (int l = 0; l < N; ++l) {
        int out_dim = (l == N - 1) ? input_dim : hd;
        hidden_pre_silu.push_back(Tensor({B, out_dim}));
        hidden_post_silu.push_back(Tensor({B, out_dim}));
    }
    output = Tensor({B, input_dim});
}

void MLPDenoiser::forward(const Tensor& x_t,
                          const int* timesteps_device,
                          const int* labels_device,
                          int B)
{
    int td = cfg.time_dim;

    // 1. Sinusoidal timestep embedding
    sinusoidal_embed(t_emb_raw, timesteps_device, B);

    // 1b. Class embedding lookup, then add to timestep embedding
    if (labels_device != nullptr) {
        // embedding_forward expects (B, T, D) output. Our c_emb is (B, D); reshape
        // to (B, 1, D) for the call, then restore.
        c_emb.shape = {B, 1, td};
        embedding_forward(c_emb, labels_device, class_emb.w, B, 1);
        c_emb.shape = {B, td};
    } else {
        CUDA_CHECK(cudaMemset(c_emb.data, 0, c_emb.numel * sizeof(float)));
    }

    // t_emb_raw += c_emb (in place)
    add(t_emb_raw, t_emb_raw, c_emb);

    // 2. Time MLP: Linear → SiLU → Linear
    linear_forward(t_emb_h, t_emb_raw, time_W1.w, time_b1.w);
    silu_forward(t_emb_h_silu, t_emb_h);
    linear_forward(t_emb, t_emb_h_silu, time_W2.w, time_b2.w);

    // 3. Concatenate x_t and t_emb
    concat_2d(concat_input, x_t, t_emb);

    // 4. Main MLP
    int N = cfg.n_layers;
    const Tensor* current_input = &concat_input;
    for (int l = 0; l < N; ++l) {
        linear_forward(hidden_pre_silu[l], *current_input, Ws[l].w, bs[l].w);
        if (l < N - 1) {
            silu_forward(hidden_post_silu[l], hidden_pre_silu[l]);
            current_input = &hidden_post_silu[l];
        } else {
            output.copy_from(hidden_pre_silu[l]);
        }
    }
}

void MLPDenoiser::backward(const Tensor& dy, const Tensor& x_t,
                            const int* timesteps_device,
                            const int* labels_device,
                            int B)
{
    int N = cfg.n_layers;
    int td = cfg.time_dim;

    // Backward through main MLP, in reverse
    Tensor d_current = Tensor({dy.size(0), dy.size(1)});
    d_current.copy_from(dy);

    for (int l = N - 1; l >= 0; --l) {
        Tensor d_pre_silu({d_current.size(0), d_current.size(1)});
        if (l < N - 1) {
            silu_backward(d_pre_silu, d_current, hidden_pre_silu[l]);
        } else {
            d_pre_silu.copy_from(d_current);
        }

        const Tensor* x_in = (l == 0) ? &concat_input : &hidden_post_silu[l - 1];
        int in_dim = Ws[l].w.size(0);
        Tensor d_x_in({d_current.size(0), in_dim});
        linear_backward(d_x_in, Ws[l].grad, bs[l].grad,
                        *x_in, Ws[l].w, d_pre_silu);

        d_current = std::move(d_x_in);
    }

    // Split d_current → d_xt (unused) + d_t_emb
    Tensor d_xt({B, cfg.input_dim});
    Tensor d_t_emb({B, td});
    split_2d(d_xt, d_t_emb, d_current);

    // Time MLP backward: Linear → SiLU → Linear (in reverse)
    Tensor d_t_emb_h_silu({B, td});
    linear_backward(d_t_emb_h_silu, time_W2.grad, time_b2.grad,
                    t_emb_h_silu, time_W2.w, d_t_emb);

    Tensor d_t_emb_h({B, td});
    silu_backward(d_t_emb_h, d_t_emb_h_silu, t_emb_h);

    Tensor d_t_emb_raw({B, td});
    linear_backward(d_t_emb_raw, time_W1.grad, time_b1.grad,
                    t_emb_raw, time_W1.w, d_t_emb_h);

    // t_emb_raw was the SUM of (sinusoidal output, no learned params) + c_emb.
    // Sinusoidal has no params; c_emb's gradient backprops into class_emb.grad.
    if (labels_device != nullptr) {
        d_t_emb_raw.shape = {B, 1, td};
        embedding_backward(class_emb.grad, d_t_emb_raw, labels_device, B, 1);
        d_t_emb_raw.shape = {B, td};
    }
}

std::vector<Parameter*> MLPDenoiser::params() {
    std::vector<Parameter*> out;
    out.push_back(&class_emb);
    out.push_back(&time_W1); out.push_back(&time_b1);
    out.push_back(&time_W2); out.push_back(&time_b2);
    for (size_t l = 0; l < Ws.size(); ++l) {
        out.push_back(&Ws[l]);
        out.push_back(&bs[l]);
    }
    return out;
}

size_t MLPDenoiser::param_count() const {
    size_t total = class_emb.w.numel;
    total += time_W1.w.numel + time_b1.w.numel + time_W2.w.numel + time_b2.w.numel;
    for (size_t l = 0; l < Ws.size(); ++l) {
        total += Ws[l].w.numel + bs[l].w.numel;
    }
    return total;
}

static const uint32_t MLP_DEN_MAGIC = 0x4D4C504E;

void MLPDenoiser::save(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "MLPDenoiser::save: cannot open %s\n", path.c_str());
        std::abort();
    }
    f.write(reinterpret_cast<const char*>(&MLP_DEN_MAGIC), 4);
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
    std::printf("Saved MLP denoiser to %s\n", path.c_str());
}

void MLPDenoiser::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "MLPDenoiser::load: cannot open %s\n", path.c_str());
        std::abort();
    }
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != MLP_DEN_MAGIC) {
        std::fprintf(stderr, "MLPDenoiser::load: bad magic\n");
        std::abort();
    }
    auto ps = params();
    for (auto* p : ps) {
        uint32_t ndim;
        f.read(reinterpret_cast<char*>(&ndim), 4);
        for (size_t d = 0; d < ndim; ++d) {
            uint32_t dim;
            f.read(reinterpret_cast<char*>(&dim), 4);
        }
        std::vector<float> host(p->w.numel);
        f.read(reinterpret_cast<char*>(host.data()), host.size() * sizeof(float));
        p->w.from_host(host);
    }
    std::printf("Loaded MLP denoiser from %s\n", path.c_str());
}