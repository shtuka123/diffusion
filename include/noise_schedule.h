#pragma once
#include "tensor.h"
#include "cuda_utils.h"
#include <cmath>
#include <cstdio>
#include <vector>

// DDPM-style linear noise schedule.
// All arrays are length T, on GPU.
struct NoiseSchedule {
    int T = 0;

    // Per-timestep, length T:
    Tensor beta;                  // β_t
    Tensor alpha;                 // 1 - β_t
    Tensor alpha_bar;             // ∏ α_s for s ≤ t
    Tensor sqrt_alpha_bar;        // √αbar_t
    Tensor sqrt_one_minus_ab;     // √(1 - αbar_t)

    // For the reverse process (used in samplers later):
    Tensor sqrt_recip_alpha;      // 1 / √α_t  (used in DDPM step)
    Tensor posterior_variance;    // β̃_t
    Tensor posterior_log_var;     // log β̃_t (clipped at t=1)

    // In the NoiseSchedule struct, after posterior_log_var:
Tensor posterior_mean_coef1;   // sqrt(αbar_{t-1}) * β_t / (1 - αbar_t)
Tensor posterior_mean_coef2;   // sqrt(α_t) * (1 - αbar_{t-1}) / (1 - αbar_t)
    NoiseSchedule() = default;

    // Linear schedule from beta_start to beta_end. Standard DDPM defaults
    // are 1e-4 and 0.02 with T=1000.
    void build(int T_in, float beta_start = 1e-4f, float beta_end = 2e-2f);
};

inline void NoiseSchedule::build(int T_in, float beta_start, float beta_end) {
    T = T_in;

    std::vector<float> b(T), a(T), ab(T), sab(T), somab(T);
    std::vector<float> sra(T), pvar(T), plv(T);

    float prev_ab = 1.0f;
    for (int t = 0; t < T; ++t) {
        // Linear interpolation between beta_start and beta_end
        float frac = (T == 1) ? 0.f : (float)t / (float)(T - 1);
        float beta_t = beta_start + frac * (beta_end - beta_start);

        float alpha_t = 1.0f - beta_t;
        float ab_t = prev_ab * alpha_t;

        b[t]    = beta_t;
        a[t]    = alpha_t;
        ab[t]   = ab_t;
        sab[t]  = std::sqrt(ab_t);
        somab[t] = std::sqrt(1.0f - ab_t);

        sra[t]  = 1.0f / std::sqrt(alpha_t);

        // Posterior variance β̃_t = (1 - αbar_{t-1}) / (1 - αbar_t) * β_t
        // For t = 0, the formula is undefined (no t-1); we set β̃_0 = β_0
        // as a sentinel. Samplers use t > 0 so this rarely matters.
        float ab_prev = (t == 0) ? 1.0f : ab[t - 1];
        float pvar_t = (1.0f - ab_prev) / (1.0f - ab_t) * beta_t;
        // For t=0, the (1 - 1) numerator gives 0; clamp to a tiny floor for log.
        if (t == 0) pvar_t = beta_t;  // sentinel; samplers shouldn't use t=0
        pvar[t] = pvar_t;
        plv[t]  = std::log(std::max(pvar_t, 1e-20f));

        prev_ab = ab_t;
        std::vector<float> pmc1(T), pmc2(T);
for (int t = 0; t < T; ++t) {
    float ab_prev = (t == 0) ? 1.0f : ab[t - 1];
    float ab_t    = ab[t];
    float beta_t  = b[t];
    float alpha_t = a[t];

    // 1 - αbar_t in denominator. Avoid division by ~0 at t=0.
    float denom = 1.0f - ab_t;
    if (denom < 1e-20f) denom = 1e-20f;

    pmc1[t] = std::sqrt(ab_prev) * beta_t / denom;
    pmc2[t] = std::sqrt(alpha_t) * (1.0f - ab_prev) / denom;
}
posterior_mean_coef1 = Tensor({T});  posterior_mean_coef1.from_host(pmc1);
posterior_mean_coef2 = Tensor({T});  posterior_mean_coef2.from_host(pmc2);
    }

    // Allocate tensors and upload
    beta              = Tensor({T}); beta.from_host(b);
    alpha             = Tensor({T}); alpha.from_host(a);
    alpha_bar         = Tensor({T}); alpha_bar.from_host(ab);
    sqrt_alpha_bar    = Tensor({T}); sqrt_alpha_bar.from_host(sab);
    sqrt_one_minus_ab = Tensor({T}); sqrt_one_minus_ab.from_host(somab);
    sqrt_recip_alpha  = Tensor({T}); sqrt_recip_alpha.from_host(sra);
    posterior_variance = Tensor({T}); posterior_variance.from_host(pvar);
    posterior_log_var = Tensor({T}); posterior_log_var.from_host(plv);
}

