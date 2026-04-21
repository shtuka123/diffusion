#pragma once
#include "tensor.h"
#include "cuda_utils.h"
#include <functional>
#include <cstdio>
#include <cmath>

// Compare analytic gradient to numerical gradient via central differences.
//
//   forward_fn(x) -> scalar loss L (as a single-element Tensor on device)
//   dx_analytic:   the gradient your backward kernel computed, shape like x
//
// For each element of x, perturbs it by ±eps, calls forward_fn twice, computes
// num_grad = (L_plus - L_minus) / (2 * eps), compares to dx_analytic[i].
//
// Returns the max relative error across all elements.
//
// This is O(numel) forward passes — use on TINY inputs only (numel < 100).
inline float grad_check(
    std::function<float(Tensor&)> forward_fn,
    Tensor& x,
    const Tensor& dx_analytic,
    float eps = 1e-3f,
    bool verbose = false)
{
    auto x_host = x.to_host();
    auto dx_host = dx_analytic.to_host();

    float max_rel_err = 0.f;
    int max_idx = -1;
    float max_num = 0.f, max_ana = 0.f;

    for (size_t i = 0; i < x.numel; ++i) {
        float orig = x_host[i];

        x_host[i] = orig + eps;
        x.from_host(x_host);
        float L_plus = forward_fn(x);

        x_host[i] = orig - eps;
        x.from_host(x_host);
        float L_minus = forward_fn(x);

        x_host[i] = orig;  // restore
        x.from_host(x_host);

        float num_grad = (L_plus - L_minus) / (2.f * eps);
        float ana_grad = dx_host[i];

        // Relative error with small-value guard
        float denom = std::fabs(num_grad) + std::fabs(ana_grad) + 1e-8f;
        float rel = std::fabs(num_grad - ana_grad) / denom;

        if (verbose) {
            std::printf("  [%zu] num=% .6f  ana=% .6f  rel=%.3e\n",
                        i, num_grad, ana_grad, rel);
        }
        if (rel > max_rel_err) {
            max_rel_err = rel;
            max_idx = (int)i;
            max_num = num_grad;
            max_ana = ana_grad;
        }
    }

    if (max_idx >= 0) {
        std::printf("  Worst element: idx=%d, num=% .6f, ana=% .6f, rel=%.3e\n",
                    max_idx, max_num, max_ana, max_rel_err);
    }
    return max_rel_err;
}