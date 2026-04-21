#pragma once
#include "parameter.h"
#include <random>
#include <cmath>
#include <cstdint>

// He (Kaiming) normal init for a Linear layer's weight tensor (D_in, D_out).
// w ~ N(0, sqrt(2 / D_in)). Biases separately; init to zero with init_zero().
inline void init_he_normal(Parameter& p, int fan_in, uint64_t seed = 42) {
    float std_dev = std::sqrt(2.0f / (float)fan_in);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.f, std_dev);

    std::vector<float> host(p.w.numel);
    for (size_t i = 0; i < host.size(); ++i) host[i] = dist(rng);
    p.w.from_host(host);
}

// Fill with zeros — used for biases.
inline void init_zero(Parameter& p) {
    CUDA_CHECK(cudaMemset(p.w.data, 0, p.w.numel * sizeof(float)));
}

// Normal init with explicit std, for cases where He doesn't apply.
inline void init_normal(Parameter& p, float std_dev, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.f, std_dev);

    std::vector<float> host(p.w.numel);
    for (size_t i = 0; i < host.size(); ++i) host[i] = dist(rng);
    p.w.from_host(host);
}