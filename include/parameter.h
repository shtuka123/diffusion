#pragma once
#include "tensor.h"
#include "cuda_utils.h"
#include <vector>

struct Parameter {
    Tensor w;
    Tensor grad;

    // Adam state (allocated lazily by the optimizer if needed)
    Tensor adam_m;
    Tensor adam_v;
    bool adam_initialized = false;

    Parameter() = default;

    explicit Parameter(std::vector<int> shape)
        : w(shape), grad(shape)
    {
        CUDA_CHECK(cudaMemset(grad.data, 0, grad.numel * sizeof(float)));
    }

    Parameter(const Parameter&) = delete;
    Parameter& operator=(const Parameter&) = delete;
    Parameter(Parameter&&) = default;
    Parameter& operator=(Parameter&&) = default;

    // Allocate m and v buffers, zeroed. Called by Adam on first use.
    void init_adam_state() {
        if (adam_initialized) return;
        adam_m = Tensor(w.shape);
        adam_v = Tensor(w.shape);
        CUDA_CHECK(cudaMemset(adam_m.data, 0, adam_m.numel * sizeof(float)));
        CUDA_CHECK(cudaMemset(adam_v.data, 0, adam_v.numel * sizeof(float)));
        adam_initialized = true;
    }
};

inline void zero_grads(const std::vector<Parameter*>& params) {
    for (auto* p : params) {
        CUDA_CHECK(cudaMemset(p->grad.data, 0, p->grad.numel * sizeof(float)));
    }
}