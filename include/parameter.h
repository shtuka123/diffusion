#pragma once
#include "tensor.h"
#include <vector>

// A parameter: weights plus the buffer for its gradient. Same shape.
//
// Design: the weights and their gradient are conceptually one thing during
// training. Bundling them eliminates a category of bugs (passing the wrong
// Tensor to the optimizer, forgetting to zero a grad buffer, etc).
struct Parameter {
    Tensor w;
    Tensor grad;

    Parameter() = default;

    explicit Parameter(std::vector<int> shape)
        : w(shape), grad(shape)
    {
        // Gradients start at zero — the first backward pass accumulates into them.
        CUDA_CHECK(cudaMemset(grad.data, 0, grad.numel * sizeof(float)));
    }

    // Non-copyable (Tensor is non-copyable)
    Parameter(const Parameter&) = delete;
    Parameter& operator=(const Parameter&) = delete;

    // Movable
    Parameter(Parameter&&) = default;
    Parameter& operator=(Parameter&&) = default;
};

// Zero all gradient buffers in a list of parameters.
inline void zero_grads(const std::vector<Parameter*>& params) {
    for (auto* p : params) {
        CUDA_CHECK(cudaMemset(p->grad.data, 0, p->grad.numel * sizeof(float)));
    }
}