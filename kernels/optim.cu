#include "cuda_utils.h"
#include "tensor.h"
#include "parameter.h"

// SGD step: w -= lr * grad, elementwise.
// The simplest optimizer there is. One kernel launch per parameter tensor.
__global__ void sgd_step_kernel(
    float* w, const float* grad, float lr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    w[i] -= lr * grad[i];
}

// Run SGD on one parameter.
void sgd_step(Parameter& p, float lr) {
    int n = (int)p.w.numel;
    int block = 256;
    int grid = (n + block - 1) / block;
    sgd_step_kernel<<<grid, block>>>(p.w.data, p.grad.data, lr, n);
    CUDA_CHECK_KERNEL();
}

// Run SGD on an entire parameter list.
void sgd_step_all(const std::vector<Parameter*>& params, float lr) {
    for (auto* p : params) sgd_step(*p, lr);
}

// Adam update, elementwise.
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g^2
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   w -= lr * m_hat / (sqrt(v_hat) + eps)
//
// Caller passes (1 - beta1^t) and (1 - beta2^t) precomputed since they're
// constant across the kernel and per-step on host.
__global__ void adam_step_kernel(
    float* w, float* m, float* v, const float* g,
    float lr, float beta1, float beta2,
    float bc1, float bc2,    // 1 - beta1^t, 1 - beta2^t
    float eps, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float gi = g[i];
    float mi = beta1 * m[i] + (1.f - beta1) * gi;
    float vi = beta2 * v[i] + (1.f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;

    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    w[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// Adam state for the whole training process. Holds the global step counter
// and hyperparameters. Pass by reference.
struct AdamState {
    int step = 0;
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
};

// One Adam step on a single parameter.
void adam_step(Parameter& p, AdamState& state) {
    p.init_adam_state();

    int n = (int)p.w.numel;
    int block = 256;
    int grid = (n + block - 1) / block;

    // Bias correction terms (computed on host once per step)
    float bc1 = 1.f - std::pow(state.beta1, (float)state.step);
    float bc2 = 1.f - std::pow(state.beta2, (float)state.step);

    adam_step_kernel<<<grid, block>>>(
        p.w.data, p.adam_m.data, p.adam_v.data, p.grad.data,
        state.lr, state.beta1, state.beta2,
        bc1, bc2, state.eps, n);
    CUDA_CHECK_KERNEL();
}

// Adam step for a list of parameters. Increments state.step before applying.
void adam_step_all(const std::vector<Parameter*>& params, AdamState& state) {
    state.step++;
    for (auto* p : params) adam_step(*p, state);
}