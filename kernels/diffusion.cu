#include "cuda_utils.h"
#include "tensor.h"
#include "noise_schedule.h"

#include <cstdio>

// q_sample: produce noisy samples x_t given clean x_0, timesteps t, and noise ε.
//
// Shapes:
//   x_0:  (B, ...flat...) — the data, batch-first, anything after dim 0
//   t:    (B,) device int* — one timestep per batch element
//   eps:  same shape as x_0
//   x_t:  output, same shape as x_0
//
// sqrt_ab, sqrt_omab: (T,) — schedule arrays
//
// Each thread handles one element of x_t. We figure out which batch element
// it belongs to from the thread's index, then look up the schedule using t[b].
__global__ void q_sample_kernel(
    const float* x0, const float* eps, const int* timesteps,
    const float* sqrt_ab, const float* sqrt_omab,
    float* xt,
    int B, int per_batch)   // per_batch = numel / B
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * per_batch;
    if (tid >= total) return;

    int b = tid / per_batch;
    int t_idx = timesteps[b];

    float sa  = sqrt_ab[t_idx];
    float som = sqrt_omab[t_idx];
    xt[tid] = sa * x0[tid] + som * eps[tid];
}

void q_sample(
    Tensor& xt,
    const Tensor& x0, const Tensor& eps,
    const int* timesteps_device,
    const NoiseSchedule& sched)
{
    if (xt.numel != x0.numel || xt.numel != eps.numel) {
        std::fprintf(stderr, "q_sample: shape mismatch\n");
        std::abort();
    }
    int B = x0.size(0);
    int per_batch = (int)(x0.numel / B);

    int total = B * per_batch;
    int block = 256;
    int grid = (total + block - 1) / block;
    q_sample_kernel<<<grid, block>>>(
        x0.data, eps.data, timesteps_device,
        sched.sqrt_alpha_bar.data, sched.sqrt_one_minus_ab.data,
        xt.data, B, per_batch);
    CUDA_CHECK_KERNEL();
}

// MSE loss between predicted noise and true noise.
// Inputs: pred, target both (B, D) or any shape with same numel.
// Output: scalar loss = mean((pred - target)^2)
//         dpred = 2 * (pred - target) / numel
//
// Two kernels: one to compute per-element squared diffs and reduce,
// one to compute the gradient.

// Reduction kernel — single thread, sums all (pred - target)^2 / numel.
__global__ void mse_loss_kernel(
    const float* pred, const float* target, float* loss, int n)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = pred[i] - target[i];
        s += d * d;
    }
    *loss = s / (float)n;
}

void mse_loss_forward(Tensor& loss, const Tensor& pred, const Tensor& target) {
    if (pred.numel != target.numel || loss.numel != 1) {
        std::fprintf(stderr, "mse_loss_forward: shape mismatch\n");
        std::abort();
    }
    int n = (int)pred.numel;
    mse_loss_kernel<<<1, 1>>>(pred.data, target.data, loss.data, n);
    CUDA_CHECK_KERNEL();
}

// Gradient: dL/dpred = 2 * (pred - target) / n
__global__ void mse_loss_backward_kernel(
    const float* pred, const float* target, float* dpred,
    float inv_n, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dpred[i] = 2.f * (pred[i] - target[i]) * inv_n;
}

void mse_loss_backward(Tensor& dpred, const Tensor& pred, const Tensor& target) {
    if (dpred.numel != pred.numel || pred.numel != target.numel) {
        std::fprintf(stderr, "mse_loss_backward: shape mismatch\n");
        std::abort();
    }
    int n = (int)pred.numel;
    float inv_n = 1.f / (float)n;
    int block = 256;
    int grid = (n + block - 1) / block;
    mse_loss_backward_kernel<<<grid, block>>>(
        pred.data, target.data, dpred.data, inv_n, n);
    CUDA_CHECK_KERNEL();
}

// One DDPM reverse step (with x_0 parameterization).
//
//   x_{t-1} = coef1_t * x_0_hat + coef2_t * x_t + sqrt(post_var_t) * z
//
// where z ~ N(0, I) for t > 0, and z = 0 for t = 0 (last step is deterministic).
//
// Inputs:
//   x_t:      (B, D)  current sample
//   x_0_hat:  (B, D)  model's prediction of x_0 at this t
//   z:        (B, D)  fresh noise (caller provides; for t=0 caller passes zeros)
//   t:        scalar int (the current timestep — same for entire batch)
//   schedule arrays: per-t coefficients
//
// Output:
//   x_prev: (B, D)  x_{t-1}
__global__ void ddpm_step_kernel(
    const float* x_t, const float* x_0_hat, const float* z,
    const float* pmc1, const float* pmc2, const float* post_var,
    int t, float* x_prev, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float c1 = pmc1[t];
    float c2 = pmc2[t];
    float pv = post_var[t];
    float noise_scale = (t > 0) ? sqrtf(pv) : 0.0f;

    x_prev[i] = c1 * x_0_hat[i] + c2 * x_t[i] + noise_scale * z[i];
}

void ddpm_step(
    Tensor& x_prev,
    const Tensor& x_t, const Tensor& x_0_hat, const Tensor& z,
    int t, const NoiseSchedule& sched)
{
    if (x_prev.numel != x_t.numel || x_t.numel != x_0_hat.numel ||
        x_t.numel != z.numel) {
        std::fprintf(stderr, "ddpm_step: shape mismatch\n");
        std::abort();
    }
    int n = (int)x_t.numel;
    int block = 256;
    int grid = (n + block - 1) / block;
    ddpm_step_kernel<<<grid, block>>>(
        x_t.data, x_0_hat.data, z.data,
        sched.posterior_mean_coef1.data, sched.posterior_mean_coef2.data,
        sched.posterior_variance.data,
        t, x_prev.data, n);
    CUDA_CHECK_KERNEL();
}

// One DDIM (η=0, deterministic) reverse step with x_0 parameterization.
//
// Given x_t and the model's prediction x_0_hat, compute x_{t_prev}.
// t_prev can be any timestep < t (typically t - stride).
//
//   ε̂ = (x_t - sqrt(αbar_t) * x_0_hat) / sqrt(1 - αbar_t)
//   x_{t_prev} = sqrt(αbar_{t_prev}) * x_0_hat + sqrt(1 - αbar_{t_prev}) * ε̂
//
// For t_prev = -1 (i.e., we're at the last step and want x_0), use αbar = 1
// (the data is undisturbed). This makes the formula reduce to x_0_hat.
__global__ void ddim_step_kernel(
    const float* x_t, const float* x_0_hat,
    float sqrt_ab_t, float sqrt_omab_t,
    float sqrt_ab_prev, float sqrt_omab_prev,
    float* x_prev, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float eps_hat = (x_t[i] - sqrt_ab_t * x_0_hat[i]) / sqrt_omab_t;
    x_prev[i] = sqrt_ab_prev * x_0_hat[i] + sqrt_omab_prev * eps_hat;
}

void ddim_step(
    Tensor& x_prev,
    const Tensor& x_t, const Tensor& x_0_hat,
    int t, int t_prev,
    const NoiseSchedule& sched)
{
    if (x_prev.numel != x_t.numel || x_t.numel != x_0_hat.numel) {
        std::fprintf(stderr, "ddim_step: shape mismatch\n");
        std::abort();
    }

    // Look up coefficients on host (these are tiny lookups, no kernel needed)
    auto h_sab  = sched.sqrt_alpha_bar.to_host();
    auto h_somab = sched.sqrt_one_minus_ab.to_host();

    float sqrt_ab_t = h_sab[t];
    float sqrt_omab_t = h_somab[t];

    float sqrt_ab_prev, sqrt_omab_prev;
    if (t_prev < 0) {
        // No more denoising — output should equal x_0_hat exactly
        sqrt_ab_prev = 1.0f;
        sqrt_omab_prev = 0.0f;
    } else {
        sqrt_ab_prev = h_sab[t_prev];
        sqrt_omab_prev = h_somab[t_prev];
    }

    int n = (int)x_t.numel;
    int block = 256;
    int grid = (n + block - 1) / block;
    ddim_step_kernel<<<grid, block>>>(
        x_t.data, x_0_hat.data,
        sqrt_ab_t, sqrt_omab_t,
        sqrt_ab_prev, sqrt_omab_prev,
        x_prev.data, n);
    CUDA_CHECK_KERNEL();
}