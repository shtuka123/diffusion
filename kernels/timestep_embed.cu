#include "cuda_utils.h"
#include "tensor.h"
#include <cmath>
#include <cstdio>

// Sinusoidal timestep embedding (Vaswani-style).
//
// Input:
//   timesteps: (B,) device int* — one integer t per batch element
// Output:
//   emb: (B, d) — sinusoidal embedding
//
// emb[b, 2k]   = sin(t * freq_k)
// emb[b, 2k+1] = cos(t * freq_k)
//   where freq_k = exp(-(2k/d) * ln(10000))   for k = 0, ..., d/2 - 1
//
// One thread per output element.
__global__ void sinusoidal_embed_kernel(
    const int* timesteps, float* emb,
    int B, int d, float log_max_period)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * d;
    if (tid >= total) return;

    int b = tid / d;
    int dim = tid % d;

    int t = timesteps[b];
    int half = d / 2;

    // Index k in [0, half) and whether this is sin (even) or cos (odd).
    int k = dim / 2;
    bool is_cos = (dim % 2) == 1;

    float frac = (float)k / (float)half;
    float freq = expf(-frac * log_max_period);
    float arg = (float)t * freq;
    emb[tid] = is_cos ? cosf(arg) : sinf(arg);
}

void sinusoidal_embed(
    Tensor& emb,
    const int* timesteps_device, int B,
    float max_period = 10000.0f)
{
    if (emb.ndim() != 2 || emb.size(0) != B) {
        std::fprintf(stderr, "sinusoidal_embed: expected emb shape (B=%d, d), got %dD\n",
                     B, emb.ndim());
        std::abort();
    }
    int d = emb.size(1);
    if (d % 2 != 0) {
        std::fprintf(stderr, "sinusoidal_embed: d (%d) must be even\n", d);
        std::abort();
    }

    int total = B * d;
    int block = 256;
    int grid = (total + block - 1) / block;
    sinusoidal_embed_kernel<<<grid, block>>>(
        timesteps_device, emb.data, B, d, std::log(max_period));
    CUDA_CHECK_KERNEL();
}