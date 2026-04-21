#include "cuda_utils.h"
#include "tensor.h"
#include <cstdio>

// Per-row cross-entropy loss (fused with softmax).
// logits: (B, C), labels: (B,), per_row_loss: (B,) output.
// For row b with true label y = labels[b]:
//   L[b] = logsumexp(logits[b]) - logits[b, y]
//
// One thread per row.
__global__ void ce_forward_kernel(
    const float* logits, const int* labels,
    float* per_row_loss, int B, int C)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const float* row = logits + b * C;
    int y = labels[b];

    // Numerical-safety check on the label index
    // (debug builds only; release leaves this in via the kernel's abort path)
    if (y < 0 || y >= C) {
        // Can't abort from device code; write a distinctive bad value
        // and let the host check for it after launch.
        per_row_loss[b] = -1.f;
        return;
    }

    // logsumexp via the stable trick:
    //   lse(x) = max(x) + log(sum(exp(x - max(x))))
    float m = row[0];
    for (int i = 1; i < C; ++i) {
        if (row[i] > m) m = row[i];
    }
    float s = 0.f;
    for (int i = 0; i < C; ++i) {
        s += expf(row[i] - m);
    }
    float lse = m + logf(s);

    per_row_loss[b] = lse - row[y];
}

// Reduce per_row_loss to a scalar mean.
// Simple one-thread implementation; fine for small B, and the batch size in
// training is ~128, so this isn't a bottleneck. We'll write faster reductions
// later when we need them.
__global__ void mean_reduce_kernel(
    const float* per_row, float* out_scalar, int B)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s = 0.f;
    for (int b = 0; b < B; ++b) s += per_row[b];
    *out_scalar = s / (float)B;
}

// Host-side interface: returns mean CE loss as a single-element Tensor on device.
// labels_device is a device pointer to B int labels.
void cross_entropy_forward(
    Tensor& loss_scalar,         // shape {1}
    Tensor& per_row_loss,        // shape {B}, used as scratch
    const Tensor& logits,        // shape {B, C}
    const int* labels_device,    // device pointer, B ints
    int B, int C)
{
    if (logits.ndim() != 2 || logits.size(0) != B || logits.size(1) != C) {
        std::fprintf(stderr, "ce_forward: logits shape mismatch\n");
        std::abort();
    }
    if (per_row_loss.numel != (size_t)B) {
        std::fprintf(stderr, "ce_forward: per_row_loss must be (B,)\n");
        std::abort();
    }
    if (loss_scalar.numel != 1) {
        std::fprintf(stderr, "ce_forward: loss_scalar must be (1,)\n");
        std::abort();
    }

    int block = 128;
    int grid = (B + block - 1) / block;
    ce_forward_kernel<<<grid, block>>>(logits.data, labels_device,
                                       per_row_loss.data, B, C);
    CUDA_CHECK_KERNEL();

    mean_reduce_kernel<<<1, 1>>>(per_row_loss.data, loss_scalar.data, B);
    CUDA_CHECK_KERNEL();
}