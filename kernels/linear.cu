#include "cuda_utils.h"
#include "tensor.h"
#include "matmul.h"
#include "elementwise.h"
#include <cstdio>

// y = x @ W + b
// Shapes: x (B, D_in), W (D_in, D_out), b (D_out,), y (B, D_out)
void linear_forward(Tensor& y,
                    const Tensor& x,
                    const Tensor& W,
                    const Tensor& b)
{
    if (x.ndim() != 2 || W.ndim() != 2 || b.ndim() != 1) {
        std::fprintf(stderr, "linear_forward: expected x(2D), W(2D), b(1D)\n");
        std::abort();
    }
    if (W.size(0) != x.size(1)) {
        std::fprintf(stderr, "linear_forward: x inner dim %d != W outer dim %d\n",
                     x.size(1), W.size(0));
        std::abort();
    }
    if (b.size(0) != W.size(1)) {
        std::fprintf(stderr, "linear_forward: b size %d != W.size(1) %d\n",
                     b.size(0), W.size(1));
        std::abort();
    }
    if (y.size(0) != x.size(0) || y.size(1) != W.size(1)) {
        std::fprintf(stderr, "linear_forward: y shape mismatch\n");
        std::abort();
    }

    // y = x @ W
    matmul(y, x, W);
    // y += broadcast(b)
    bias_add(y, y, b);
}

// dx = dy @ W^T
//
// dy shape: (B, D_out)
// W shape:  (D_in, D_out)    (we treat rows of W^T as columns of W)
// dx shape: (B, D_in)
//
// dx[b, i] = sum over j of dy[b, j] * W[i, j]
//
// One thread per output element, serial loop over j (the inner dim).
__global__ void linear_backward_dx_kernel(
    const float* dy, const float* W, float* dx,
    int B, int D_in, int D_out)
{
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || i >= D_in) return;

    float acc = 0.f;
    for (int j = 0; j < D_out; ++j) {
        acc += dy[b * D_out + j] * W[i * D_out + j];
    }
    dx[b * D_in + i] = acc;
}

// dW = x^T @ dy
//
// x shape:  (B, D_in)
// dy shape: (B, D_out)
// dW shape: (D_in, D_out)
//
// dW[i, j] = sum over b of x[b, i] * dy[b, j]
//
// One thread per output element, serial loop over b (the batch dim).
__global__ void linear_backward_dW_kernel(
    const float* x, const float* dy, float* dW,
    int B, int D_in, int D_out)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D_in || j >= D_out) return;

    float acc = 0.f;
    for (int b = 0; b < B; ++b) {
        acc += x[b * D_in + i] * dy[b * D_out + j];
    }
    dW[i * D_out + j] = acc;
}

// db[j] = sum over b of dy[b, j]
//
// dy shape: (B, D_out), db shape: (D_out,)
// One thread per output feature, serial loop over batch.
__global__ void linear_backward_db_kernel(
    const float* dy, float* db, int B, int D_out)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D_out) return;

    float s = 0.f;
    for (int b = 0; b < B; ++b) {
        s += dy[b * D_out + j];
    }
    db[j] = s;
}

// Host-side: all three gradients in one call.
void linear_backward(Tensor& dx, Tensor& dW, Tensor& db,
                     const Tensor& x, const Tensor& W, const Tensor& dy)
{
    int B     = x.size(0);
    int D_in  = x.size(1);
    int D_out = W.size(1);

    // Shape checks
    if (W.size(0) != D_in || dy.size(0) != B || dy.size(1) != D_out ||
        dx.size(0) != B || dx.size(1) != D_in ||
        dW.size(0) != D_in || dW.size(1) != D_out ||
        db.size(0) != D_out) {
        std::fprintf(stderr, "linear_backward: shape mismatch\n");
        std::abort();
    }

    // dx = dy @ W^T
    {
        dim3 block(16, 16);
        dim3 grid((D_in + 15) / 16, (B + 15) / 16);
        linear_backward_dx_kernel<<<grid, block>>>(
            dy.data, W.data, dx.data, B, D_in, D_out);
        CUDA_CHECK_KERNEL();
    }

    // dW = x^T @ dy
    {
        dim3 block(16, 16);
        dim3 grid((D_out + 15) / 16, (D_in + 15) / 16);
        linear_backward_dW_kernel<<<grid, block>>>(
            x.data, dy.data, dW.data, B, D_in, D_out);
        CUDA_CHECK_KERNEL();
    }

    // db = sum(dy, axis=0)
    {
        int block = 256;
        int grid = (D_out + block - 1) / block;
        linear_backward_db_kernel<<<grid, block>>>(
            dy.data, db.data, B, D_out);
        CUDA_CHECK_KERNEL();
    }
}