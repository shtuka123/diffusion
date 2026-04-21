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