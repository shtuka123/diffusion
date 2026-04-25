#include "tensor.h"
#include "tensor_io.h"
#include "elementwise.h"
#include "grad_check.h"
#include "cuda_utils.h"
#include <cstdio>

__global__ void sum_squares_kernel(const float* y, float* loss, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += y[i] * y[i];
    *loss = s;
}

int main() {
    std::printf("=== GELU forward + backward ===\n");

    // Forward and backward against PyTorch
    Tensor x          = load_tensor("data/ref/gelu_x.bin");
    Tensor expected_y = load_tensor("data/ref/gelu_y.bin");
    Tensor dy         = load_tensor("data/ref/gelu_dy.bin");
    Tensor expected_dx = load_tensor("data/ref/gelu_dx.bin");

    Tensor y(x.shape);
    gelu_forward(y, x);
    float err_y = max_abs_diff(y, expected_y);
    std::printf("  forward max abs err:  %.3e  %s\n",
                err_y, err_y < 1e-5f ? "PASS" : "FAIL");

    Tensor dx(x.shape);
    gelu_backward(dx, dy, x);
    float err_dx = max_abs_diff(dx, expected_dx);
    std::printf("  backward max abs err: %.3e  %s\n",
                err_dx, err_dx < 1e-4f ? "PASS" : "FAIL");

    int fails = 0;
    if (err_y  >= 1e-5f) ++fails;
    if (err_dx >= 1e-4f) ++fails;

    // Grad check on tiny input
    std::printf("\n=== GELU backward: grad_check ===\n");
    Tensor x_c = Tensor::randn({2, 5}, 7);
    Tensor y_c({2, 5});
    Tensor loss({1});
    auto forward = [&](Tensor& x_in) -> float {
        gelu_forward(y_c, x_in);
        sum_squares_kernel<<<1, 1>>>(y_c.data, loss.data, (int)y_c.numel);
        CUDA_CHECK_KERNEL();
        return loss.to_host()[0];
    };
    forward(x_c);
    auto y_host = y_c.to_host();
    std::vector<float> dy_host(y_c.numel);
    for (size_t i = 0; i < y_c.numel; ++i) dy_host[i] = 2.f * y_host[i];
    Tensor dy_c(y_c.shape); dy_c.from_host(dy_host);
    Tensor dx_c({2, 5});
    gelu_backward(dx_c, dy_c, x_c);

    float rel = grad_check(forward, x_c, dx_c, 1e-3f, false);
    std::printf("Max relative error: %.3e  %s\n",
                rel, rel < 1e-2f ? "PASS" : "FAIL");
    if (rel >= 1e-2f) ++fails;

    if (fails == 0) {
        std::printf("\nAll GELU tests passed.\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n%d failed.\n", fails);
        return 1;
    }
}