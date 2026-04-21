#include "tensor.h"
#include "grad_check.h"
#include "elementwise.h"
#include "cuda_utils.h"
#include <cstdio>
#include <cmath>

// Simple scalar loss for testing: L = sum(relu(x)^2)
// This has a smooth, nontrivial dependence on x, so numerical
// gradients will be informative.
// Analytic: dL/dx = 2 * relu(x) * [x > 0] = 2 * x if x > 0 else 0
__global__ void square_sum_kernel(const float* y, float* loss, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += y[i] * y[i];
    *loss = s;
}

int main() {
    std::printf("=== ReLU backward: grad check ===\n");

    // Tiny input: 2x3 so grad_check doesn't take forever
    Tensor x({2, 3});
    x.from_host({-0.5f, 1.0f, -2.0f, 0.3f, -0.1f, 2.5f});

    Tensor y(x.shape);
    Tensor loss({1});

    // Forward closure: x -> relu -> sum(y^2) -> scalar
    auto forward = [&](Tensor& x_in) -> float {
        relu_forward(y, x_in);
        square_sum_kernel<<<1, 1>>>(y.data, loss.data, (int)y.numel);
        CUDA_CHECK_KERNEL();
        auto h = loss.to_host();
        return h[0];
    };

    // Get the loss at x (needed to compute dL/dy = 2y for the analytic check)
    forward(x);
    auto y_host = y.to_host();
    std::vector<float> dy_host(y.numel);
    for (size_t i = 0; i < y.numel; ++i) dy_host[i] = 2.f * y_host[i];

    Tensor dy(y.shape);
    dy.from_host(dy_host);

    // Compute analytic gradient via our backward kernel
    Tensor dx(x.shape);
    relu_backward(dx, dy, x);

    // Run grad_check
    float err = grad_check(forward, x, dx, 1e-3f, true);
    std::printf("\nMax relative error: %.3e\n", err);

    if (err < 1e-2f) {
        std::printf("PASS\n");
        return 0;
    } else {
        std::fprintf(stderr, "FAIL (tol=1e-2)\n");
        return 1;
    }
}