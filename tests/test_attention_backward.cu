#include "tensor.h"
#include "grad_check.h"
#include "attention.h"
#include "cuda_utils.h"
#include <cstdio>

// Test scalar loss: L = sum(Y^2), so dL/dY = 2*Y.
__global__ void sum_squares_kernel(const float* y, float* loss, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += y[i] * y[i];
    *loss = s;
}

__global__ void double_kernel(const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 2.f * y[i];
}

int main() {
    std::printf("=== Attention backward: grad check ===\n");

    // Tiny problem: B=2, T=4, d_k=4, d_v=4.
    // Total params across Q, K, V: 3*B*T*d_k = 96. grad_check does 192 forward
    // passes — should take a fraction of a second.
    int B = 2, T = 4, d_k = 4, d_v = 4;

    Tensor Q = Tensor::randn({B, T, d_k}, 1);
    Tensor K = Tensor::randn({B, T, d_k}, 2);
    Tensor V = Tensor::randn({B, T, d_v}, 3);

    Tensor Y({B, T, d_v});
    Tensor P({B, T, T});
    Tensor S_buf({B, T, T});
    Tensor loss({1});

    // Forward closure that we can pass to grad_check.
    // It needs to take a single tensor and return a scalar loss, so we close
    // over K, V, etc. and vary Q. Then repeat for K, then for V.
    // Testing Q first:
    auto forward_vary_Q = [&](Tensor& Q_in) -> float {
        attention_forward(Y, P, Q_in, K, V, S_buf);
        sum_squares_kernel<<<1, 1>>>(Y.data, loss.data, (int)Y.numel);
        CUDA_CHECK_KERNEL();
        return loss.to_host()[0];
    };

    // Compute analytic gradients: run full forward, then backward.
    forward_vary_Q(Q);  // populates Y and P as a side effect

    // dY = 2 * Y
    Tensor dY({B, T, d_v});
    {
        int n = (int)Y.numel;
        int blk = 256, grd = (n + blk - 1) / blk;
        double_kernel<<<grd, blk>>>(Y.data, dY.data, n);
        CUDA_CHECK_KERNEL();
    }

    Tensor dQ({B, T, d_k}), dK({B, T, d_k}), dV({B, T, d_v});
    Tensor dP_buf({B, T, T}), dS_buf({B, T, T});
    attention_backward(dQ, dK, dV, dY, Q, K, V, P, dP_buf, dS_buf);

    // Grad-check dQ
    std::printf("\nGrad check dQ:\n");
    float err_Q = grad_check(forward_vary_Q, Q, dQ, 1e-2f, false);
    std::printf("  max relative error: %.3e  %s\n",
                err_Q, err_Q < 1e-2f ? "PASS" : "FAIL");

    // Grad-check dK
    auto forward_vary_K = [&](Tensor& K_in) -> float {
        attention_forward(Y, P, Q, K_in, V, S_buf);
        sum_squares_kernel<<<1, 1>>>(Y.data, loss.data, (int)Y.numel);
        CUDA_CHECK_KERNEL();
        return loss.to_host()[0];
    };
    std::printf("\nGrad check dK:\n");
    float err_K = grad_check(forward_vary_K, K, dK, 1e-2f, false);
    std::printf("  max relative error: %.3e  %s\n",
                err_K, err_K < 1e-2f ? "PASS" : "FAIL");

    // Grad-check dV
    auto forward_vary_V = [&](Tensor& V_in) -> float {
        attention_forward(Y, P, Q, K, V_in, S_buf);
        sum_squares_kernel<<<1, 1>>>(Y.data, loss.data, (int)Y.numel);
        CUDA_CHECK_KERNEL();
        return loss.to_host()[0];
    };
    std::printf("\nGrad check dV:\n");
    float err_V = grad_check(forward_vary_V, V, dV, 1e-2f, false);
    std::printf("  max relative error: %.3e  %s\n",
                err_V, err_V < 1e-2f ? "PASS" : "FAIL");

    int fails = 0;
    if (err_Q >= 1e-2f) ++fails;
    if (err_K >= 1e-2f) ++fails;
    if (err_V >= 1e-2f) ++fails;

    if (fails == 0) {
        std::printf("\nAll attention gradient checks passed.\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n%d failed.\n", fails);
        return 1;
    }
}