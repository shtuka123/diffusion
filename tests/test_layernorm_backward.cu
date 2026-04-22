#include "tensor.h"
#include "tensor_io.h"
#include "layernorm.h"
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
    std::printf("=== LayerNorm backward ===\n");

    // Part 1: compare against PyTorch exactly
    Tensor x     = load_tensor("data/ref/ln_x.bin");
    Tensor gamma = load_tensor("data/ref/ln_gamma.bin");
    Tensor beta  = load_tensor("data/ref/ln_beta.bin");
    Tensor dy    = load_tensor("data/ref/ln_dy.bin");
    Tensor expected_dx     = load_tensor("data/ref/ln_dx.bin");
    Tensor expected_dgamma = load_tensor("data/ref/ln_dgamma.bin");
    Tensor expected_dbeta  = load_tensor("data/ref/ln_dbeta.bin");

    int B = x.size(0), T = x.size(1), D = x.size(2);
    int N = B * T;

    // Forward to get rstd, then backward
    Tensor y(x.shape);
    Tensor rstd({N});
    layernorm_forward(y, rstd, x, gamma, beta);

    Tensor dx(x.shape), dgamma({D}), dbeta({D});
    layernorm_backward(dx, dgamma, dbeta, dy, x, gamma, rstd);

    float err_dx     = max_abs_diff(dx,     expected_dx);
    float err_dgamma = max_abs_diff(dgamma, expected_dgamma);
    float err_dbeta  = max_abs_diff(dbeta,  expected_dbeta);

    std::printf("  dx max abs error:     %.3e  %s\n",
                err_dx,     err_dx     < 1e-4f ? "PASS" : "FAIL");
    std::printf("  dgamma max abs error: %.3e  %s\n",
                err_dgamma, err_dgamma < 1e-4f ? "PASS" : "FAIL");
    std::printf("  dbeta max abs error:  %.3e  %s\n",
                err_dbeta,  err_dbeta  < 1e-4f ? "PASS" : "FAIL");

    int fails = 0;
    if (err_dx     >= 1e-4f) ++fails;
    if (err_dgamma >= 1e-4f) ++fails;
    if (err_dbeta  >= 1e-4f) ++fails;

    // Part 2: grad_check on dx (tiny input, so numerical is fast)
    std::printf("\n=== LayerNorm backward: grad_check on dx ===\n");

    int Bc = 2, Tc = 2, Dc = 4;
    int Nc = Bc * Tc;
    Tensor x_c     = Tensor::randn({Bc, Tc, Dc}, 1);
    Tensor gamma_c = Tensor::randn({Dc}, 2);
    Tensor beta_c  = Tensor::randn({Dc}, 3);
    Tensor y_c({Bc, Tc, Dc});
    Tensor rstd_c({Nc});
    Tensor loss({1});

    auto forward = [&](Tensor& x_in) -> float {
        layernorm_forward(y_c, rstd_c, x_in, gamma_c, beta_c);
        sum_squares_kernel<<<1, 1>>>(y_c.data, loss.data, (int)y_c.numel);
        CUDA_CHECK_KERNEL();
        return loss.to_host()[0];
    };

    // Analytic dx via our backward
    forward(x_c);  // populates y_c and rstd_c
    Tensor dy_c(y_c.shape);
    {
        auto y_host = y_c.to_host();
        std::vector<float> dy_host(y_c.numel);
        for (size_t i = 0; i < y_c.numel; ++i) dy_host[i] = 2.f * y_host[i];
        dy_c.from_host(dy_host);
    }

    Tensor dx_c({Bc, Tc, Dc}), dgamma_c({Dc}), dbeta_c({Dc});
    layernorm_backward(dx_c, dgamma_c, dbeta_c, dy_c, x_c, gamma_c, rstd_c);

    float rel = grad_check(forward, x_c, dx_c, 1e-2f, false);
    std::printf("Max relative error (dx): %.3e  %s\n",
                rel, rel < 1e-2f ? "PASS" : "FAIL");
    if (rel >= 1e-2f) ++fails;

    if (fails == 0) {
        std::printf("\nAll LayerNorm backward tests passed.\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n%d failed.\n", fails);
        return 1;
    }
}