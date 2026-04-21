#include "tensor.h"
#include "tensor_io.h"
#include "linear.h"
#include "grad_check.h"
#include "cuda_utils.h"
#include <cstdio>

// Scalar loss = sum(y^2), so dL/dy = 2*y. We compute this ourselves.
__global__ void sum_squares_kernel(const float* y, float* loss, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += y[i] * y[i];
    *loss = s;
}

__global__ void double_kernel(const float* y, float* dy, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dy[i] = 2.f * y[i];
}

int main() {
    std::printf("=== Linear backward ===\n");

    // --- Part 1: Compare against PyTorch's exact gradients ---
    Tensor x  = load_tensor("data/ref/linear_x.bin");
    Tensor W  = load_tensor("data/ref/linear_W.bin");
    Tensor b  = load_tensor("data/ref/linear_b.bin");
    Tensor dy = load_tensor("data/ref/linear_dy.bin");
    Tensor expected_dx = load_tensor("data/ref/linear_dx.bin");
    Tensor expected_dW = load_tensor("data/ref/linear_dW.bin");
    Tensor expected_db = load_tensor("data/ref/linear_db.bin");

    int B = x.size(0), D_in = x.size(1), D_out = W.size(1);

    Tensor dx({B, D_in}), dW({D_in, D_out}), db({D_out});
    linear_backward(dx, dW, db, x, W, dy);

    float err_dx = max_abs_diff(dx, expected_dx);
    float err_dW = max_abs_diff(dW, expected_dW);
    float err_db = max_abs_diff(db, expected_db);

    std::printf("  dx max abs error: %.3e  %s\n", err_dx, err_dx < 1e-3f ? "PASS" : "FAIL");
    std::printf("  dW max abs error: %.3e  %s\n", err_dW, err_dW < 1e-3f ? "PASS" : "FAIL");
    std::printf("  db max abs error: %.3e  %s\n", err_db, err_db < 1e-3f ? "PASS" : "FAIL");

    int fails = 0;
    if (err_dx >= 1e-3f) ++fails;
    if (err_dW >= 1e-3f) ++fails;
    if (err_db >= 1e-3f) ++fails;

    // --- Part 2: Numerical gradient check on dx ---
    // Tiny inputs for grad_check speed.
    std::printf("\n=== Numerical grad-check on dx ===\n");

    int Bc = 3, Dinc = 4, Doutc = 2;
    Tensor x_c  = Tensor::randn({Bc, Dinc}, 7);
    Tensor W_c  = Tensor::randn({Dinc, Doutc}, 11);
    Tensor b_c  = Tensor::randn({Doutc}, 13);
    Tensor y_c({Bc, Doutc});
    Tensor dy_c({Bc, Doutc});
    Tensor loss({1});

    auto forward = [&](Tensor& x_in) -> float {
        linear_forward(y_c, x_in, W_c, b_c);
        sum_squares_kernel<<<1, 1>>>(y_c.data, loss.data, (int)y_c.numel);
        CUDA_CHECK_KERNEL();
        auto h = loss.to_host();
        return h[0];
    };

    // Compute dy at the current x (we need it for analytic dx)
    forward(x_c);
    int n = (int)y_c.numel;
    int blk = 256, grd = (n + blk - 1) / blk;
    double_kernel<<<grd, blk>>>(y_c.data, dy_c.data, n);
    CUDA_CHECK_KERNEL();

    // Analytic dx via our backward
    Tensor dx_c({Bc, Dinc}), dW_c({Dinc, Doutc}), db_c({Doutc});
    linear_backward(dx_c, dW_c, db_c, x_c, W_c, dy_c);

    float rel = grad_check(forward, x_c, dx_c, 1e-3f, false);
    std::printf("Max relative error (dx): %.3e  %s\n",
                rel, rel < 1e-2f ? "PASS" : "FAIL");
    if (rel >= 1e-2f) ++fails;

    if (fails == 0) {
        std::printf("\nAll linear backward tests passed.\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n%d test(s) failed.\n", fails);
        return 1;
    }
}