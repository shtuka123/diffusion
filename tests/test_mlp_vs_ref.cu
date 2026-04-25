#include "tensor.h"
#include "tensor_io.h"
#include "mlp.h"
#include "cuda_utils.h"
#include <cstdio>

int main() {
    std::printf("=== MLP forward + backward vs PyTorch ===\n");

    Tensor x  = load_tensor("data/ref/mlp_x.bin");
    Tensor W1 = load_tensor("data/ref/mlp_W1.bin");
    Tensor b1 = load_tensor("data/ref/mlp_b1.bin");
    Tensor W2 = load_tensor("data/ref/mlp_W2.bin");
    Tensor b2 = load_tensor("data/ref/mlp_b2.bin");
    Tensor expected_y = load_tensor("data/ref/mlp_y.bin");

    int B = x.size(0), T = x.size(1), D = x.size(2);
    int D_hidden = W1.size(1);

    MLPCache cache;
    cache.h_pre  = Tensor({B, T, D_hidden});
    cache.h_post = Tensor({B, T, D_hidden});

    Tensor y(x.shape);
    mlp_forward(y, cache, x, W1, b1, W2, b2);

    float err_fwd = max_abs_diff(y, expected_y);
    std::printf("  forward max abs err: %.3e  %s\n",
                err_fwd, err_fwd < 1e-4f ? "PASS" : "FAIL");

    int fails = (err_fwd >= 1e-4f) ? 1 : 0;

    // Backward
    Tensor dy = load_tensor("data/ref/mlp_dy.bin");
    Tensor expected_dx  = load_tensor("data/ref/mlp_dx.bin");
    Tensor expected_dW1 = load_tensor("data/ref/mlp_dW1.bin");
    Tensor expected_db1 = load_tensor("data/ref/mlp_db1.bin");
    Tensor expected_dW2 = load_tensor("data/ref/mlp_dW2.bin");
    Tensor expected_db2 = load_tensor("data/ref/mlp_db2.bin");

    Tensor dx(x.shape);
    Tensor dW1({D, D_hidden}), db1({D_hidden});
    Tensor dW2({D_hidden, D}), db2({D});
    mlp_backward(dx, dW1, db1, dW2, db2, dy, x, cache, W1, W2);

    float err_dx  = max_abs_diff(dx,  expected_dx);
    float err_dW1 = max_abs_diff(dW1, expected_dW1);
    float err_db1 = max_abs_diff(db1, expected_db1);
    float err_dW2 = max_abs_diff(dW2, expected_dW2);
    float err_db2 = max_abs_diff(db2, expected_db2);

    std::printf("  dx  max abs err: %.3e  %s\n",
                err_dx, err_dx < 1e-3f ? "PASS" : "FAIL");
    std::printf("  dW1 max abs err: %.3e  %s\n",
                err_dW1, err_dW1 < 1e-3f ? "PASS" : "FAIL");
    std::printf("  db1 max abs err: %.3e  %s\n",
                err_db1, err_db1 < 1e-3f ? "PASS" : "FAIL");
    std::printf("  dW2 max abs err: %.3e  %s\n",
                err_dW2, err_dW2 < 1e-3f ? "PASS" : "FAIL");
    std::printf("  db2 max abs err: %.3e  %s\n",
                err_db2, err_db2 < 1e-3f ? "PASS" : "FAIL");

    if (err_dx  >= 1e-3f) ++fails;
    if (err_dW1 >= 1e-3f) ++fails;
    if (err_db1 >= 1e-3f) ++fails;
    if (err_dW2 >= 1e-3f) ++fails;
    if (err_db2 >= 1e-3f) ++fails;

    return fails == 0 ? 0 : 1;
}