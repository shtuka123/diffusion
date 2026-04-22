#include "tensor.h"
#include "tensor_io.h"
#include "mha.h"
#include "cuda_utils.h"
#include <cstdio>

int main() {
    std::printf("=== MHA forward vs PyTorch ===\n");

    Tensor x = load_tensor("data/ref/mha_x.bin");
    Tensor W_Q = load_tensor("data/ref/mha_W_Q.bin");
    Tensor b_Q = load_tensor("data/ref/mha_b_Q.bin");
    Tensor W_K = load_tensor("data/ref/mha_W_K.bin");
    Tensor b_K = load_tensor("data/ref/mha_b_K.bin");
    Tensor W_V = load_tensor("data/ref/mha_W_V.bin");
    Tensor b_V = load_tensor("data/ref/mha_b_V.bin");
    Tensor W_O = load_tensor("data/ref/mha_W_O.bin");
    Tensor b_O = load_tensor("data/ref/mha_b_O.bin");
    Tensor expected_y = load_tensor("data/ref/mha_y.bin");

    int B = x.size(0), T = x.size(1), D = x.size(2);
    int n_heads = 4;
    int d_k = D / n_heads;

    std::printf("  shapes: B=%d T=%d D=%d n_heads=%d d_k=%d\n",
                B, T, D, n_heads, d_k);

    Tensor y({B, T, D});

    // Allocate the MHA cache
    MHACache cache;
    cache.Q_raw = Tensor({B, T, D});
    cache.K_raw = Tensor({B, T, D});
    cache.V_raw = Tensor({B, T, D});
    cache.Q_heads = Tensor({B, n_heads, T, d_k});
    cache.K_heads = Tensor({B, n_heads, T, d_k});
    cache.V_heads = Tensor({B, n_heads, T, d_k});
    cache.attn_out_heads = Tensor({B, n_heads, T, d_k});
    cache.attn_out_flat = Tensor({B, T, D});
    cache.P = Tensor({B * n_heads, T, T});
    cache.S_buf = Tensor({B * n_heads, T, T});

    mha_forward(y, cache, x,
                W_Q, b_Q, W_K, b_K, W_V, b_V,
                W_O, b_O, n_heads);

    float err = max_abs_diff(y, expected_y);
    std::printf("  max abs error vs PyTorch: %.3e\n", err);

    if (err < 1e-4f) {
        std::printf("  PASS\n");
        return 0;
    } else {
        std::fprintf(stderr, "  FAIL (tol=1e-4)\n");
        return 1;
    }
}