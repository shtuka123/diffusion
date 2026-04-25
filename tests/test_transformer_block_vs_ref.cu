#include "tensor.h"
#include "tensor_io.h"
#include "transformer_block.h"
#include <cstdio>
#include <vector>
#include <string>

static int check(const char* name, const Tensor& got, const Tensor& expected,
                 float tol)
{
    float err = max_abs_diff(got, expected);
    bool pass = err < tol;
    std::printf("  %-12s max abs err: %.3e  %s\n",
                name, err, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main() {
    std::printf("=== Transformer block forward + backward vs PyTorch ===\n");

    // Load all parameters and the upstream gradient.
    Tensor x       = load_tensor("data/ref/tb_x.bin");
    Tensor gamma1  = load_tensor("data/ref/tb_gamma1.bin");
    Tensor beta1   = load_tensor("data/ref/tb_beta1.bin");
    Tensor W_Q     = load_tensor("data/ref/tb_W_Q.bin");
    Tensor b_Q     = load_tensor("data/ref/tb_b_Q.bin");
    Tensor W_K     = load_tensor("data/ref/tb_W_K.bin");
    Tensor b_K     = load_tensor("data/ref/tb_b_K.bin");
    Tensor W_V     = load_tensor("data/ref/tb_W_V.bin");
    Tensor b_V     = load_tensor("data/ref/tb_b_V.bin");
    Tensor W_O     = load_tensor("data/ref/tb_W_O.bin");
    Tensor b_O     = load_tensor("data/ref/tb_b_O.bin");
    Tensor gamma2  = load_tensor("data/ref/tb_gamma2.bin");
    Tensor beta2   = load_tensor("data/ref/tb_beta2.bin");
    Tensor W1_mlp  = load_tensor("data/ref/tb_W1_mlp.bin");
    Tensor b1_mlp  = load_tensor("data/ref/tb_b1_mlp.bin");
    Tensor W2_mlp  = load_tensor("data/ref/tb_W2_mlp.bin");
    Tensor b2_mlp  = load_tensor("data/ref/tb_b2_mlp.bin");

    Tensor expected_y = load_tensor("data/ref/tb_y.bin");
    Tensor dy         = load_tensor("data/ref/tb_dy.bin");

    int B = x.size(0), T = x.size(1), D = x.size(2);
    int n_heads = 4;
    int d_k = D / n_heads;
    int D_mlp = 4 * D;

    // Allocate cache
    TransformerBlockCache cache;
    cache.t1 = Tensor({B, T, D});
    cache.rstd1 = Tensor({B * T});
    cache.a = Tensor({B, T, D});
    cache.r1 = Tensor({B, T, D});
    cache.t2 = Tensor({B, T, D});
    cache.rstd2 = Tensor({B * T});
    cache.m = Tensor({B, T, D});

    cache.mha_cache.Q_raw = Tensor({B, T, D});
    cache.mha_cache.K_raw = Tensor({B, T, D});
    cache.mha_cache.V_raw = Tensor({B, T, D});
    cache.mha_cache.Q_heads = Tensor({B, n_heads, T, d_k});
    cache.mha_cache.K_heads = Tensor({B, n_heads, T, d_k});
    cache.mha_cache.V_heads = Tensor({B, n_heads, T, d_k});
    cache.mha_cache.attn_out_heads = Tensor({B, n_heads, T, d_k});
    cache.mha_cache.attn_out_flat  = Tensor({B, T, D});
    cache.mha_cache.P     = Tensor({B * n_heads, T, T});
    cache.mha_cache.S_buf = Tensor({B * n_heads, T, T});

    cache.mlp_cache.h_pre  = Tensor({B, T, D_mlp});
    cache.mlp_cache.h_post = Tensor({B, T, D_mlp});

    Tensor y(x.shape);
    transformer_block_forward(
        y, cache, x,
        gamma1, beta1,
        W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O,
        gamma2, beta2,
        W1_mlp, b1_mlp, W2_mlp, b2_mlp,
        n_heads);

    int fails = 0;
    fails += check("y", y, expected_y, 1e-4f);

    // Backward
    Tensor dx({B, T, D});
    Tensor dgamma1({D}), dbeta1({D});
    Tensor dW_Q({D, D}), db_Q({D});
    Tensor dW_K({D, D}), db_K({D});
    Tensor dW_V({D, D}), db_V({D});
    Tensor dW_O({D, D}), db_O({D});
    Tensor dgamma2({D}), dbeta2({D});
    Tensor dW1_mlp({D, D_mlp}), db1_mlp({D_mlp});
    Tensor dW2_mlp({D_mlp, D}), db2_mlp({D});

    transformer_block_backward(
        dx,
        dgamma1, dbeta1,
        dW_Q, db_Q, dW_K, db_K, dW_V, db_V, dW_O, db_O,
        dgamma2, dbeta2,
        dW1_mlp, db1_mlp, dW2_mlp, db2_mlp,
        dy, x, cache,
        gamma1, W_Q, W_K, W_V, W_O,
        gamma2, W1_mlp, W2_mlp,
        n_heads);

    Tensor expected_dx       = load_tensor("data/ref/tb_dx.bin");
    Tensor expected_dgamma1  = load_tensor("data/ref/tb_dgamma1.bin");
    Tensor expected_dbeta1   = load_tensor("data/ref/tb_dbeta1.bin");
    Tensor expected_dW_Q     = load_tensor("data/ref/tb_dW_Q.bin");
    Tensor expected_db_Q     = load_tensor("data/ref/tb_db_Q.bin");
    Tensor expected_dW_K     = load_tensor("data/ref/tb_dW_K.bin");
    Tensor expected_db_K     = load_tensor("data/ref/tb_db_K.bin");
    Tensor expected_dW_V     = load_tensor("data/ref/tb_dW_V.bin");
    Tensor expected_db_V     = load_tensor("data/ref/tb_db_V.bin");
    Tensor expected_dW_O     = load_tensor("data/ref/tb_dW_O.bin");
    Tensor expected_db_O     = load_tensor("data/ref/tb_db_O.bin");
    Tensor expected_dgamma2  = load_tensor("data/ref/tb_dgamma2.bin");
    Tensor expected_dbeta2   = load_tensor("data/ref/tb_dbeta2.bin");
    Tensor expected_dW1_mlp  = load_tensor("data/ref/tb_dW1_mlp.bin");
    Tensor expected_db1_mlp  = load_tensor("data/ref/tb_db1_mlp.bin");
    Tensor expected_dW2_mlp  = load_tensor("data/ref/tb_dW2_mlp.bin");
    Tensor expected_db2_mlp  = load_tensor("data/ref/tb_db2_mlp.bin");

    fails += check("dx",       dx,       expected_dx,       1e-3f);
    fails += check("dgamma1",  dgamma1,  expected_dgamma1,  1e-3f);
    fails += check("dbeta1",   dbeta1,   expected_dbeta1,   1e-3f);
    fails += check("dW_Q",     dW_Q,     expected_dW_Q,     1e-3f);
    fails += check("db_Q",     db_Q,     expected_db_Q,     1e-3f);
    fails += check("dW_K",     dW_K,     expected_dW_K,     1e-3f);
    fails += check("db_K",     db_K,     expected_db_K,     1e-3f);
    fails += check("dW_V",     dW_V,     expected_dW_V,     1e-3f);
    fails += check("db_V",     db_V,     expected_db_V,     1e-3f);
    fails += check("dW_O",     dW_O,     expected_dW_O,     1e-3f);
    fails += check("db_O",     db_O,     expected_db_O,     1e-3f);
    fails += check("dgamma2",  dgamma2,  expected_dgamma2,  1e-3f);
    fails += check("dbeta2",   dbeta2,   expected_dbeta2,   1e-3f);
    fails += check("dW1_mlp",  dW1_mlp,  expected_dW1_mlp,  1e-3f);
    fails += check("db1_mlp",  db1_mlp,  expected_db1_mlp,  1e-3f);
    fails += check("dW2_mlp",  dW2_mlp,  expected_dW2_mlp,  1e-3f);
    fails += check("db2_mlp",  db2_mlp,  expected_db2_mlp,  1e-3f);

    if (fails == 0) {
        std::printf("\nAll Transformer block tests passed.\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n%d failed.\n", fails);
        return 1;
    }
}