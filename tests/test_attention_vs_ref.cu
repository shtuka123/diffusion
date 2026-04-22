#include "tensor.h"
#include "tensor_io.h"
#include "attention.h"
#include <cstdio>

int main() {
    std::printf("=== Attention forward vs PyTorch ===\n");

    Tensor Q = load_tensor("data/ref/attn_Q.bin");
    Tensor K = load_tensor("data/ref/attn_K.bin");
    Tensor V = load_tensor("data/ref/attn_V.bin");
    Tensor expected_Y = load_tensor("data/ref/attn_Y.bin");

    int B = Q.size(0), T = Q.size(1);
    int d_k = Q.size(2), d_v = V.size(2);

    std::printf("  shapes: B=%d T=%d d_k=%d d_v=%d\n", B, T, d_k, d_v);

    Tensor Y({B, T, d_v});
    Tensor P({B, T, T});
    Tensor S_buf({B, T, T});

    attention_forward(Y, P, Q, K, V, S_buf);

    float err = max_abs_diff(Y, expected_Y);
    std::printf("  max abs error vs PyTorch: %.3e\n", err);

    // Sanity: rows of P (over valid positions) should sum to 1.
    auto h_P = P.to_host();
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < T; ++i) {
            float s = 0.f;
            for (int j = 0; j <= i; ++j) s += h_P[b * T * T + i * T + j];
            // Also verify masked positions are exactly zero.
            for (int j = i + 1; j < T; ++j) {
                float v = h_P[b * T * T + i * T + j];
                if (v != 0.f) {
                    std::fprintf(stderr, "  FAIL: P[%d, %d, %d] = %f, expected 0\n",
                                 b, i, j, v);
                    return 1;
                }
            }
            if (std::fabs(s - 1.0f) > 1e-5f) {
                std::fprintf(stderr, "  FAIL: row %d sum=%f, expected 1\n", i, s);
                return 1;
            }
        }
    }
    std::printf("  softmax rows sum to 1, masked positions are zero\n");

    if (err < 1e-4f) {
        std::printf("  PASS\n");
        return 0;
    } else {
        std::fprintf(stderr, "  FAIL (tol=1e-4)\n");
        return 1;
    }
}