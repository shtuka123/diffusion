#include "tensor.h"
#include "tensor_io.h"
#include "embedding.h"
#include "cuda_utils.h"
#include <cstdio>
#include <fstream>
#include <vector>

static std::vector<int> load_int_tokens(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "can't open %s\n", path.c_str()); std::abort(); }
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<int> out(bytes / sizeof(int));
    f.read(reinterpret_cast<char*>(out.data()), bytes);
    return out;
}

int main() {
    std::printf("=== Embedding forward + backward vs PyTorch ===\n");

    Tensor E = load_tensor("data/ref/emb_E.bin");
    Tensor P = load_tensor("data/ref/emb_P.bin");
    Tensor expected_y  = load_tensor("data/ref/emb_y.bin");
    Tensor dy          = load_tensor("data/ref/emb_dy.bin");
    Tensor expected_dE = load_tensor("data/ref/emb_dE.bin");
    Tensor expected_dP = load_tensor("data/ref/emb_dP.bin");

    auto h_tokens = load_int_tokens("data/ref/emb_tokens.bin");

    int V = E.size(0), D = E.size(1);
    int T_max = P.size(0);
    int B = 2, T = 8;
    if ((int)h_tokens.size() != B * T) {
        std::fprintf(stderr, "tokens count mismatch (got %zu, expected %d)\n",
                     h_tokens.size(), B * T);
        return 1;
    }

    // Upload tokens to device
    int* d_tokens = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tokens, B * T * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), B * T * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Forward: y = E[tokens] + P[:T]
    Tensor y({B, T, D});
    embedding_forward(y, d_tokens, E, B, T);
    positional_add_forward(y, P, B, T);

    float err_y = max_abs_diff(y, expected_y);
    std::printf("  forward y max abs err: %.3e  %s\n",
                err_y, err_y < 1e-5f ? "PASS" : "FAIL");
    int fails = (err_y >= 1e-5f) ? 1 : 0;

    // Backward
    Tensor dE({V, D});
    embedding_backward(dE, dy, d_tokens, B, T);

    Tensor dP({T_max, D});
    positional_add_backward(dP, dy, B, T);

    float err_dE = max_abs_diff(dE, expected_dE);
    float err_dP = max_abs_diff(dP, expected_dP);
    std::printf("  dE max abs err: %.3e  %s\n",
                err_dE, err_dE < 1e-5f ? "PASS" : "FAIL");
    std::printf("  dP max abs err: %.3e  %s\n",
                err_dP, err_dP < 1e-5f ? "PASS" : "FAIL");
    if (err_dE >= 1e-5f) ++fails;
    if (err_dP >= 1e-5f) ++fails;

    cudaFree(d_tokens);

    if (fails == 0) {
        std::printf("\nAll embedding tests passed.\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n%d failed.\n", fails);
        return 1;
    }
}