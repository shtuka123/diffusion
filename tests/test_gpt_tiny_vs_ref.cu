// End-to-end validation: load same weights as PyTorch, run forward+backward,
// compare every output to PyTorch's saved tensors.

#include "gpt_tiny.h"
#include "cross_entropy.h"
#include "tensor.h"
#include "tensor_io.h"
#include "cuda_utils.h"

#include <cstdio>
#include <vector>
#include <fstream>
#include <string>

static std::vector<int> load_int_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "can't open %s\n", path.c_str()); std::abort(); }
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<int> out(bytes / sizeof(int));
    f.read(reinterpret_cast<char*>(out.data()), bytes);
    return out;
}

static int check(const char* name, const Tensor& got, const std::string& ref_path,
                 float tol)
{
    Tensor expected = load_tensor(ref_path);
    float err = max_abs_diff(got, expected);
    bool pass = err < tol;
    std::printf("  %-30s err=%.3e  %s\n", name, err, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main() {
    std::printf("=== GPT-tiny end-to-end vs PyTorch ===\n");

    // Same config as PyTorch
    GPTTinyConfig cfg;
    cfg.vocab_size  = 65;
    cfg.max_seq_len = 64;
    cfg.d_model     = 128;
    cfg.n_layers    = 4;
    cfg.n_heads     = 4;
    cfg.d_mlp       = 512;

    int B = 2;
    int T = 16;

    GPTTiny model;
    model.build(cfg, B, T, 0);
    model.load("checkpoints/gpt_tiny.bin");

    // Load tokens from PyTorch-saved binary
    auto h_tokens = load_int_bin("data/ref/gpt_validate/tokens.bin");
    auto h_labels = load_int_bin("data/ref/gpt_validate/labels.bin");
    if ((int)h_tokens.size() != B * T) {
        std::fprintf(stderr, "tokens.bin has wrong size\n"); return 1;
    }

    int* d_tokens = nullptr;
    int* d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tokens, B * T * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels, B * T * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), B * T * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), B * T * sizeof(int),
                          cudaMemcpyHostToDevice));

    // ----- Forward -----
    model.forward(d_tokens, B, T);

    int fails = 0;
    fails += check("logits", model.logits, "data/ref/gpt_validate/logits.bin", 1e-3f);

    // ----- Loss -----
    Tensor per_row({B * T});
    Tensor loss({1});
    model.logits.shape = {B * T, cfg.vocab_size};
    cross_entropy_forward(loss, per_row, model.logits, d_labels, B * T, cfg.vocab_size);
    fails += check("loss", loss, "data/ref/gpt_validate/loss.bin", 1e-4f);

    // ----- Backward -----
    Tensor dlogits({B * T, cfg.vocab_size});
    cross_entropy_backward(dlogits, model.logits, d_labels, B * T, cfg.vocab_size);
    model.logits.shape = {B, T, cfg.vocab_size};
    dlogits.shape = {B, T, cfg.vocab_size};
    model.backward(dlogits, d_tokens, B, T);

    // Compare every parameter's gradient
    fails += check("dE",       model.E.grad,       "data/ref/gpt_validate/grad_E.bin",       1e-3f);
    fails += check("dP",       model.P.grad,       "data/ref/gpt_validate/grad_P.bin",       1e-3f);
    for (int l = 0; l < cfg.n_layers; ++l) {
        const auto& bp = model.blocks[l];
        std::string p = "data/ref/gpt_validate/grad_b" + std::to_string(l) + "_";
        fails += check(("dgamma1_b" + std::to_string(l)).c_str(),  bp.gamma1.grad, p + "gamma1.bin",  1e-3f);
        fails += check(("dbeta1_b"  + std::to_string(l)).c_str(),  bp.beta1.grad,  p + "beta1.bin",   1e-3f);
        fails += check(("dW_Q_b"    + std::to_string(l)).c_str(),  bp.W_Q.grad,    p + "W_Q.bin",     1e-3f);
        fails += check(("db_Q_b"    + std::to_string(l)).c_str(),  bp.b_Q.grad,    p + "b_Q.bin",     1e-3f);
        fails += check(("dW_K_b"    + std::to_string(l)).c_str(),  bp.W_K.grad,    p + "W_K.bin",     1e-3f);
        fails += check(("db_K_b"    + std::to_string(l)).c_str(),  bp.b_K.grad,    p + "b_K.bin",     1e-3f);
        fails += check(("dW_V_b"    + std::to_string(l)).c_str(),  bp.W_V.grad,    p + "W_V.bin",     1e-3f);
        fails += check(("db_V_b"    + std::to_string(l)).c_str(),  bp.b_V.grad,    p + "b_V.bin",     1e-3f);
        fails += check(("dW_O_b"    + std::to_string(l)).c_str(),  bp.W_O.grad,    p + "W_O.bin",     1e-3f);
        fails += check(("db_O_b"    + std::to_string(l)).c_str(),  bp.b_O.grad,    p + "b_O.bin",     1e-3f);
        fails += check(("dgamma2_b" + std::to_string(l)).c_str(),  bp.gamma2.grad, p + "gamma2.bin",  1e-3f);
        fails += check(("dbeta2_b"  + std::to_string(l)).c_str(),  bp.beta2.grad,  p + "beta2.bin",   1e-3f);
        fails += check(("dW1_mlp_b" + std::to_string(l)).c_str(),  bp.W1_mlp.grad, p + "W1_mlp.bin",  1e-3f);
        fails += check(("db1_mlp_b" + std::to_string(l)).c_str(),  bp.b1_mlp.grad, p + "b1_mlp.bin",  1e-3f);
        fails += check(("dW2_mlp_b" + std::to_string(l)).c_str(),  bp.W2_mlp.grad, p + "W2_mlp.bin",  1e-3f);
        fails += check(("db2_mlp_b" + std::to_string(l)).c_str(),  bp.b2_mlp.grad, p + "b2_mlp.bin",  1e-3f);
    }
    fails += check("dgamma_f", model.gamma_f.grad, "data/ref/gpt_validate/grad_gamma_f.bin", 1e-3f);
    fails += check("dbeta_f",  model.beta_f.grad,  "data/ref/gpt_validate/grad_beta_f.bin",  1e-3f);
    fails += check("dW_head",  model.W_head.grad,  "data/ref/gpt_validate/grad_W_head.bin",  1e-3f);
    fails += check("db_head",  model.b_head.grad,  "data/ref/gpt_validate/grad_b_head.bin",  1e-3f);

    cudaFree(d_tokens);
    cudaFree(d_labels);

    if (fails == 0) {
        std::printf("\nAll GPT-tiny end-to-end checks passed.\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n%d failed.\n", fails);
        return 1;
    }
}