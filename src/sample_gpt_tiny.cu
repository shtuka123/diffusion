// Load a trained GPT-tiny, sample N characters of text from it.

#include "tensor.h"
#include "gpt_tiny.h"
#include "shakespeare.h"
#include "cuda_utils.h"

#include <cstdio>
#include <vector>
#include <random>
#include <string>

int main(int argc, char** argv) {
    // Defaults
    std::string prompt = "ROMEO:";
    int n_generate = 300;
    float temperature = 0.8f;
    int top_k = 40;
    uint64_t seed = 1234;
    std::string ckpt_path = "checkpoints/gpt_tiny.bin";

    // Crude arg parsing
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string flag = argv[i];
        std::string val = argv[i + 1];
        if (flag == "--prompt") prompt = val;
        else if (flag == "--n") n_generate = std::stoi(val);
        else if (flag == "--temp") temperature = std::stof(val);
        else if (flag == "--top-k") top_k = std::stoi(val);
        else if (flag == "--seed") seed = std::stoull(val);
        else if (flag == "--ckpt") ckpt_path = val;
    }

    // Build model with same config as training
    GPTTinyConfig cfg;
    cfg.vocab_size  = 65;
    cfg.max_seq_len = 64;
    cfg.d_model     = 128;
    cfg.n_layers    = 4;
    cfg.n_heads     = 4;
    cfg.d_mlp       = 512;

    int B = 1;
    int T = cfg.max_seq_len;

    GPTTiny model;
    model.build(cfg, B, T, 0);
    model.load(ckpt_path);

    // Load tokenizer (same chars as training)
    Shakespeare tok("data/tinyshakespeare.txt");
    if (tok.vocab_size != cfg.vocab_size) {
        std::fprintf(stderr, "vocab mismatch\n"); return 1;
    }

    // Tokenize the prompt
    std::vector<int> ctx;
    for (char c : prompt) {
        auto it = tok.char_to_id.find(c);
        if (it == tok.char_to_id.end()) {
            std::fprintf(stderr, "Prompt has char not in vocab: '%c'\n", c);
            return 1;
        }
        ctx.push_back(it->second);
    }
    if ((int)ctx.size() >= T) {
        // Truncate left
        ctx.erase(ctx.begin(), ctx.end() - (T - 1));
    }

    // Print the prompt
    std::printf("%s", prompt.c_str());
    std::fflush(stdout);

    // Setup
    int* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, B * T * sizeof(int)));
    std::mt19937 rng(seed);

    // Generate
    for (int step = 0; step < n_generate; ++step) {
        // Build the input context: pad with zeros if shorter than T,
        // or truncate left if longer.
        int ctx_len = (int)ctx.size();
        int pad = std::max(0, T - ctx_len);
        std::vector<int> input(T, 0);
        if (ctx_len >= T) {
            std::copy(ctx.end() - T, ctx.end(), input.begin());
        } else {
            std::copy(ctx.begin(), ctx.end(), input.begin() + pad);
        }

        // The "last meaningful position" — we want to sample from there
        int last_pos = T - 1;  // since we right-aligned the context

        CUDA_CHECK(cudaMemcpy(d_x, input.data(), T * sizeof(int),
                              cudaMemcpyHostToDevice));

        model.forward(d_x, B, T);

        int next = sample_token(
            model.logits, B, T, cfg.vocab_size,
            0, last_pos,
            temperature, top_k,
            rng);

        ctx.push_back(next);
        std::putchar(tok.id_to_char[next]);
        std::fflush(stdout);
    }

    std::printf("\n");
    cudaFree(d_x);
    return 0;
}