#pragma once
#include "tensor.h"
#include "cuda_utils.h"
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdio>
#include <random>

// Character-level tokenizer + dataset for tiny-shakespeare.
// Loads the entire text into RAM, builds char-to-int mapping, exposes batch sampling.
class Shakespeare {
public:
    std::vector<int> tokens;         // entire text as token IDs
    std::unordered_map<char, int> char_to_id;
    std::vector<char> id_to_char;
    int vocab_size = 0;

    Shakespeare() = default;

    explicit Shakespeare(const std::string& path) {
        load(path);
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "Shakespeare: cannot open %s\n", path.c_str());
            std::abort();
        }
        std::string text((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());

        // Build vocabulary from sorted unique chars
        std::vector<char> uniq(text.begin(), text.end());
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

        id_to_char = uniq;
        for (int i = 0; i < (int)uniq.size(); ++i) {
            char_to_id[uniq[i]] = i;
        }
        vocab_size = (int)uniq.size();

        // Tokenize
        tokens.resize(text.size());
        for (size_t i = 0; i < text.size(); ++i) {
            tokens[i] = char_to_id.at(text[i]);
        }

        std::printf("Shakespeare loaded: %zu chars, vocab size %d\n",
                    tokens.size(), vocab_size);
    }

    // Sample a random batch.
    // For each of B batch items, pick a random start offset, copy T tokens to
    // x_out and the same T tokens shifted by 1 to y_out (the target).
    //
    // x_out, y_out: device int* buffers of length B*T.
    void get_random_batch(int B, int T,
                          int* x_out_device, int* y_out_device,
                          std::mt19937& rng) const
    {
        int n = (int)tokens.size();
        if (n < T + 1) {
            std::fprintf(stderr, "Shakespeare: dataset too small for T=%d\n", T);
            std::abort();
        }
        std::uniform_int_distribution<int> dist(0, n - T - 1);

        std::vector<int> batch_x(B * T);
        std::vector<int> batch_y(B * T);
        for (int b = 0; b < B; ++b) {
            int off = dist(rng);
            for (int t = 0; t < T; ++t) {
                batch_x[b * T + t] = tokens[off + t];
                batch_y[b * T + t] = tokens[off + t + 1];   // next-token target
            }
        }

        CUDA_CHECK(cudaMemcpy(x_out_device, batch_x.data(),
                              B * T * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(y_out_device, batch_y.data(),
                              B * T * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    // Decode a sequence of token IDs back to a string.
    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        out.reserve(ids.size());
        for (int id : ids) {
            if (id >= 0 && id < (int)id_to_char.size()) out.push_back(id_to_char[id]);
        }
        return out;
    }
};