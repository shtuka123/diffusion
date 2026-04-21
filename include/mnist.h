#pragma once
#include "tensor.h"
#include "cuda_utils.h"
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <random>
#include <algorithm>

// Read a big-endian uint32 from a binary stream.
inline uint32_t read_be32(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16)
         | ((uint32_t)b[2] << 8)  |  (uint32_t)b[3];
}

// MNIST dataset: holds all images flattened into a single float vector on host,
// and all labels as ints. Batches are sampled into GPU tensors on demand.
class MNIST {
public:
    std::vector<float> images;    // flattened (N * 784), normalized to [0, 1]
    std::vector<int> labels;      // (N,) integer labels 0-9
    int n_samples = 0;
    int image_dim = 28 * 28;      // 784

    MNIST() = default;

    MNIST(const std::string& images_path, const std::string& labels_path) {
        load(images_path, labels_path);
    }

    void load(const std::string& images_path, const std::string& labels_path) {
        // --- Images ---
        std::ifstream fi(images_path, std::ios::binary);
        if (!fi) {
            std::fprintf(stderr, "MNIST: cannot open %s\n", images_path.c_str());
            std::abort();
        }
        uint32_t magic = read_be32(fi);
        if (magic != 0x00000803) {
            std::fprintf(stderr, "MNIST: bad magic 0x%x for images file\n", magic);
            std::abort();
        }
        uint32_t n = read_be32(fi);
        uint32_t rows = read_be32(fi);
        uint32_t cols = read_be32(fi);
        if (rows != 28 || cols != 28) {
            std::fprintf(stderr, "MNIST: expected 28x28 images, got %ux%u\n",
                         rows, cols);
            std::abort();
        }

        std::vector<uint8_t> raw_pixels((size_t)n * rows * cols);
        fi.read(reinterpret_cast<char*>(raw_pixels.data()), raw_pixels.size());
        if (!fi) {
            std::fprintf(stderr, "MNIST: truncated images file\n");
            std::abort();
        }

        images.resize(raw_pixels.size());
        for (size_t i = 0; i < raw_pixels.size(); ++i) {
            images[i] = (float)raw_pixels[i] / 255.0f;
        }

        // --- Labels ---
        std::ifstream fl(labels_path, std::ios::binary);
        if (!fl) {
            std::fprintf(stderr, "MNIST: cannot open %s\n", labels_path.c_str());
            std::abort();
        }
        magic = read_be32(fl);
        if (magic != 0x00000801) {
            std::fprintf(stderr, "MNIST: bad magic 0x%x for labels file\n", magic);
            std::abort();
        }
        uint32_t n_lab = read_be32(fl);
        if (n_lab != n) {
            std::fprintf(stderr, "MNIST: image count %u != label count %u\n",
                         n, n_lab);
            std::abort();
        }

        std::vector<uint8_t> raw_labels(n_lab);
        fl.read(reinterpret_cast<char*>(raw_labels.data()), raw_labels.size());

        labels.resize(raw_labels.size());
        for (size_t i = 0; i < raw_labels.size(); ++i) {
            labels[i] = (int)raw_labels[i];
        }

        n_samples = (int)n;
    }

    // Upload a contiguous range of samples to GPU tensors.
    // x_out should be (B, 784); labels_out should be a device int* of length B.
    void get_batch(int start, int batch_size,
                   Tensor& x_out, int* d_labels_out) const
    {
        if (start + batch_size > n_samples) {
            std::fprintf(stderr, "MNIST::get_batch: out of range (%d+%d > %d)\n",
                         start, batch_size, n_samples);
            std::abort();
        }
        if (x_out.numel != (size_t)batch_size * image_dim) {
            std::fprintf(stderr, "MNIST::get_batch: x_out size mismatch\n");
            std::abort();
        }

        // Images are already flat in memory; just upload the right slice.
        CUDA_CHECK(cudaMemcpy(x_out.data,
                              images.data() + (size_t)start * image_dim,
                              (size_t)batch_size * image_dim * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_labels_out,
                              labels.data() + start,
                              batch_size * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    // Random batch: sample B indices (with replacement, for simplicity).
    void get_random_batch(int batch_size,
                          Tensor& x_out, int* d_labels_out,
                          std::mt19937& rng) const
    {
        if (x_out.numel != (size_t)batch_size * image_dim) {
            std::fprintf(stderr, "MNIST::get_random_batch: x_out size mismatch\n");
            std::abort();
        }

        // Gather into host buffers, then one HtoD copy each.
        std::vector<float> batch_images((size_t)batch_size * image_dim);
        std::vector<int> batch_labels(batch_size);

        std::uniform_int_distribution<int> dist(0, n_samples - 1);
        for (int b = 0; b < batch_size; ++b) {
            int idx = dist(rng);
            const float* src = images.data() + (size_t)idx * image_dim;
            std::copy(src, src + image_dim,
                      batch_images.data() + (size_t)b * image_dim);
            batch_labels[b] = labels[idx];
        }

        CUDA_CHECK(cudaMemcpy(x_out.data, batch_images.data(),
                              batch_images.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_labels_out, batch_labels.data(),
                              batch_size * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
};