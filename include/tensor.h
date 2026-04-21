#pragma once
#include <vector>
#include <string>
#include <random>
#include <cstdio>
#include <cmath>
#include "cuda_utils.h"

class Tensor {
public:
    float* data = nullptr;
    std::vector<int> shape; 
    size_t numel = 0;

    Tensor() = default;

    explicit Tensor(std::vector<int> shape_) : shape(std::move(shape_)) {
        numel = 1; 
        for (int s : shape) numel *= (size_t)s;
        if (numel > 0) {
            CUDA_CHECK(cudaMalloc(&data, numel * sizeof(float)));
        }
    }

    Tensor(const Tensor&) = delete;
    Tensor &operator=(const Tensor&) = delete;

    // Move constructor
    Tensor(Tensor&& other) noexcept : data(other.data), shape(std::move(other.shape)), numel(other.numel) {
        other.data = nullptr; 
        other.numel = 0; 
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (data) cudaFree(data);
            data = other.data;
            shape = std::move(other.shape); 
            numel  = other.numel;
            other.data = nullptr;
            other.numel  = 0;
        }
        return *this;
    }

    // Destructor
    ~Tensor() {
        if (data) cudaFree(data);
    }

    // Host <-> Device
    std::vector<float> to_host() const {
        std::vector<float> h(numel);
        if (numel > 0) {
            CUDA_CHECK(cudaMemcpy(h.data(), data, numel * sizeof(float), cudaMemcpyDeviceToHost));
        }
        return h;   
    }

    void from_host(const std::vector<float>& h) {
        if (h.size() != numel) {
            std::fprintf(stderr, "Tensor::from_host size mismatch: got %zu, expected %zu\n", h.size(), numel);
            std::abort();
        }
        if (numel > 0) {
            CUDA_CHECK(cudaMemcpy(data, h.data(), numel * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    static Tensor zeros(std::vector<int> shape) {
        Tensor t(std::move(shape));
        if (t.numel > 0) {
            CUDA_CHECK(cudaMemset(t.data, 0, t.numel * sizeof(float)));
        }
        return t;
    }

    static Tensor randn(std::vector<int> shape, uint64_t seed = 42) {
        Tensor t(std::move(shape)); 
        std::mt19937_64 rng(seed); 
        std::normal_distribution<float> dist(0.f, 1.f);
        std::vector<float> h(t.numel);
        for (size_t i = 0; i < t.numel; ++i) h[i] = dist(rng);
        t.from_host(h);
        return t;
    }

    int ndim() const { return (int)shape.size(); }
    int size(int dim) const { return shape.at(dim); }

    std::string shape_str() const {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) s += ", ";
            s += std::to_string(shape[i]);
        }
        s += "]";
        return s;
    }

    void print(int max_elements = 10) const {
        auto h = to_host();
        std::printf("Tensor %s, numel=%zu: [", shape_str().c_str(), numel);
        size_t n = std::min((size_t)max_elements, numel);
        for (size_t i = 0; i < n; ++i) {
            std::printf("%.4f%s", h[i], (i + 1 < n) ? ", " : "");
        }
        if (numel > n) std::printf(", ...");
        std::printf("]\n");
    }

    // Device-to-device copy. Must be same shape (numel check).
    void copy_from(const Tensor& other) {
        if (other.numel != numel) {
            std::fprintf(stderr, "Tensor::copy_from size mismatch: %zu vs %zu\n",
                        other.numel, numel);
            std::abort();
        }
        if (numel > 0) {
            CUDA_CHECK(cudaMemcpy(data, other.data, numel * sizeof(float),
                                cudaMemcpyDeviceToDevice));
        }
    }



};