#pragma once
#include "tensor.h"
#include <fstream>
#include <cstdint>
#include <cstdio>

constexpr uint32_t TENSOR_MAGIC = 0x544E5352;  // 'TNSR'

inline Tensor load_tensor(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "load_tensor: cannot open %s\n", path.c_str());
        std::abort();
    }

    uint32_t magic = 0, ndim = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&ndim), 4);
    if (magic != TENSOR_MAGIC) {
        std::fprintf(stderr, "load_tensor: bad magic in %s: 0x%x\n",
                     path.c_str(), magic);
        std::abort();
    }

    std::vector<int> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t s = 0;
        f.read(reinterpret_cast<char*>(&s), 4);
        shape[i] = (int)s;
    }

    Tensor t(shape);
    std::vector<float> host(t.numel);
    f.read(reinterpret_cast<char*>(host.data()),
           t.numel * sizeof(float));
    t.from_host(host);
    return t;
}

inline void save_tensor(const std::string& path, const Tensor& t) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "save_tensor: cannot open %s\n", path.c_str());
        std::abort();
    }
    uint32_t magic = TENSOR_MAGIC;
    uint32_t ndim = (uint32_t)t.ndim();
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&ndim), 4);
    for (int s : t.shape) {
        uint32_t u = (uint32_t)s;
        f.write(reinterpret_cast<const char*>(&u), 4);
    }
    auto host = t.to_host();
    f.write(reinterpret_cast<const char*>(host.data()),
            t.numel * sizeof(float));
}

// Compare two tensors elementwise. Returns max absolute error.
inline float max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.numel != b.numel) {
        std::fprintf(stderr, "max_abs_diff: size mismatch %zu vs %zu\n",
                     a.numel, b.numel);
        std::abort();
    }
    auto ha = a.to_host();
    auto hb = b.to_host();
    float m = 0.f;
    for (size_t i = 0; i < ha.size(); ++i) {
        float d = std::fabs(ha[i] - hb[i]);
        if (d > m) m = d;
    }
    return m;
}