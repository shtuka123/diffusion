#include "tensor.h"
#include "tensor_io.h"
#include "cross_entropy.h"
#include "cuda_utils.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <fstream>
#include <vector>

// Read a raw int32 binary file into a vector.
static std::vector<int> load_int_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "can't open %s\n", path.c_str()); std::abort(); }
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<int> labels(bytes / sizeof(int));
    f.read(reinterpret_cast<char*>(labels.data()), bytes);
    return labels;
}

int test_case(const std::string& name,
              const std::string& logits_path,
              const std::string& labels_path,
              const std::string& loss_path,
              float tol = 1e-5f)
{
    std::printf("=== %s ===\n", name.c_str());
    Tensor logits = load_tensor(logits_path);
    Tensor expected_loss = load_tensor(loss_path);
    auto h_labels = load_int_labels(labels_path);

    int B = logits.size(0);
    int C = logits.size(1);
    if ((int)h_labels.size() != B) {
        std::fprintf(stderr, "label count (%zu) != B (%d)\n", h_labels.size(), B);
        return 1;
    }

    // Upload labels to GPU
    int* d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_labels, B * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), B * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Allocate scratch + scalar
    Tensor per_row({B});
    Tensor loss({1});

    cross_entropy_forward(loss, per_row, logits, d_labels, B, C);

    // Compare
    auto h_loss = loss.to_host();
    auto h_expected = expected_loss.to_host();
    float err = std::fabs(h_loss[0] - h_expected[0]);
    std::printf("  computed: %.6f, expected: %.6f, abs err: %.3e\n",
                h_loss[0], h_expected[0], err);

    // Sanity: check per-row losses for NaN/inf
    auto h_row = per_row.to_host();
    for (int b = 0; b < B; ++b) {
        if (std::isnan(h_row[b]) || std::isinf(h_row[b])) {
            std::fprintf(stderr, "  FAIL: row %d has NaN/inf\n", b);
            cudaFree(d_labels);
            return 1;
        }
        if (h_row[b] < 0.f) {
            std::fprintf(stderr, "  FAIL: row %d loss is negative (%.3f) — likely bad label\n",
                         b, h_row[b]);
            cudaFree(d_labels);
            return 1;
        }
    }

    cudaFree(d_labels);

    if (err < tol) {
        std::printf("  PASS\n");
        return 0;
    } else {
        std::fprintf(stderr, "  FAIL (tol=%.0e)\n", tol);
        return 1;
    }
}

int main() {
    int fails = 0;
    fails += test_case("tiny (hand-verified)",
                       "data/ref/ce_tiny_logits.bin",
                       "data/ref/ce_tiny_labels.bin",
                       "data/ref/ce_tiny_loss.bin");
    fails += test_case("batch (B=128, C=10)",
                       "data/ref/ce_batch_logits.bin",
                       "data/ref/ce_batch_labels.bin",
                       "data/ref/ce_batch_loss.bin");
    fails += test_case("large logits (stability test)",
                       "data/ref/ce_large_logits.bin",
                       "data/ref/ce_large_labels.bin",
                       "data/ref/ce_large_loss.bin");

    if (fails == 0) {
        std::printf("\nAll CE tests passed.\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n%d test(s) failed.\n", fails);
        return 1;
    }
}