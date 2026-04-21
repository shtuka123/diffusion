#include "mnist.h"
#include "tensor.h"
#include "cuda_utils.h"
#include <cstdio>

// Render one 28x28 image as ASCII. Values in [0,1], thresholds chosen so
// handwritten digits are readable.
void ascii_render(const float* img) {
    const char* shades = " .:-=+*#%@";
    for (int r = 0; r < 28; ++r) {
        for (int c = 0; c < 28; ++c) {
            float v = img[r * 28 + c];
            int idx = (int)(v * 9.99f);  // 0-9
            if (idx < 0) idx = 0;
            if (idx > 9) idx = 9;
            std::putchar(shades[idx]);
            std::putchar(shades[idx]);  // 2x wide so digits look square
        }
        std::putchar('\n');
    }
}

int main() {
    std::printf("=== MNIST loader test ===\n");

    MNIST train("data/mnist/train-images-idx3-ubyte",
                "data/mnist/train-labels-idx1-ubyte");
    std::printf("Training set loaded: %d samples\n", train.n_samples);

    MNIST test("data/mnist/t10k-images-idx3-ubyte",
               "data/mnist/t10k-labels-idx1-ubyte");
    std::printf("Test set loaded: %d samples\n\n", test.n_samples);

    // Verify counts
    if (train.n_samples != 60000) {
        std::fprintf(stderr, "FAIL: expected 60000 training samples, got %d\n",
                     train.n_samples);
        return 1;
    }
    if (test.n_samples != 10000) {
        std::fprintf(stderr, "FAIL: expected 10000 test samples, got %d\n",
                     test.n_samples);
        return 1;
    }

    // Show first 3 training samples as ASCII
    std::printf("First 3 training images (label shown above each):\n\n");
    for (int i = 0; i < 3; ++i) {
        std::printf("Label: %d\n", train.labels[i]);
        ascii_render(train.images.data() + i * 784);
        std::printf("\n");
    }

    // Verify a batch upload works
    const int B = 4;
    Tensor x_batch({B, 784});
    int* d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_labels, B * sizeof(int)));

    train.get_batch(0, B, x_batch, d_labels);

    // Round-trip check: download back, compare to host source
    auto h_gpu = x_batch.to_host();
    for (int i = 0; i < B * 784; ++i) {
        if (h_gpu[i] != train.images[i]) {
            std::fprintf(stderr,
                         "FAIL: GPU upload mismatch at index %d (%.4f vs %.4f)\n",
                         i, h_gpu[i], train.images[i]);
            cudaFree(d_labels);
            return 1;
        }
    }

    std::vector<int> h_gpu_labels(B);
    CUDA_CHECK(cudaMemcpy(h_gpu_labels.data(), d_labels, B * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < B; ++i) {
        if (h_gpu_labels[i] != train.labels[i]) {
            std::fprintf(stderr, "FAIL: label mismatch at %d\n", i);
            cudaFree(d_labels);
            return 1;
        }
    }

    std::printf("Batch upload round-trip: PASS\n");

    // Random batch smoke test (just verify it runs)
    std::mt19937 rng(12345);
    train.get_random_batch(B, x_batch, d_labels, rng);
    std::printf("Random batch sampling: PASS\n\n");

    cudaFree(d_labels);
    std::printf("All MNIST loader tests passed.\n");
    return 0;
}