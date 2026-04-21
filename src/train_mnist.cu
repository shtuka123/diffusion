// Train a (784 -> 256 -> 10) MLP on MNIST using SGD + cross-entropy.
// No new kernels — this file is pure orchestration.
//
// Day 18: get loss dropping on real data. Accuracy measurement comes Day 19.

#include "tensor.h"
#include "parameter.h"
#include "linear.h"
#include "elementwise.h"
#include "cross_entropy.h"
#include "optim.h"
#include "init.h"
#include "mnist.h"
#include "cuda_utils.h"

#include <cstdio>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

// Evaluate on the test set. Returns accuracy in [0, 1].
// Forward pass only — no gradients, no optimizer state.
float evaluate(const MNIST& test,
               const Parameter& W1, const Parameter& b1,
               const Parameter& W2, const Parameter& b2,
               int B)
{
    int D_in = test.image_dim;
    int H = W1.w.size(1);
    int D_out = W2.w.size(1);
    int N = test.n_samples;

    // Preallocate once. Same Tensors we'd use in training.
    Tensor x({B, D_in});
    Tensor h({B, H});
    Tensor h_relu({B, H});
    Tensor logits({B, D_out});

    int* d_labels = nullptr;
    int* d_preds = nullptr;
    CUDA_CHECK(cudaMalloc(&d_labels, B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_preds,  B * sizeof(int)));

    int n_full_batches = N / B;
    int correct = 0;
    int total = 0;

    for (int step = 0; step < n_full_batches; ++step) {
        // Sequential batch (no shuffling — we iterate the test set in order).
        test.get_batch(step * B, B, x, d_labels);

        // Forward
        linear_forward(h, x, W1.w, b1.w);
        relu_forward(h_relu, h);
        linear_forward(logits, h_relu, W2.w, b2.w);

        // Argmax + compare on host (tiny data, easier than device-side reduce)
        argmax_row(d_preds, logits);

        std::vector<int> h_preds(B), h_labels(B);
        CUDA_CHECK(cudaMemcpy(h_preds.data(),  d_preds,  B * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels, B * sizeof(int),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < B; ++i) {
            if (h_preds[i] == h_labels[i]) ++correct;
        }
        total += B;
    }

    // Handle any leftover samples at the tail (N not divisible by B).
    // For MNIST test (N=10000, B=128): 10000 = 78*128 + 16 leftover.
    int tail = N - n_full_batches * B;
    if (tail > 0) {
        // Reuse the same buffers — we just don't care about positions past 'tail'.
        // Grab a batch that includes the tail plus some already-evaluated samples.
        int start = N - B;  // last B samples
        test.get_batch(start, B, x, d_labels);
        linear_forward(h, x, W1.w, b1.w);
        relu_forward(h_relu, h);
        linear_forward(logits, h_relu, W2.w, b2.w);
        argmax_row(d_preds, logits);

        std::vector<int> h_preds(B), h_labels(B);
        CUDA_CHECK(cudaMemcpy(h_preds.data(),  d_preds,  B * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels, B * sizeof(int),
                              cudaMemcpyDeviceToHost));
        // Only count the tail samples (last 'tail' of this batch)
        for (int i = B - tail; i < B; ++i) {
            if (h_preds[i] == h_labels[i]) ++correct;
        }
        total += tail;
    }

    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_preds));

    return (float)correct / (float)total;
}


int main() {
    // ---------- Config ----------
    const int D_in = 784;
    const int H = 256;
    const int D_out = 10;
    const int B = 128;
    const int epochs = 10;
    const float lr = 0.1f;
    const uint64_t seed = 42;

    // ---------- Load data ----------
    std::printf("Loading MNIST...\n");
    MNIST train("data/mnist/train-images-idx3-ubyte",
                "data/mnist/train-labels-idx1-ubyte");
    MNIST test("data/mnist/t10k-images-idx3-ubyte",
               "data/mnist/t10k-labels-idx1-ubyte");
    std::printf("  train: %d samples, test: %d samples\n\n",
                train.n_samples, test.n_samples);

    // ---------- Parameters ----------
    std::printf("Initializing model (%d -> %d -> %d)...\n", D_in, H, D_out);
    Parameter W1({D_in, H});    init_he_normal(W1, D_in, seed + 1);
    Parameter b1({H});          init_zero(b1);
    Parameter W2({H, D_out});   init_he_normal(W2, H,    seed + 2);
    Parameter b2({D_out});      init_zero(b2);

    std::vector<Parameter*> params = {&W1, &b1, &W2, &b2};
    int n_params = 0;
    for (auto* p : params) n_params += (int)p->w.numel;
    std::printf("  parameters: %d (%.2f MB of weights)\n\n",
                n_params, n_params * 4.0f / (1024 * 1024));

    // ---------- Preallocate everything ----------
    // Inputs (per batch)
    Tensor x({B, D_in});
    int* d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_labels, B * sizeof(int)));

    // Forward intermediates
    Tensor h({B, H});            // pre-ReLU
    Tensor h_relu({B, H});       // post-ReLU
    Tensor logits({B, D_out});   // network output
    Tensor per_row({B});
    Tensor loss({1});

    // Backward intermediates — these get written by each backward kernel
    // and then copied into the Parameter's grad field.
    Tensor dlogits({B, D_out});
    Tensor dh_post_relu({B, H});
    Tensor dh_pre_relu({B, H});
    Tensor dx({B, D_in});        // we don't actually need dx (input has no grad)
                                 // but linear_backward produces it anyway

    Tensor dW1({D_in, H}), db1({H});
    Tensor dW2({H, D_out}), db2({D_out});

    // ---------- Index shuffling for epochs ----------
    std::vector<int> indices(train.n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);

    // Loss tracking
    const int log_every = 100;
    float running_loss = 0.f;
    int running_count = 0;

    // ---------- Training loop ----------
    std::printf("Training for %d epochs, batch=%d, lr=%.3f\n", epochs, B, lr);
    std::printf("----------------------------------------\n");

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        int steps_per_epoch = train.n_samples / B;
        for (int step = 0; step < steps_per_epoch; ++step) {

            // ----- Assemble the batch from shuffled indices -----
            // We could call get_random_batch, but that samples with replacement.
            // For a proper epoch, we use the shuffled order instead.
            std::vector<float> batch_x(B * D_in);
            std::vector<int> batch_y(B);
            for (int i = 0; i < B; ++i) {
                int idx = indices[step * B + i];
                std::copy(
                    train.images.data() + idx * D_in,
                    train.images.data() + idx * D_in + D_in,
                    batch_x.data() + i * D_in);
                batch_y[i] = train.labels[idx];
            }
            CUDA_CHECK(cudaMemcpy(x.data, batch_x.data(),
                                  batch_x.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_labels, batch_y.data(),
                                  B * sizeof(int),
                                  cudaMemcpyHostToDevice));

            // ----- Forward -----
            linear_forward(h, x, W1.w, b1.w);
            relu_forward(h_relu, h);
            linear_forward(logits, h_relu, W2.w, b2.w);
            cross_entropy_forward(loss, per_row, logits, d_labels, B, D_out);

            // ----- Log loss (copy scalar back) -----
            auto h_loss = loss.to_host();
            running_loss += h_loss[0];
            running_count += 1;

            // ----- Backward -----
            cross_entropy_backward(dlogits, logits, d_labels, B, D_out);

            // Second linear's backward: produces dh_post_relu and the grads for W2, b2
            linear_backward(dh_post_relu, dW2, db2, h_relu, W2.w, dlogits);
            W2.grad.copy_from(dW2);
            b2.grad.copy_from(db2);

            // ReLU's backward: produces dh_pre_relu
            relu_backward(dh_pre_relu, dh_post_relu, h);

            // First linear's backward: produces dx (unused) and grads for W1, b1
            linear_backward(dx, dW1, db1, x, W1.w, dh_pre_relu);
            W1.grad.copy_from(dW1);
            b1.grad.copy_from(db1);

            // ----- Optimizer step -----
            sgd_step_all(params, lr);
            zero_grads(params);

            // ----- Report -----
            if ((step + 1) % log_every == 0) {
                float avg = running_loss / running_count;
                std::printf("  epoch %d  step %4d/%d  loss %.4f\n",
                            epoch + 1, step + 1, steps_per_epoch, avg);
                running_loss = 0.f;
                running_count = 0;
            }
        }
        std::printf("  --- epoch %d complete ---\n", epoch + 1);
        float acc = evaluate(test, W1, b1, W2, b2, B);
        std::printf("  test accuracy: %.4f\n\n", acc);
    }

    std::printf("\nTraining finished.\n");
    CUDA_CHECK(cudaFree(d_labels));
    return 0;
}