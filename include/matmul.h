#pragma once
#include "tensor.h"
#include <vector>

// All three implementations of C = A @ B.
void matmul(Tensor& C, const Tensor& A, const Tensor& B);          // naive CUDA (Day 4)
void matmul_cublas(Tensor& C, const Tensor& A, const Tensor& B);   // cuBLAS

// CPU reference operates on host vectors directly.
void matmul_cpu(std::vector<float>& C,
                const std::vector<float>& A,
                const std::vector<float>& B,
                int M, int K, int N);