#include <cstdio>
#include <random>
#include <cstdlib>

__global__ void hello_kernel() {
    printf("hello from thread %d of block %d\n",
           threadIdx.x, blockIdx.x);
}

__global__ void vec_add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
    printf("%d + %d = %d\n", a[blockIdx.x], b[blockIdx.x], c[blockIdx.x]);
}

void random_ints(int *a, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = rand() % 100; 
    }
}


int main() {
    int n = 10;
    int *a, *b, *c; 
    int *d_a, *d_b, *d_c;
    int size = n * sizeof(int); 

    cudaMalloc((void **) &d_a, size); 
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    a = (int *)malloc(size); random_ints(a, n);
    b = (int *)malloc(size); random_ints(b, n);
    c = (int *)malloc(size); 

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice); 

    vec_add<<<n, 1>>>(d_a, d_b, d_c); 

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost); 

    free(a); free(b); free(c); 
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);


    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();  // wait for GPU prints before exiting
    return 0;
}