
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define N 1024

__global__ void matrix_mul(int* a, int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int sum = 0;
    if (i < n && j < n) {
        for (int k = 0; k < n; k++)
            sum += a[i * n + k] * b[k * n + j];
        c[i * n + j] = sum;
    }
}


int main() {
    int n = N;
    int* a, * b, * c;
    int* d_a, * d_b, * d_c;
    int size = n * n * sizeof(int);

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    srand((unsigned)time(NULL));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            a[i * n + j] = rand() % 10;
            b[i * n + j] = rand() % 10;
        }

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    //for measuring time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(64, 64); 
    std::cout << "threads per block: " << threadsPerBlock.x << ", " << threadsPerBlock.y<<"\n";

    dim3 numBlocks((n+threadsPerBlock.x-1)/threadsPerBlock.x, (n + threadsPerBlock.y - 1)/threadsPerBlock.y); 
    std::cout << "Num blocks: " << numBlocks.x << ", " << numBlocks.y<<"\n";

    cudaEventRecord(start);
    matrix_mul << <numBlocks, threadsPerBlock >> > (d_a, d_b, d_c, n);
    cudaEventRecord(stop);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    /*
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            std::cout<<c[i * n + j]<<" ";            
        std::cout<<"\n";
    }
    */    

    std::cout << "Time: " << milliseconds;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}