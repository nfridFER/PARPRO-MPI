#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define N 4096
#define COLS 256

/* Euklidska dist. (L2 norm)*/
__global__ void l2norm_kernel(float* A, float* out, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            float val = A[idx * cols + i];
            sum += val * val;
        }
        out[idx] = sqrtf(sum);
    }
}

int main() {
    float* h_A = (float*)malloc(N * COLS * sizeof(float));
    float* h_out = (float*)malloc(N * sizeof(float));
    float* d_A, * d_out;

    for (int i = 0; i < N * COLS; i++)
        h_A[i] = rand() % 10;

    cudaMalloc(&d_A, N * COLS * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_A, h_A, N * COLS * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    l2norm_kernel << <gridDim, blockDim >> > (d_A, d_out, COLS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("CUDA L2 norm kernel time: %f ms \n", time_ms);
    printf("Last few results: \n");
    for (int i = N - 5; i < N; i++) {
        printf("%f\\n", h_out[i]);
    }

    cudaFree(d_A);
    cudaFree(d_out);
    free(h_A);
    free(h_out);

    return 0;
}
