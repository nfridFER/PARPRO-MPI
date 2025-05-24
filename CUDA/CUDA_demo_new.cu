#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Matrix size: N x N

void printDeviceInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Detected " << deviceCount << " CUDA device(s):\n";

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory: " << (prop.totalGlobalMem >> 20) << " MB\n";
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "  Warp size: " << prop.warpSize << "\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Shared memory per block: " << (prop.sharedMemPerBlock >> 10) << " KB\n";
        std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz\n";
    }
}

__global__ void mat_add(const float* A, const float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * width + col;

    if (row < width && col < width) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    printDeviceInfo();

    size_t size = N * N * sizeof(float);
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = i%1000;
        h_B[i] = 2.0*(i%1000);
    }

    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    mat_add << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, N);
   
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N*N; i+=999) {
        std::cout << "C[" << i << "] = " << h_C[i] << "\n";
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    return 0;
}
