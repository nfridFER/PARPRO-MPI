#include <iostream>
#include <cuda_runtime.h>

__global__ void divergent_branch_demo(int* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // divergencija
    if (threadIdx.x % 3 == 0) {
        output[tid] = tid * 1; 
    }
    else if (threadIdx.x % 3 == 1) {
        output[tid] = tid * 2;  
    }
    else {
        output[tid] = tid * 3;  
    }
}


int main() {
    const int numThreads = 32;
    const int numBlocks = 2;
    const int N = numThreads * numBlocks;

    int* h_output = new int[N];
    int* d_output;
    cudaMalloc(&d_output, N * sizeof(int));

    divergent_branch_demo << <numBlocks, numThreads >> > (d_output);
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

  
    for (int i = 0; i < numBlocks*numThreads; ++i)
        std::cout << "output[" << i << "] = " << h_output[i] << "\n";

    delete[] h_output;
    cudaFree(d_output);
    return 0;
}
