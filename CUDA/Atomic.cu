#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


__global__ void vectSumRace(int* d_vect, size_t size, int* result) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	while (tid < size) {
		*result += d_vect[tid];
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void vectSumAtomic(int* d_vect, size_t size, int* result) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	while (tid < size) {
        atomicAdd(result, d_vect[tid]);
		tid += blockDim.x * gridDim.x;
	}
}



int main()
{
    int vect[1000] = { 0 };
    int result=0, result2=0;

    for (int i = 0; i < 1000; i++) {
        vect[i] = i;
    }


    int* dev_vect = 0;
    int* dev_res = 0;
    int* dev_res2 = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_vect, 1000 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_res, 1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_res2, 1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_vect, vect, 1000 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_res, &result, 1 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_res2, &result2, 1 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    vectSumRace <<<1, 1000 >>> (dev_vect, 1000, dev_res);
    vectSumAtomic << <1, 1000 >> > (dev_vect, 1000, dev_res2);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(&result, dev_res, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    std::cout << "Sum-race: "<<result << "\n\n";


    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(&result2, dev_res2, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

   
    std::cout << "Sum-atomic: "<<result2;
    

Error:
    cudaFree(dev_res);
    cudaFree(dev_vect);
 



    return 0;
}