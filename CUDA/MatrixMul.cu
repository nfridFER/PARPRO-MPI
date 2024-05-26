
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define M 1024
#define BLOCKSIZE 32

__global__ void matrix_mul(int* A, int* B, int* C, int N) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int sum = 0;
    for (int i = 0; i < N; i++) {
        double tempA = A[gy * N + i];
        double tempB = B[i * N + gx];
        sum += tempA * tempB;
    }
    C[gy * N + gx] = sum;
}

__global__ void matrix_mul2(int* A, int* B, int* C, int N)
{
    int lx = threadIdx.x;    // stupac unutar bloka
    int ly = threadIdx.y;    // redak unutar bloka
    int gx = blockIdx.x;     // koordinate bloka
    int gy = blockIdx.y;
    int n = gridDim.x;       // broj grupa (blokova) u jednom retku/stupcu


    // lokalna memorija za pohranjivanje blokova
    __shared__ int tA[BLOCKSIZE][BLOCKSIZE];
    __shared__ int tB[BLOCKSIZE][BLOCKSIZE];

    // posmak za pocetak bloka u izvornim matricama
    int iSubA = BLOCKSIZE * gy * N;
    int iSubB = BLOCKSIZE * gx;

    int sum = 0;
    for (int i = 0; i < n; i++) { // za sve blokove u jednom retku/stupcu
        //kopiraju se blokovi matrica iz globalne memorije u lokalnu memoriju
        tA[ly][lx] = A[ly * N + lx + (iSubA + i * BLOCKSIZE)];
        tB[ly][lx] = B[ly * N + lx + (iSubB + i * BLOCKSIZE * N)];

        // sinkroniziraju se sve dretve u grupi
        __syncthreads();

        // mnozenje dva bloka
        for (int k = 0; k < BLOCKSIZE; k++)
            sum += tA[ly][k] * tB[k][lx];        
    }

    // pohrana u globalnu mem
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdy = blockIdx.y * blockDim.y + threadIdx.y;
    C[globalIdy * N + globalIdx] = sum;
}


int main() {
    int n = M;
    int* a, * b, * c;
    int* d_a, * d_b, * d_c;
    int size = n * n * sizeof(int);

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    srand((unsigned)time(NULL));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            a[i * n + j] = i * j;//rand() % 10;
            b[i * n + j] = i + j;//rand() % 10;
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

    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE); 
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

    //ISPIS MATRICA
    /*
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            std::cout << a[i * n + j] << " ";
        std::cout << "\n";
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            std::cout << b[i * n + j] << " ";
        std::cout << "\n";
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            std::cout<<c[i * n + j]<<" ";            
        std::cout<<"\n";
    }
    */  

    std::cout << "VRIJEME: " << milliseconds<<"\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
