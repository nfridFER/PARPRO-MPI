import numpy as np
import time
from numba import cuda, float32

N = 2048
TPB = 16  # Threads per block

@cuda.jit
def matmul_shared(A, B, C):
    # Shared memory tiles
    sA = cuda.shared.array((TPB, TPB), dtype=float32)
    sB = cuda.shared.array((TPB, TPB), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    row = by * TPB + ty
    col = bx * TPB + tx

    tmp = 0.0

    # Loop over tiles
    for m in range((N + TPB - 1) // TPB):
        # Load tile from A and B into shared memory
        if row < N and (m * TPB + tx) < N:
            sA[ty, tx] = A[row, m * TPB + tx]
        else:
            sA[ty, tx] = 0.0

        if col < N and (m * TPB + ty) < N:
            sB[ty, tx] = B[m * TPB + ty, col]
        else:
            sB[ty, tx] = 0.0

        
        cuda.syncthreads()


        for k in range(TPB):
            tmp += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()


    if row < N and col < N:
        C[row, col] = tmp
        
        

# ---- Host ----

A = np.random.randint(0, 10, size=(N, N)).astype(np.float32)
B = np.random.randint(0, 10, size=(N, N)).astype(np.float32)

C = np.zeros((N, N), dtype=np.float32)

dA = cuda.to_device(A)
dB = cuda.to_device(B)
dC = cuda.device_array((N, N), dtype=np.float32)

threads_per_block = (TPB, TPB)
blocks_per_grid_x = (N + TPB - 1) // TPB
blocks_per_grid_y = (N + TPB - 1) // TPB
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Warm-up (compilation overhead)
matmul_shared[blocks_per_grid, threads_per_block](dA, dB, dC)
cuda.synchronize()

# Timing the kernel execution
start = time.time()
matmul_shared[blocks_per_grid, threads_per_block](dA, dB, dC)
cuda.synchronize()
end = time.time()

C = dC.copy_to_host()

print(f"Exec time: {(end - start)*1000:.2f} ms")

" zadnjih nekoliko vrijednosti"
print("Zadnjih nekoliko vrijednosti C...")
print(C[-5:, -5:])  # Last 5 rows and columns
