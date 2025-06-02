# Euklidska dist. (L2 norm)

import numpy as np
import time
from numba import cuda, float32

@cuda.jit
def l2norm_kernel(A, out, cols):
    row = cuda.grid(1)
    if row < A.shape[0]:
        tmp = 0.0
        for i in range(cols):
            val = A[row, i]
            tmp += val * val
        out[row] = tmp ** 0.5


rows = 4096
cols = 256

A = np.random.randint(0, 10, size=(rows, cols)).astype(np.float32)
out = np.zeros(rows, dtype=np.float32)

dA = cuda.to_device(A)
dOut = cuda.device_array(rows, dtype=np.float32)

threads_per_block = 256
blocks_per_grid = (rows + threads_per_block - 1) // threads_per_block

# Warm-up
l2norm_kernel[blocks_per_grid, threads_per_block](dA, dOut, cols)
cuda.synchronize()

# Timing
start = time.time()
l2norm_kernel[blocks_per_grid, threads_per_block](dA, dOut, cols)
cuda.synchronize()
end = time.time()

out = dOut.copy_to_host()

print(f"L2 norm comp. t: {(end - start)*1000:.2f} ms")

print("Zadnjih 5 L2 normi:", out[-5:])
