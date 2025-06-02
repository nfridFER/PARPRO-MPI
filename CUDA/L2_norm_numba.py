# Euklidska dist. (L2 norm)

import numpy as np
import time
from numba import guvectorize, float32, cuda

@guvectorize([(float32[:], float32[:])], '(n)->()', target='cuda')
def row_l2norm(x, out):
    tmp = 0.0
    for i in range(x.shape[0]):
        tmp += x[i] * x[i]
    out[0] = tmp ** 0.5

N = 4096
COLS = 256
A = np.random.randint(0, 10, size=(N, COLS)).astype(np.float32)
out = np.zeros(N, dtype=np.float32)

# Warm-up
_ = row_l2norm(A)
cuda.synchronize()

# Timing
start = time.time()
out = row_l2norm(A)
cuda.synchronize()
end = time.time()

print(f"L2 norm comp. t: {(end - start)*1000:.2f} ms")
print("Zadnjih 5 L2 normi:", out[-5:])
