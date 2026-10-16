
def my_matmul(A, B):
    m, n = A.shape
    n2, p = B.shape
    if n != n2:
        raise ValueError("Incompatible matrix shapes")

    C = np.empty((m, p), dtype=object)

    for i in range(m):
        for j in range(p):
            C[i, j] = hy.sum([
                A[i, k] * B[k, j]
                for k in range(n)
            ])

    return C

"""Benchmark symbolic Heyoka matrix multiplication."""

import numpy as np
import heyoka as hy
import time

# Same matrix dimensions as the earlier benchmark
N = 240

# Repeat enough times to obtain useful profiling samples
REPEATS = 5

rng = np.random.default_rng(42)

# Create symbolic variables
x, y = hy.make_vars("x", "y")

# Random coefficients
a = rng.random((N, N))
b = rng.random((N, N))

print("Constructing symbolic matrices...", flush=True)

A = np.array(
    [[float(value) * x for value in row] for row in a],
    dtype=object
)

B = np.array(
    [[float(value) * y for value in row] for row in b],
    dtype=object
)

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print("Starting multiplication benchmark...", flush=True)

times = []

for i in range(REPEATS):
    start = time.perf_counter()

    C = A @ B
    #C = my_matmul(A, B)

    elapsed = time.perf_counter() - start
    times.append(elapsed)

    print(
        f"Run {i + 1}: {elapsed:.3f} seconds",
        flush=True
    )

print(f"\nAverage: {np.mean(times):.3f} seconds")
print(f"Result shape: {C.shape}")
print(f"Result type: {type(C[0, 0])}")
