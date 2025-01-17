import time
import torch
import triton
import triton.language as tl


def timed(fn, *args, n_warmup=3):
    for _ in range(n_warmup):
        fn(*args)

    t = time.time()
    torch.cuda.synchronize(A.device)
    result = fn(*args)
    torch.cuda.synchronize(A.device)

    return result, (time.time() - t) * 1000


def matmul_basic(A, B, C):
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Initialize the C value to zero
    value = 0.0

    # Perform the matmul operation (sum of element-wise products)
    for k in range(K):
        a = tl.load(A + row * K + k)
        b = tl.load(B + k * N + col)
        value += a * b

    # Store the result in C
    tl.store(C + row * N + col, value)


if __name__ == "__main__":

    device = torch.device("cuda:0")

    M, N, K = 4096, 4096, 4096

    A = torch.rand(M, N, dtype=torch.bfloat16, device=device)
    B = torch.rand(N, K, dtype=torch.bfloat16, device=device)
    C = torch.empty(M, K, dtype=torch.bfloat16, device=device)

    C1, t = timed(torch.matmul, A, B)
    print(f"torch_matmul: {t:.4f} ms")
