import torch
import triton
import triton.language as tl


@triton.jit
def demo(x_ptr):
    range = tl.arange(0, 8)
    x = tl.load(x_ptr + range, range < 5, 0)
    return x


@triton.jit
def demo2(x_ptr):
    i_range = tl.arange(0, 8)[:, None]
    j_range = tl.arange(0, 4)[None, :]
    range = i_range * 4 + j_range
    # print works in the interpreter
    print(range)
    x = tl.load(x_ptr + range, (i_range < 4) & (j_range < 3), 0)
    print(x)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    print(demo[(12)](torch.ones(12, device=device)))
