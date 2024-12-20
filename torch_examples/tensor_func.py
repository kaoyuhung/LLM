import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=1000)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create a tensor and transpose it (this results in a non-contiguous tensor)
x = torch.randn(args.n, args.n).to(device)

# Transpose, which results in a non-contiguous tensor but shares the same memory address
y = x.T
p = x.permute(1, 0)

print(x.stride(), x.is_contiguous(), id(x))
print(y.stride(), y.is_contiguous(), id(y))
print(p.stride(), p.is_contiguous(), id(p))

# Use .contiguous() to make the tensor contiguous
if torch.cuda.is_available():
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    z = y.contiguous()
    end_event.record()

    torch.cuda.synchronize()  # Wait for all CUDA operations to finish
    elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
    print(f"(contiguous()) elapse_time: {elapsed_time}ms")

    start_event.record()
    # You can now safely use .view() or .reshape() on z
    z_reshaped = z.view(-1)  # This will work now
    end_event.record()
    torch.cuda.synchronize()  # Wait for all CUDA operations to finish
    elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
    print(f"(view()) elapse_time: {elapsed_time}ms")

    start_event.record()
    z_reshaped = z.reshape(-1)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
    print(f"(reshape() of contiguous tensor) elapse_time: {elapsed_time}ms")

    start_event.record()
    y_reshape = y.reshape(-1)
    end_event.record()
    torch.cuda.synchronize()  # Wait for all CUDA operations to finish
    elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
    print(f"(reshape() of non-contiguous tensor) elapse_time: {elapsed_time}ms")

    start_event.record()
    z_reshaped = z.flatten()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
    print(
        f"(flatten() of contiguous tensor) elapse_time: {elapsed_time}ms, memid: {id(z)} {id(z_reshaped)}"
    )

    start_event.record()
    y_reshape = y.flatten()
    end_event.record()
    torch.cuda.synchronize()  # Wait for all CUDA operations to finish
    elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
    print(
        f"(flatten() of non-contiguous tensor) elapse_time: {elapsed_time}ms, memid: {id(y)} {id(y_reshape)}"
    )

    start_event.record()
    z_reshaped = z.ravel()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
    print(
        f"(ravel() of contiguous tensor) elapse_time: {elapsed_time}ms, memid: {id(z)} {id(z_reshaped)}"
    )

    start_event.record()
    y_reshape = y.ravel()
    end_event.record()
    torch.cuda.synchronize()  # Wait for all CUDA operations to finish
    elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
    print(
        f"(ravel() of non-contiguous tensor) elapse_time: {elapsed_time}ms, memid: {id(y)} {id(y_reshape)}"
    )
