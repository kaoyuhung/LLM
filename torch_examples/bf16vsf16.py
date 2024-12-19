import torch
import time
from torch import nn


# Set device to GPU
device = torch.device("cuda")


# Function to test performance using torch.cuda.Event
def benchmark(dtype):
    # Create a simple model (e.g., a small MLP)
    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
    ).to(device, dtype=dtype)

    # Create random input tensor
    input_tensor = torch.randn(256, 1024).to(device, dtype=dtype)

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Start the timer
    start_event.record()

    # Perform the forward pass 100 times
    for _ in range(100):
        output = model(input_tensor)

    # End the timer
    end_event.record()

    # Wait for the events to finish and calculate the elapsed time
    torch.cuda.synchronize()  # Wait for all CUDA operations to finish
    elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds

    return elapsed_time


# Test with float16
if (
    torch.cuda.get_device_capability(device)[0] >= 5.3
):  # Check for float16 support (Volta or newer)
    print("Testing with torch.float16:")
    float16_time = benchmark(torch.float16)
    print(f"Average time per iteration (float16): {float16_time:.6f} ms")
else:
    print("This GPU doesn't support torch.float16.")

# Test with bfloat16 (only available on certain GPUs, e.g., A100)
if (
    torch.cuda.get_device_capability(device)[0] >= 8.0
):  # Check for bfloat16 support (A100, etc.)
    print("\nTesting with torch.bfloat16:")
    bfloat16_time = benchmark(torch.bfloat16)
    print(f"Average time per iteration (bfloat16): {bfloat16_time:.6f} ms")
else:
    print("This GPU doesn't support torch.bfloat16.")
