import torch

if __name__ == "__main__":
    tensor = torch.rand(2, 8192, 32768, dtype=torch.bfloat16).pin_memory()
    memory_size_in_bytes = tensor.element_size() * tensor.numel() / 1024 / 1024
    print(f"tensor size: {memory_size_in_bytes} MB")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(0, torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")

        start_event.record()
        for _ in range(5):
            tensor.to(device)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 5
        throughput = memory_size_in_bytes / (elapsed_time / 1000) / 1024

        print(f"Elapsed time of memcpy from CPU to GPU{i}: {elapsed_time:.4f} ms")
        print(f"Throughput to GPU{i}: {throughput:.4f} GB/s")
