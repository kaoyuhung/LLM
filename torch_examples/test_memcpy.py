import os
import time
import torch
import torch.distributed as dist


def test_intra_node():
    tensor = torch.rand(10, 4096, 14336, dtype=torch.bfloat16)
    memory_size_in_bytes = tensor.element_size() * tensor.numel() / 1024 / 1024
    print(f"tensor size: {memory_size_in_bytes} MB")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(0, torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")

        start_event.record()
        for _ in range(5):
            gpu_tensor = tensor.to(device)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 5
        throughput = memory_size_in_bytes / (elapsed_time / 1000) / 1024
        print(
            f"Memcpy from CPU to GPU{i}: Lantency={elapsed_time:.4f} ms, Throughput={throughput:.4f} GB/s"
        )

        for j in range(0, torch.cuda.device_count()):
            if i == j:
                continue
            device = torch.device(f"cuda:{j}")
            start_event.record()
            for _ in range(5):
                gpu_tensor.to(device)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 5
            throughput = memory_size_in_bytes / (elapsed_time / 1000) / 1024
            print(
                f"Memcpy from GPU{i} to GPU{j}: Lantency={elapsed_time:.4f} ms, Throughput={throughput:.4f} GB/s"
            )


def test_inter_node():

    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    WORLD_RANK = int(os.environ["RANK"])

    device = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        backend="nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=device
    )

    tensor = torch.rand(10, 4096, 14336, dtype=torch.bfloat16, device=device)
    memory_size_in_bytes = tensor.element_size() * tensor.numel() / 1024 / 1024
    if LOCAL_RANK == 0:
        print(f"tensor size: {memory_size_in_bytes} MB")

    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            if i == j:
                continue

            if i == WORLD_RANK:
                t = time.time()
                for _ in range(10):
                    dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, j)])[
                        0
                    ].wait()
                torch.cuda.synchronize(device)
                elapsed_time = ((time.time() - t) / 10) * 1000
                throughput = memory_size_in_bytes / (elapsed_time / 1000) / 1024
                print(
                    f"Memcpy from GPU{i} to GPU{j}: Lantency={elapsed_time:.4f} ms, Throughput={throughput:.4f} GB/s"
                )

            if j == WORLD_RANK:
                for _ in range(10):
                    dist.batch_isend_irecv([dist.P2POp(dist.irecv, tensor, i)])[
                        0
                    ].wait()
                torch.cuda.synchronize(device)
            dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    if "WORLD_SIZE" in os.environ:
        test_inter_node()

    else:
        test_intra_node()
