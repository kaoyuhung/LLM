import os
import time
import torch
import torch.distributed as dist


def test_intra_node():
    tensor = torch.rand(16, 4096, 14336, dtype=torch.bfloat16)
    memory_size_in_bytes = tensor.element_size() * tensor.numel()
    print(f"tensor size: {memory_size_in_bytes / 2**20} MB")

    for i in range(0, torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")

        for _ in range(3):  # warm-up
            tensor.to(device)

        t = time.time()
        for _ in range(5):
            gpu_tensor = tensor.to(device)
            torch.cuda.synchronize(device=device)
        elapsed_time = (time.time() - t) / 5
        throughput = (memory_size_in_bytes / 2**30) / elapsed_time
        print(
            f"Memcpy from CPU to GPU{i}: Lantency={elapsed_time:.4f} ms, Throughput={throughput:.4f} GB/s"
        )

        for j in range(0, torch.cuda.device_count()):
            if i == j:
                continue
            device = torch.device(f"cuda:{j}")

            for _ in range(3):
                gpu_tensor.to(device)

            t = time.time()
            for _ in range(5):
                gpu_tensor.to(device)
                torch.cuda.synchronize(device=device)
            elapsed_time = (time.time() - t) / 5
            throughput = (memory_size_in_bytes / 2**30) / elapsed_time
            print(
                f"Memcpy from GPU{i} to GPU{j}: Lantency={elapsed_time:.4f} ms, Throughput={throughput:.4f} GB/s"
            )


def test_inter_node():

    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    WORLD_RANK = int(os.environ["RANK"])

    device = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        backend="nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=device
    )

    tensor = torch.rand(16, 4096, 14336, dtype=torch.bfloat16, device=device)
    memory_size_in_bytes = tensor.element_size() * tensor.numel()
    if LOCAL_RANK == 0:
        print(f"tensor size: {memory_size_in_bytes / 2**20} MB")

    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            if i == j:
                continue

            if i == WORLD_RANK:
                for _ in range(3):
                    dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, j)])[
                        0
                    ].wait()

                torch.cuda.synchronize(device)
                t = time.time()
                for _ in range(5):
                    dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, j)])[
                        0
                    ].wait()
                    torch.cuda.synchronize(device)
                elapsed_time = (time.time() - t) / 5

                throughput = (memory_size_in_bytes / 2**30) / elapsed_time
                print(
                    f"Memcpy from GPU{i} to GPU{j}: Lantency={elapsed_time:.4f} ms, Throughput={throughput:.4f} GB/s"
                )

            if j == WORLD_RANK:
                for _ in range(8):
                    dist.batch_isend_irecv([dist.P2POp(dist.irecv, tensor, i)])[
                        0
                    ].wait()
            dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    if "WORLD_SIZE" in os.environ:
        test_inter_node()

    else:
        test_intra_node()
