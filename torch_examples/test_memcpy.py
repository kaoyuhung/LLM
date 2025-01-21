import os
import time
import torch
import torch.distributed as dist
import argparse


def test_intra_node(nsys: bool):
    tensor = torch.rand(4, 4096, 14336, dtype=torch.bfloat16)
    memory_size_in_bytes = tensor.element_size() * tensor.numel()
    print(f"tensor size: {memory_size_in_bytes / 2**20} MB")

    if nsys:
        torch.cuda.cudart().cudaProfilerStart()

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
            f"Memcpy from CPU to GPU{i}: Latency={elapsed_time:.4f} s, Throughput={throughput:.4f} GB/s"
        )

        for j in range(0, torch.cuda.device_count()):
            if i == j:
                continue
            deviceTo = torch.device(f"cuda:{j}")

            for _ in range(3):
                gpu_tensor.to(deviceTo)
                torch.cuda.synchronize(device=device)
                torch.cuda.synchronize(device=deviceTo)

            t = time.time()
            for _ in range(5):
                gpu_tensor.to(deviceTo)
                torch.cuda.synchronize(device=device)
                torch.cuda.synchronize(device=deviceTo)
            elapsed_time = (time.time() - t) / 5
            throughput = (memory_size_in_bytes / 2**30) / elapsed_time
            print(
                f"Memcpy from GPU{i} to GPU{j}: Latency={elapsed_time:.4f} s, Throughput={throughput:.4f} GB/s"
            )

    if nsys:
        torch.cuda.cudart().cudaProfilerStop()


def test_inter_node(nsys: bool):

    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    WORLD_RANK = int(os.environ["RANK"])

    device = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        backend="nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=device
    )

    tensor = torch.rand(4, 4096, 14336, dtype=torch.bfloat16, device=device)
    memory_size_in_bytes = tensor.element_size() * tensor.numel()
    if LOCAL_RANK == 0:
        print(f"tensor size: {memory_size_in_bytes / 2**20} MB")

    if nsys:
        torch.cuda.cudart().cudaProfilerStart()

    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            if i == j:
                continue

            if i == WORLD_RANK:
                dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, j)])[
                    0
                ].wait()  # warmup

                torch.cuda.synchronize(device)
                t = time.time()
                dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, j)])[0].wait()
                torch.cuda.synchronize(device)
                elapsed_time = time.time() - t

                throughput = (memory_size_in_bytes / 2**30) / elapsed_time
                print(
                    f"NCCL SendRecv from GPU{i} to GPU{j}: Latency={elapsed_time:.4f} s, Throughput={throughput:.4f} GB/s"
                )

            if j == WORLD_RANK:
                for _ in range(2):
                    dist.batch_isend_irecv([dist.P2POp(dist.irecv, tensor, i)])[
                        0
                    ].wait()
            dist.barrier()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    t = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    elapsed_time = time.time() - t
    throughput = (memory_size_in_bytes / 2**30) / elapsed_time
    if LOCAL_RANK == 0:
        print(
            f"NCCL Allreduce: Latency={elapsed_time:.4f} s, Throughput={throughput:.4f} GB/s"
        )

    tensor_list = [torch.empty_like(tensor) for _ in range(WORLD_SIZE)]
    dist.all_gather(tensor_list, tensor)
    t = time.time()
    dist.all_gather(tensor_list, tensor)
    torch.cuda.synchronize(device)
    elapsed_time = time.time() - t
    throughput = (memory_size_in_bytes / 2**30) / elapsed_time
    if LOCAL_RANK == 0:
        print(
            f"NCCL Allgather: Latency={elapsed_time:.4f} s, Throughput={throughput:.4f} GB/s"
        )

    shape = list(tensor.shape)
    shape[0] *= WORLD_SIZE
    output_tensor = torch.empty(shape, dtype=tensor.dtype, device=device)
    dist.all_gather_into_tensor(output_tensor, tensor)
    t = time.time()
    dist.all_gather_into_tensor(output_tensor, tensor)
    torch.cuda.synchronize(device)
    elapsed_time = time.time() - t
    throughput = (memory_size_in_bytes / 2**30) / elapsed_time
    if LOCAL_RANK == 0:
        print(
            f"NCCL Allgather_into_tensor: Latency={elapsed_time:.4f} s, Throughput={throughput:.4f} GB/s"
        )

    dist.barrier()
    dist.destroy_process_group()

    if nsys:
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys", type=eval, default=False)
    args = parser.parse_args()

    if "WORLD_SIZE" in os.environ:
        test_inter_node(args.nsys)

    else:
        test_intra_node(args.nsys)
