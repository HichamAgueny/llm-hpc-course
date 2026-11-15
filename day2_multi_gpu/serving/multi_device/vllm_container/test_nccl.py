
import os
import torch
import torch.distributed as dist


def run_nccl_test():

    # Select GPU before NCCL initialization
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Initialize NCCL
    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size < 2:
            if rank == 0:
                print("World size 1 - nothing to measure.")
            return

        # 100 MiB tensor
        tensor = torch.zeros(
            25 * 1024 * 1024,
            dtype=torch.float32,
            device=device
        )

        # Warmup
        warmup_iterations = 10

        for _ in range(warmup_iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize(device)
        dist.barrier()
        torch.cuda.synchronize(device)

        # Benchmark
        iterations = 100

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        end.record()
        torch.cuda.synchronize(device)

        # Average elapsed time per iteration (seconds)
        elapsed = start.elapsed_time(end) / iterations / 1000.0

        # Aggregate maximum elapsed time across ranks
        elapsed_tensor = torch.tensor(
            [elapsed],
            dtype=torch.float64,
            device=device
        )

        dist.all_reduce(
            elapsed_tensor,
            op=dist.ReduceOp.MAX
        )

        torch.cuda.synchronize(device)
        elapsed = elapsed_tensor.item()

        # Calculate bandwidth
        tensor_bytes = tensor.numel() * tensor.element_size()

        algbw = tensor_bytes / elapsed / 1e9

        busbw = algbw * 2 * (world_size - 1) / world_size

        if rank == 0:
            print("\n==========================================")
            print("         NCCL AllReduce Benchmark")
            print("==========================================")
            print(f"World size:          {world_size}")
            print(f"Tensor size:         {tensor_bytes / 1e6:.2f} MB")
            print(f"Iterations:          {iterations}")
            print(f"Average latency:     {elapsed * 1000:.3f} ms")
            print(f"Algorithm bandwidth: {algbw:.2f} GB/s")
            print(f"Bus bandwidth:       {busbw:.2f} GB/s")
            print("==========================================")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run_nccl_test()

