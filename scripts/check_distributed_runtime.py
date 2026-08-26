#!/usr/bin/env python3
"""Run a tiny torchrun/NCCL collective and DDP backward diagnostic.

This script does not read the precipitation dataset. GPU memory usage is only
that of a two-layer toy model, so it can safely verify selected idle cards
before launching a long training job.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default=None)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_cuda = torch.cuda.is_available()
    backend = args.backend or ("nccl" if use_cuda else "gloo")
    if backend == "nccl" and not use_cuda:
        raise RuntimeError("NCCL was selected but CUDA is unavailable")

    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    distributed.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(seconds=args.timeout_seconds),
        device_id=device if device.type == "cuda" else None,
    )
    try:
        # Each rank starts with rank+1; SUM must become 1+...+world_size.
        collective = torch.tensor(float(rank + 1), device=device)
        distributed.all_reduce(collective)
        expected = world_size * (world_size + 1) / 2
        if float(collective.item()) != expected:
            raise RuntimeError(
                f"all_reduce returned {collective.item()}, expected {expected}"
            )

        # Exercise parameter broadcast and gradient all-reduce, not just NCCL
        # process-group creation. Tensor shape: (B=2,F=4) -> (B=2,F=1).
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        ).to(device)
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            static_graph=True,
        )
        inputs = torch.full((2, 4), float(rank + 1), device=device)
        loss = ddp_model(inputs).square().mean()
        loss.backward()
        gradient_checksum = sum(
            float(parameter.grad.float().sum().item())
            for parameter in ddp_model.parameters()
            if parameter.grad is not None
        )
        checksum = torch.tensor(gradient_checksum, dtype=torch.float64, device=device)
        gathered = [torch.zeros_like(checksum) for _ in range(world_size)]
        distributed.all_gather(gathered, checksum)
        if not all(torch.equal(gathered[0], value) for value in gathered[1:]):
            raise RuntimeError("DDP gradients differ between ranks")

        local_report = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": socket.gethostname(),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "all_reduce_sum": float(collective.item()),
            "gradient_checksum": gradient_checksum,
        }
        reports: list[dict[str, object] | None] = [None] * world_size
        distributed.all_gather_object(reports, local_report)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "backend": backend,
                        "world_size": world_size,
                        "master_addr": os.environ.get("MASTER_ADDR"),
                        "master_port": os.environ.get("MASTER_PORT"),
                        "ranks": reports,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
