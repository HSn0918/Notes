"""Run with torchrun on allocated GPUs (NCCL) or CPUs (Gloo)."""

import argparse
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

INPUTS = [[2., 1.], [-1., 2.], [1., -2.], [-2., 1.]]
TARGETS = [0, 1, 0, 1]
RATE = 0.1
TOL = 1e-6


def check_step(device, rank, size):
    if size not in (1, 2):
        raise ValueError("use one or two ranks")
    x = torch.tensor(INPUTS, device=device)
    y = torch.tensor(TARGETS, device=device)
    model = nn.Linear(2, 2, bias=False).to(device)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.1, 0.05], [-0.2, 0.3]], device=device))
    reference = nn.Linear(2, 2, bias=False).to(device)
    reference.load_state_dict(model.state_dict())
    wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    local = slice(None) if size == 1 else (slice(0, 1) if rank == 0 else slice(1, None))
    count = torch.tensor(y[local].numel(), device=device)
    dist.all_reduce(count)
    (criterion(reference(x), y) / len(y)).backward()
    (criterion(wrapped(x[local]), y[local]) * size / count).backward()
    for net in (wrapped, reference):
        torch.optim.SGD(net.parameters(), lr=RATE).step()
    error = (model.weight - reference.weight).detach().abs().max()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"ranks={size} tokens={count.item()} max_weight_error={error.item():.3e}")
    if not error.item() < TOL:
        raise AssertionError("distributed update differs from full batch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        check_step(device, dist.get_rank(), dist.get_world_size())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
