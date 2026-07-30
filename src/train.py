"""Distributed training harness: small model, real DDP mechanics.

The model is deliberately tiny. The point of this repository is the harness
around it: process-group setup that works identically under torchrun, Slurm,
and single-process CI; deterministic seeding; checkpointing that survives
preemption; and a clean separation between "what backend am I on" and "what am
I training".

Run modes, all through the same entry point:

    # single process, CPU, no process group  (CI, laptops)
    python src/train.py --backend none

    # single node, N processes, gloo (CPU) or nccl (GPU)
    torchrun --nproc_per_node 4 src/train.py --backend gloo
    torchrun --nproc_per_node 4 src/train.py --backend nccl

    # multi-node under Slurm: see slurm/submit.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# --------------------------------------------------------------------------
# Distributed context
# --------------------------------------------------------------------------


class DistContext:
    """Everything the training loop needs to know about the topology.

    Reads the environment torchrun / srun set up. Falls back to a
    single-process context so the same script runs in CI without a launcher.
    """

    def __init__(self, backend: str):
        self.backend = backend
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.distributed = backend != "none" and self.world_size > 1

        if self.distributed:
            dist.init_process_group(backend=backend)

        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def close(self) -> None:
        if self.distributed:
            dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    """Deterministic per-rank seeding.

    Same seed on every rank for the model (so initial weights agree without a
    broadcast), seed offset by rank for data (so ranks do not train on
    identical noise).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Synthetic task
# --------------------------------------------------------------------------


class SyntheticRegression(Dataset):
    """y = Wx + noise, generated on the fly from a per-rank seed.

    Synthetic on purpose: no dataset download in CI, no data files in the
    repo, and loss should fall toward the noise floor on any hardware, which
    makes "did the distributed run actually train?" checkable.
    """

    def __init__(self, n: int, dim: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, dim, generator=g)
        w = torch.arange(1, dim + 1, dtype=torch.float32) / dim
        self.y = self.x @ w.unsqueeze(1) + 0.01 * torch.randn(n, 1, generator=g)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def build_model(dim: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------


def save_checkpoint(path: Path, model, optimizer, epoch: int, ctx: DistContext) -> None:
    """Rank 0 writes; everyone waits. Written atomically via a temp file so a
    preemption mid-write cannot leave a truncated checkpoint behind."""
    if ctx.is_main:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": (model.module if isinstance(model, DDP) else model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.rename(path)
    ctx.barrier()


def load_checkpoint(path: Path, model, optimizer, ctx: DistContext) -> int:
    """Every rank loads the same file. Returns the epoch to resume from."""
    if not path.exists():
        return 0
    payload = torch.load(path, map_location=ctx.device, weights_only=True)
    (model.module if isinstance(model, DDP) else model).load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if ctx.is_main:
        print(f"resumed from {path} at epoch {payload['epoch']}")
    return payload["epoch"]


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------


def train(args: argparse.Namespace) -> dict:
    ctx = DistContext(args.backend)
    seed_everything(args.seed, ctx.rank)

    dataset = SyntheticRegression(args.samples, args.dim, seed=args.seed + ctx.rank)
    sampler = (
        DistributedSampler(dataset, num_replicas=ctx.world_size, rank=ctx.rank)
        if ctx.distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=True,
    )

    model = build_model(args.dim).to(ctx.device)
    if ctx.distributed:
        model = DDP(
            model,
            device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    ckpt = Path(args.checkpoint_dir) / "latest.pt"
    start_epoch = load_checkpoint(ckpt, model, optimizer, ctx) if args.resume else 0

    history = []
    t0 = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)  # reshuffle across ranks each epoch
        model.train()
        total, batches = 0.0, 0
        for x, y in loader:
            x, y = x.to(ctx.device), y.to(ctx.device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()  # DDP all-reduces gradients here
            optimizer.step()
            total += loss.item()
            batches += 1

        epoch_loss = total / max(batches, 1)
        if ctx.distributed:
            # average the reported loss across ranks so rank 0 logs a global number
            t = torch.tensor([epoch_loss], device=ctx.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            epoch_loss = t.item()

        history.append(epoch_loss)
        if ctx.is_main:
            print(f"epoch {epoch:03d}  loss {epoch_loss:.6f}")
        if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(ckpt, model, optimizer, epoch + 1, ctx)

    wall = time.perf_counter() - t0
    result = {
        "backend": args.backend,
        "world_size": ctx.world_size,
        "device": str(ctx.device),
        "first_loss": history[0] if history else None,
        "final_loss": history[-1] if history else None,
        "epochs": args.epochs - start_epoch,
        "wall_seconds": round(wall, 2),
    }
    if ctx.is_main:
        print(json.dumps(result))
        if args.result_file:
            Path(args.result_file).write_text(json.dumps(result))
    ctx.close()
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["none", "gloo", "nccl"], default="none")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=0, help="epochs; 0 disables")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--result-file", default="", help="write final metrics JSON here")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
