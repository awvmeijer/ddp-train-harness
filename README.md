# ddp-train-harness

> A minimal, reproducible harness for distributed PyTorch training: one script that runs identically under CI, torchrun, Docker, and multi-node Slurm.

[![CI](https://github.com/awvmeijer/ddp-train-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/awvmeijer/ddp-train-harness/actions/workflows/ci.yml)
[![Portfolio](https://img.shields.io/badge/portfolio-anthonymeijer.dev-262626?style=flat-square)](https://anthonymeijerdev.vercel.app)

## What it does

The model is a deliberately tiny regression net. The repository is about the
harness around it: process-group setup that works the same from a laptop to a
multi-node Slurm allocation, deterministic seeding, preemption-safe
checkpointing, and CI that tests the distributed path honestly on hardware
that has no GPU.

```bash
make setup   # CPU torch, pinned to the same version the lock file pins
make test    # unit tests: determinism, checkpoint round-trip, loss decreases
make smoke   # real 2-process DDP run over gloo, via torchrun
```

The same entry point covers every mode; only the backend and launcher change:

| Where | Launcher | Backend |
| --- | --- | --- |
| CI, laptop | none or `torchrun` | `none` / `gloo` |
| Docker | `torchrun` (image default) | `gloo`, `nccl` with `--gpus all` |
| Cluster | `sbatch slurm/submit.sbatch` | `nccl` |

## Design decisions

- **Checkpoints are written atomically**: rank 0 writes to a temp file and
  renames, so a preemption mid-write cannot leave a truncated checkpoint that
  poisons the resume. Everyone else waits at a barrier.
- **Seeds are split by role.** The model seed is identical on every rank so
  initial weights agree without a broadcast; the data seed is offset by rank so
  ranks do not train on identical samples. `make test` asserts both.
- **The base image is slim Ubuntu, not `nvidia/cuda`.** torch's Linux wheels
  bundle their own CUDA runtime (the pinned `nvidia-*-cu13` wheels), so a CUDA
  base image would ship a second, mismatched runtime next to them. Only the
  host driver matters, and it reaches the container through
  nvidia-container-toolkit.
- **Everything is pinned.** The base image by digest, every Python dependency
  by exact version in [`requirements.lock`](requirements.lock). The same build
  next year is the same image.
- **Synthetic data, generated in-process.** No downloads in CI, no data files
  in the repo, and a known noise floor, which makes "did the distributed run
  actually train?" a checkable assertion rather than a vibe.

## What CI does and does not prove

GitHub-hosted runners have no GPU. CI therefore runs the unit tests, a real
two-process `torchrun` smoke test on the **gloo** backend, and the Docker
build with the same smoke test inside the container. That exercises the
launcher, rendezvous, `DistributedSampler`, gradient all-reduce, and
checkpointing, everything in the DDP path **except NCCL itself**.

The NCCL path runs on cluster hardware through
[`slurm/submit.sbatch`](slurm/submit.sbatch): one launcher per node via
`srun`, c10d rendezvous on the first node, one worker per GPU,
`NCCL_ASYNC_ERROR_HANDLING=1` so failures surface as stack traces instead of
hangs.

## Benchmarks

No numbers are published here yet, deliberately: this machine has no NVIDIA
GPU, and invented scaling figures are worse than none. `make bench` produces
one JSON result per run; the table fills in from real cluster runs.

| GPUs | Nodes | Backend | Samples/s | Scaling efficiency |
| --- | --- | --- | --- | --- |
| | | | *pending hardware* | |

## Stack

`Python` · `PyTorch` · `DDP` · `NCCL / gloo` · `Docker` · `Slurm` · `GitHub Actions`

## About

Built by [Anthony Meijer](https://anthonymeijerdev.vercel.app). Companion to my
work on real-time ML systems, where the same reproducibility discipline is what
makes a detection pipeline safe to change.
