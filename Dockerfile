# Slim Python base on purpose, twice over.
#
# Not nvidia/cuda: torch's Linux wheels bundle their own CUDA runtime (the
# nvidia-*-cu13 wheels pinned in requirements.lock), so a CUDA base image
# would ship a second, mismatched runtime next to them. Only the host driver
# matters at run time, via nvidia-container-toolkit (`docker run --gpus all`).
#
# Not ubuntu + apt python3.11: Ubuntu 22.04's python3.11 package is
# 3.11.0~rc1, a pre-release that lacks sys.get_int_max_str_digits and crashes
# torch's sympy import. CI caught that. The official python image is an
# actual 3.11 release.
#
# Pinned by digest, not just tag: the same build next year is the same image.
FROM python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency layer first, so code edits do not re-download torch.
COPY requirements.lock .
RUN python -m pip install -r requirements.lock

COPY src/ src/
COPY tests/ tests/

# Non-root: clusters will not run your container as root, so CI should not either.
RUN useradd --create-home trainer
USER trainer

# Default command proves the distributed path without any GPU present.
# Real GPU runs override it:
#   docker run --gpus all <image> python -m torch.distributed.run \
#     --nproc_per_node 4 src/train.py --backend nccl
CMD ["python", "-m", "torch.distributed.run", "--nproc_per_node", "2", \
     "src/train.py", "--backend", "gloo", "--epochs", "3"]
