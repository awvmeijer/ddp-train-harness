# Slim base on purpose. torch's Linux wheels bundle their own CUDA runtime
# (the nvidia-*-cu13 wheels pinned in requirements.lock), so a nvidia/cuda base
# image would ship a second, mismatched CUDA runtime next to them. Only the
# host driver matters at run time, and that reaches the container through
# nvidia-container-toolkit (`docker run --gpus all ...`).
#
# Pinned by digest, not just tag: the same build next year is the same image.
FROM ubuntu:22.04@sha256:0e0a0fc6d18feda9db1590da249ac93e8d5abfea8f4c3c0c849ce512b5ef8982

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first, so code edits do not re-download torch.
COPY requirements.lock .
RUN python3.11 -m pip install -r requirements.lock

COPY src/ src/
COPY tests/ tests/

# Non-root: clusters will not run your container as root, so CI should not either.
RUN useradd --create-home trainer
USER trainer

# Default command proves the distributed path without any GPU present.
# Real GPU runs override it:
#   docker run --gpus all <image> python3.11 -m torch.distributed.run \
#     --nproc_per_node 4 src/train.py --backend nccl
CMD ["python3.11", "-m", "torch.distributed.run", "--nproc_per_node", "2", \
     "src/train.py", "--backend", "gloo", "--epochs", "3"]
