# One command per verb. Everything here runs without a GPU except `bench`.

PY ?= python3
IMAGE ?= ddp-train-harness

.PHONY: setup test smoke build docker-smoke bench clean

setup:            ## install CPU torch for local work (GPU boxes: pip install -r requirements.lock)
	$(PY) -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

test:             ## unit tests, single process, no launcher
	$(PY) -m unittest discover -s tests -v

smoke:            ## 2-process DDP over gloo, the distributed path minus NCCL
	$(PY) -m torch.distributed.run --nproc_per_node 2 src/train.py \
		--backend gloo --epochs 3 --samples 512

build:            ## reproducible image, pinned base + pinned deps
	docker build -t $(IMAGE) .

docker-smoke:     ## the gloo smoke test inside the container
	docker run --rm $(IMAGE)

bench:            ## NCCL scaling run; needs NVIDIA GPUs, fills the README table
	$(PY) -m torch.distributed.run --nproc_per_node $${GPUS:-4} src/train.py \
		--backend nccl --epochs 20 --samples 65536 --dim 256 \
		--result-file runs/bench-$$(date +%s).json

clean:
	rm -rf runs checkpoints __pycache__ src/__pycache__ tests/__pycache__
