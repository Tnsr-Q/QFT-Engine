# QÜFT Verification Suite

Computational verification scaffold for bounded tests of flow stability, spectral consistency, and bootstrap-inspired non-perturbative constraints.

## Layout

- `src/`: numerical and symbolic helper modules
- `tests/`: pytest verification suite mapped to roadmap claims
- `configs/params.yaml`: central tolerances and assumption bounds
- `scripts/run_suite.sh`: local/CI execution entrypoint (with timeout)
- `scripts/deploy_gce.sh`: preemptible Google Compute Engine launcher

## Local run

```bash
python -m pip install -U pip
pip install numpy scipy sympy pytest pyyaml
bash scripts/run_suite.sh
```

## Docker run

```bash
docker build -t quft-test -f docker/Dockerfile .
docker run --rm quft-test
```

## GCE deployment

```bash
export BUCKET="your-verify-bucket"
bash scripts/deploy_gce.sh
```

The startup routine executes the suite, uploads JUnit XML output to Cloud Storage, and shuts the instance down automatically.

## Distributed extensions

This repository also includes production-oriented distributed components for
large parameter scans and memory-constrained training:

- `src/regge_shard_map.py`: JAX `shard_map`-based Regge trajectory scanning with
  explicit fakeon virtualization certification (`VERIFIED`/`PENDING`).
- `src/callbacks/zero3_hessian_pl.py`: PyTorch Lightning callback for
  ZeRO-3/FSDP-compatible Hessian spectrum monitoring, PL-condition checks, and
  adaptive learning-rate updates.

### JAX shard_map solver quickstart

```python
from src.regge_shard_map import ShardedReggeSolver
import jax.numpy as jnp

solver = ShardedReggeSolver(N_t=256)
delta_mock = jnp.zeros(256) + 0.05
trajectory = solver.scan_regge_trajectory_sharded(delta_mock)
certificate = solver.verify_fakeon_virtualization(trajectory)
print(certificate["status"])
```

### ZeRO-3/FSDP callback quickstart

```python
from pytorch_lightning import Trainer
from src.callbacks.zero3_hessian_pl import Zero3CheckpointedHessianPLCallback

trainer = Trainer(
    strategy="fsdp",  # or "deepspeed_stage_3"
    callbacks=[Zero3CheckpointedHessianPLCallback(monitor_every=25)],
    accelerator="gpu",
    devices=4,
)
```
