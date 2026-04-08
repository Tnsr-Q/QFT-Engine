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
