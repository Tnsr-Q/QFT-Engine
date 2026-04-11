<div align="center">

# QFT-Engine

### High-performance verification framework for computational QFT workflows

<p>
  QFT-Engine unifies symbolic validation, numerical solvers, distributed execution,
  topology-aware orchestration, precision telemetry, and structured scientific outputs
  into one research-grade verification stack.
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/JAX-XLA%20Accelerated-FC6D26?style=for-the-badge" alt="JAX">
  <img src="https://img.shields.io/badge/PyTorch-Lightning-red?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch Lightning">
  <img src="https://img.shields.io/badge/DeepSpeed-Distributed-5B21B6?style=for-the-badge" alt="DeepSpeed">
  <img src="https://img.shields.io/badge/pytest-Verified-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="pytest">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
</p>

<p>
  <strong>Symbolic checks</strong> •
  <strong>RGE solvers</strong> •
  <strong>Spectral analysis</strong> •
  <strong>Regge tracking</strong> •
  <strong>Hessian telemetry</strong>
</p>

</div>

---

<p align="center">
  <img src="./assets/qft-engine-data-flow.png" alt="QFT-Engine data flow infographic" width="100%">
</p>

<p align="center">
  <em>End-to-end flow across validation, solver execution, topology coordination, logging, and structured outputs.</em>
</p>

---

## Particle Map

```mermaid
graph TD
    Q((QFT-Engine))

    S1((RGE))
    S2((Flow))
    S3((Bootstrap))
    S4((Regge))
    S5((Hessian))
    S6((Schemas))
    S7((Mesh))
    S8((Tolerance))
    S9((Tests))
    S10((Deploy))

    Q --- S1
    Q --- S2
    Q --- S3
    Q --- S4
    Q --- S5
    Q --- S6
    Q --- S7
    Q --- S8
    Q --- S9
    Q --- S10

    S1 --- A1((rge_solver.py))
    S2 --- A2((flow_solver.py))
    S2 --- A3((spectral_flow.py))
    S3 --- A4((bootstrap_solver.py))
    S4 --- A5((regge_jax))
    S4 --- A6((vmap/pmap/shard_map))
    S5 --- A7((hessian_jax))
    S5 --- A8((quantized Hessian))
    S6 --- A9((proto/))
    S7 --- A10((mesh/))
    S8 --- A11((tolerance/))
    S9 --- A12((pytest suite))
    S10 --- A13((GCE + Docker))


⸻

Overview

QFT-Engine is a research-grade verification and analysis framework for advanced computational physics workflows.

It combines:
	•	symbolic consistency checks
	•	renormalization group and flow-based solvers
	•	spectral and dispersive analysis
	•	bootstrap and Regge-trajectory workflows
	•	distributed Hessian estimation and precision control
	•	schema-enforced outputs, device-mesh coordination, and adaptive tolerance governance

This repository is best understood as a verification stack, not a generic end-user application. It is designed to run, compare, validate, and audit specialized solver pipelines across multiple runtime backends and execution modes.

⸻

Why QFT-Engine

<div align="center">


Physics-aware	Execution-aware	Validation-aware
Domain-specific solver families for QFT-style workflows	JAX, PyTorch Lightning, DeepSpeed, TensorBoard, Docker	Schema enforcement, tolerance ledgers, structured outputs

</div>


What makes it different
	•	It is built around solver families, not a single algorithm.
	•	It treats execution topology as part of the architecture, not an afterthought.
	•	It treats numerical tolerances and output schemas as governed system components.
	•	It includes a meaningful test and integration surface, not just demos.

⸻

Core Capabilities

Numerical and solver systems
	•	RGE solving for renormalization-flow experiments
	•	Flow-based solvers for spectral and dynamical analysis
	•	Discretized bootstrap routines for constrained amplitude workflows
	•	Regge trajectory solvers across standard, vmap, pmap, and shard_map execution paths
	•	JAX Hessian estimation with quantized variants and distributed support

Validation and consistency tooling
	•	symbolic BRST-style verification
	•	residual and predicate validation
	•	spectral consistency checks
	•	runtime schema enforcement for structured solver outputs

Infrastructure and execution layers
	•	JAX and PyTorch unified topology abstractions
	•	tolerance priors and adaptive ledger tracking
	•	PyTorch Lightning callbacks for Hessian telemetry, ZeRO-3, FP8, and CPU fallback
	•	profiling, TensorBoard, and cloud deployment scripts

⸻

Architecture

Inputs
├── runtime configs
├── tolerance priors
├── commands / scripts
└── initial solver state
        │
        ▼
Validation & preprocessing
        │
        ▼
Core solver layer
├── RGE
├── flow
├── spectral
├── bootstrap
├── Regge
└── Hessian / optimization
        │
        ▼
Governance layer
├── schema enforcement
├── mesh coordination
└── tolerance control
        │
        ▼
Outputs
├── logs and diagnostics
├── checkpoints
├── serialized artifacts
└── test / report outputs


⸻

Repository Structure

QFT-Engine/
├── configs/
│   ├── params.yaml
│   └── tolerance_priors.yaml
├── docker/
│   └── Dockerfile
├── scripts/
│   ├── deploy_gce.sh
│   ├── deploy_profiler_gce.sh
│   ├── deploy_universal.sh
│   ├── diagnose_precision.py
│   ├── launch_tensorboard_proxy.sh
│   └── run_suite.sh
├── src/
│   ├── bootstrap_solver.py
│   ├── brst_checker.py
│   ├── flow_solver.py
│   ├── hessian_jax.py
│   ├── hessian_qjax.py
│   ├── optimizer.py
│   ├── regge_bootstrap.py
│   ├── regge_jax_solver.py
│   ├── regge_pmap_solver.py
│   ├── regge_shard_map.py
│   ├── regge_vmap_solver.py
│   ├── rge_solver.py
│   ├── spectral_density.py
│   ├── spectral_flow.py
│   ├── unified_topology.py
│   ├── validators.py
│   ├── callbacks/
│   ├── discovery/
│   ├── mesh/
│   ├── proto/
│   ├── spectral/
│   ├── tolerance/
│   └── truth/
├── tests/
└── .github/workflows/


⸻

Key Subsystems

<details>
<summary><strong>Solver layer</strong></summary>


The solver surface spans multiple computational styles and execution models:
	•	src/rge_solver.py
	•	src/flow_solver.py
	•	src/spectral_flow.py
	•	src/bootstrap_solver.py
	•	src/regge_bootstrap.py
	•	src/regge_jax_solver.py
	•	src/regge_vmap_solver.py
	•	src/regge_pmap_solver.py
	•	src/regge_shard_map.py
	•	src/hessian_jax.py
	•	src/hessian_qjax.py

This structure supports method comparison, backend scaling, and validation across classical and accelerated numerical workflows.

</details>


<details>
<summary><strong>Schema and serialization layer</strong></summary>


The src/proto/ package provides:
	•	constraint schemas
	•	registries
	•	return-schema definitions
	•	schema enforcement
	•	serializers
	•	atomic checkpoint support

This gives the project a structured contract layer around solver outputs.

</details>


<details>
<summary><strong>Mesh and topology layer</strong></summary>


The src/mesh/ package provides:
	•	topology abstractions
	•	execution schemes
	•	unified mesh coordination

This makes the repository much more execution-aware than a typical research codebase.

</details>


<details>
<summary><strong>Tolerance governance</strong></summary>


The src/tolerance/ package and configs/tolerance_priors.yaml indicate an explicit system for:
	•	tolerance baselines
	•	bounded adaptation
	•	regime detection
	•	residual-aware control

Numerical thresholds are treated as managed runtime state, not hidden constants.

</details>


<details>
<summary><strong>Callback and telemetry layer</strong></summary>


The callback surface includes:
	•	checkpointed Hessian paths
	•	distributed Hessian monitoring
	•	ZeRO-3 variants
	•	FP8 variants
	•	CPU fallback variants
	•	precision control

This gives the repo strong observability and experimentation value for large-scale or precision-sensitive workloads.

</details>



⸻

Installation

Prerequisites
	•	Python 3.10+
	•	pip
	•	optional GPU or multi-device environment for advanced execution
	•	Docker for containerized runs

Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

Optional package used by some workflows

pip install sympy


⸻

Quick Start

Run the full verification suite

bash scripts/run_suite.sh

Freeze adaptive tolerances for deterministic replay

bash scripts/run_suite.sh --freeze --audit-verify

Run tests directly

pytest tests/ -v

Build and run with Docker

docker build -t qft-engine -f docker/Dockerfile .
docker run --rm qft-engine


⸻

Example Workflows

JAX sharded Regge execution

import jax.numpy as jnp
from src.regge_shard_map import ShardedReggeSolver

solver = ShardedReggeSolver(N_t=256)
delta = jnp.zeros(256) + 0.05

trajectory = solver.scan_regge_trajectory_sharded(delta)
certificate = solver.verify_fakeon_virtualization(trajectory)

print(certificate["status"])

Precision diagnostics

python scripts/diagnose_precision.py

TensorBoard helper

bash scripts/launch_tensorboard_proxy.sh

Cloud-oriented execution

export BUCKET="your-verify-bucket"
bash scripts/deploy_gce.sh


⸻

Configuration

configs/params.yaml

Contains the main runtime controls for:
	•	roadmap constants
	•	solver tolerances
	•	iteration limits
	•	checkpoint interval
	•	precision target
	•	high-level assumptions

configs/tolerance_priors.yaml

Defines tolerance policies for subsystems such as:
	•	rge_atol
	•	hessian_pl
	•	bootstrap_unitarity
	•	regge_pole

Together these files form the numerical control plane for the engine.

⸻

Testing Strategy

The test suite covers far more than a smoke test.

Coverage areas
	•	regression behavior
	•	flow fixed-point checks
	•	bootstrap and JAX integration
	•	spectral representation and robustness
	•	nonperturbative unitarity checks
	•	Regge distributed execution
	•	tolerance ledger validation
	•	memory fallback paths
	•	profiler and compression integration
	•	GCE and multi-device integration

Representative tests
	•	test_bootstrap_jax.py
	•	test_flow_fixed_point.py
	•	test_nonperturbative_unitarity.py
	•	test_regge_pl_integration.py
	•	test_shardmap_zero3_integration.py
	•	test_tolerance_ledger.py
	•	test_robust_spectral.py

One of the repo’s strongest qualities is that the architecture is backed by a substantial verification surface.

⸻

Operational Tooling

Local and CI execution
	•	scripts/run_suite.sh

Precision and runtime inspection
	•	scripts/diagnose_precision.py

Profiling and visualization
	•	scripts/launch_tensorboard_proxy.sh
	•	scripts/deploy_profiler_gce.sh

Cloud deployment
	•	scripts/deploy_gce.sh
	•	scripts/deploy_universal.sh

GitHub Actions
	•	.github/workflows/quft-verify.yml

⸻

Technology Stack

<div align="center">


Area	Tools
Numerical computing	NumPy, SciPy, JAX
ML / distributed	PyTorch, PyTorch Lightning, DeepSpeed
Validation	Pydantic
Storage / serialization	PyYAML, PyArrow
Testing	pytest
Observability	TensorBoard
Packaging / runtime	Docker

</div>



⸻

Design Philosophy

Structured computation

Solver output is not treated as an afterthought. The repository includes schema, registry, serializer, and checkpoint layers to keep computational results traceable and structured.

Execution-aware research code

The presence of vmap, pmap, shard_map, callback variants, mesh abstractions, and deployment scripts shows that scalability and runtime behavior are first-class concerns.

Verification over hype

The repository leans into tests, tolerances, validations, and explicit infrastructure around residuals and execution modes.

Modular extension

Subsystems are separated cleanly enough that contributors can extend:
	•	solver implementations
	•	validation layers
	•	topology backends
	•	tolerance policies
	•	callback instrumentation

⸻

Ideal Use Cases

This repository is a strong fit for people who want to:
	•	prototype or extend computational QFT verification workflows
	•	experiment with JAX-native and distributed solver implementations
	•	validate numerical routines with reproducible tests and tolerances
	•	build infrastructure around schema-validated scientific computation
	•	explore precision-sensitive training or Hessian-monitoring workflows

⸻

Contributing
	1.	Install dependencies.
	2.	Run the existing test suite.
	3.	Keep changes scoped to a subsystem.
	4.	Update configs, tests, and docs with behavior changes.
	5.	Preserve or improve validation and reproducibility pathways.

⸻

License

Add this once the repo includes a LICENSE file.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.


⸻


<div align="center">


QFT-Engine

Research computation, validated execution, and reproducible verification.

</div>
```


Two upgrades would make this even better:

First, replace the Mermaid particle map with a custom SVG in assets/particle-map.svg for a much more cinematic connected-node look.

Second, add a real CI badge from .github/workflows/quft-verify.yml once the repo is public.