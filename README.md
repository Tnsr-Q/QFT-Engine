<!-- VISUAL: Full-width hero banner — dark field / teal grid dissolving into Regge trajectory curves emanating from a central Mandelstam plane. Overlaid: "QFT-ENGINE" in sharp monospace, "QÜFT VERIFICATION SUITE" below in smaller weight. -->

# ⚛ QFT-Engine — QÜFT Verification Suite

> **Computational verification scaffold for Scale-Invariant Quadratic Gravity.**  
> Flow stability · Spectral consistency · Bootstrap non-perturbative constraints · Distributed Hessian telemetry.

<!-- VISUAL: 3-column badge row — CI status | Docker build | Python version -->

---

## ◈ What This Is

QFT-Engine is a precision verification harness for theoretical claims in
**Scale-Invariant Quadratic Gravity (SIQG)** with the **fakeon prescription**
(Anselmi). It is not a simulation framework — it is an **assertion engine**.
Every module is a mathematical predicate: pass or fail, with quantified residuals.

The engine spans:

| Layer | Technology |
|---|---|
| Symbolic | `sympy` BRST nilpotency checks |
| Numerical ODE | `scipy.solve_ivp` 2-loop RGE β-flow |
| JIT / Auto-diff | `jax` (x64, jvp/vjp, jit, vmap, pmap, shard_map) |
| Distributed Training | `pytorch-lightning` + ZeRO-3 / FSDP / DeepSpeed |
| Precision modes | FP64 → FP32 → FP8 (e4m3fn) with error feedback |
| Observability | TensorBoard + `jax.profiler` + GCS upload |
| Compute backends | CPU · GPU · Multi-device · GCE preemptible |

<!-- VISUAL: Layered isometric stack diagram — five tiers: Hardware / JAX runtime / solver modules / callback layer / test harness. Each tier labeled, arrows show data flow. -->

---

## ◈ Repository Map

```
QFT-Engine/
├── src/
│   ├── brst_checker.py            ← Symbolic s²=0 nilpotency verification
│   ├── validators.py              ← BFB / metastability / thermal boundary checks
│   ├── rge_solver.py              ← 2-Loop SIQG+SM RGE integration (SIQGRGESolver)
│   ├── spectral_density.py        ← Källén-Lehmann rep, Froissart-Martin bound
│   ├── spectral_flow.py           ← Lorentzian ERG dispersive flow G_k(p)
│   ├── flow_solver.py             ← Toy Lorentzian Wetterich β-function ODE
│   ├── bootstrap_solver.py        ← S-matrix bootstrap (unitarity + crossing)
│   ├── regge_bootstrap.py         ← Regge pole tracking on bootstrap solution
│   ├── optimizer.py               ← Gauss-Newton Hessian, PL-condition, Lyapunov
│   ├── hessian_jax.py             ← JAX HVP, Lanczos eigenvalue (JAXHessianEstimator)
│   ├── regge_vmap_solver.py       ← Single-device vmap Regge solver
│   ├── regge_pmap_solver.py       ← Multi-device pmap Regge solver
│   ├── regge_shard_map.py         ← Multi-device shard_map Regge solver (ShardedReggeSolver)
│   ├── regge_shard_map_profiler.py← Profiled shard_map + jax.profiler traces
│   ├── regge_shard_map_memory_snapshot.py ← Memory snapshot instrumentation
│   ├── regge_shard_map_tensorboard.py     ← TensorBoard event writer for trajectories
│   └── callbacks/
│       ├── hessian_pl_callback.py             ← Base PL Hessian callback
│       ├── checkpointed_hessian_pl.py         ← Gradient-checkpointed HVP
│       ├── distributed_hessian_pl.py          ← DDP-aware Lanczos callback
│       ├── zero3_hessian_pl.py                ← ZeRO-3/FSDP Hessian + adaptive LR
│       ├── zero3_compressed_hessian_pl.py     ← ZeRO-3 + 1-bit gradient compression
│       ├── fp8_zero3_hessian_pl.py            ← FP8 (e4m3fn) Hessian + MGS re-orthog
│       ├── zeroinfinity_fp8_hessian_pl.py     ← ZeRO-∞ + FP8 combined
│       └── zeroinfinity_cpu_fallback_pl.py    ← CPU offload fallback under OOM
├── tests/                         ← Pytest suite covering roadmap claims
├── configs/params.yaml            ← Central tolerance ledger
├── scripts/
│   ├── run_suite.sh               ← Local / CI entrypoint (timeout-guarded)
│   ├── deploy_gce.sh              ← Preemptible GCE launcher + GCS upload
│   ├── deploy_profiler_gce.sh     ← Profiler GCE variant
│   └── launch_tensorboard_proxy.sh← Proxy for remote TensorBoard
└── docker/Dockerfile              ← Reproducible build image
```

---

## ◈ Core Physics Parameters

```yaml
# configs/params.yaml — the single source of truth
roadmap:
  M2_GeV:          2.4e23   # Fakeon mass threshold (GeV) — UV cutoff
  f2_target:       1.0e-8   # Gravity coupling target at M2
  alpha_inelastic: 0.05     # S-matrix inelasticity decay constant
  froissart_C:     0.25     # Froissart-Martin coefficient
assumptions:
  pl_constant_lb:     2.4e-2   # Polyak-Łojasiewicz μ lower bound
  lyapunov_decay_min: 1.5e-2   # Lyapunov V_opt decay floor γ
  thermal_T_reh_max:  1.0e15   # Reheating temperature ceiling (GeV)
```

<!-- VISUAL: Physics constants card grid — 7 parameter tiles, each with symbol, value, physical meaning, and which test it gates. Dark background, monochrome with teal accent on value. -->

---

## ◈ Computational Environment — Strengths & Weaknesses

### Strengths

**1. JAX end-to-end autodiff on physical kernels**  
Every Regge pole condition `f(l_re, s, δ)` is JAX-traceable. Newton-Raphson
uses exact `jvp` tangent propagation — no finite-difference noise. The
`lax.while_loop` construct keeps the iteration XLA-compilable and loop-unrolled
at the hardware level. FP64 is explicitly enabled (`jax_enable_x64=True`)
everywhere, which is non-default in JAX and matters for RGE integration at
14-decade energy scales (173 GeV → 2.4×10²³ GeV).

**2. Three parallel Regge backends at different granularity**  
`VectorizedReggeSolver` (vmap), `PMappedReggeSolver` (pmap), and
`ShardedReggeSolver` (shard_map) give a clear scaling ladder. `shard_map`
is the most recent JAX primitive with explicit mesh topology control and
`check_vma`/`check_rep` fallback guards for API compatibility across JAX
versions.

**3. Hessian spectrum monitoring as a first-class training primitive**  
The callback layer implements **distributed Lanczos** (k steps, all-reduce
aggregated eigenvalues) directly in the training loop. The PL-condition
`0.5‖∇L‖² ≥ μ(L − L*)` is verified online, and `η_opt = 1/(L + μ)` is
broadcast to all optimizer param groups every `monitor_every` steps. This
is a genuine adaptive second-order feedback loop, not a post-hoc analysis.

**4. Precision cascade with error telemetry**  
FP8 (e4m3fn / e5m2) is attempted first; per-tensor dynamic scales are
stored in `fp8_scales` dict; quantization error is measured against FP32
ground truth every HVP call; if `avg_q_err > 1e-3 ∧ μ < pl_tol` a BF16
fallback warning is emitted. The ZeRO-∞ / CPU-offload fallback chain
handles OOM gracefully without silent precision silencing.

**5. Mathematically closed test suite**  
Each of the 11 test files maps to a named roadmap claim. Tests are bounded
by tolerances in `params.yaml` — meaning the CI verdicts are parameter-traceable.
Physics predicates (Froissart bound, BRST nilpotency, Källén-Lehmann
normalization, crossing symmetry residuals) are all binary-certified.

**6. Deployment portability**  
The same codebase runs in Docker, local CPU, multi-GPU via DDP/FSDP, and
preemptible GCE with automatic JUnit XML upload to GCS. TensorBoard proxy
script gives remote observability with zero firewall changes.

<!-- VISUAL: Radar chart — 6 axes: Autodiff precision / Parallelism depth / Memory efficiency / Observability / Portability / Physics coverage. Filled polygon in teal against dark grid. -->

---

### Weaknesses

**1. Python GIL ceiling on the outer solver loop**  
`regge_bootstrap.py::track_regge_poles` runs a Python `for` loop over
`t_grid` (40 points), calling `root_scalar` per point. This is not
JAX-traced and cannot be XLA-compiled. Under multi-worker scenarios the
GIL serializes this entirely. The `VectorizedReggeSolver` solves this for
the shard_map path, but `ReggeExtendedBootstrap` still carries the Python loop.

**2. No shared mesh/sharding contract between solver and callback layers**  
The Regge solvers define their own `Mesh(np.array(devices), ("dev",))` and
the PyTorch Lightning callbacks operate in a separate distributed context
(`torch.distributed`). There is no unified device-topology abstraction —
JAX mesh and PyTorch process group are independently managed, preventing
co-scheduling.

**3. Lanczos with k=3-4 is numerically thin for large models**  
The distributed Lanczos in the callback layer uses k=3 (ZeRO-3) or k=4
(FP8) iterations. For ill-conditioned Hessians in large parameter spaces
this underestimates the spectral radius. The adaptive LR formula `η = 1/(L+μ)`
will be optimistic when L is underestimated, risking divergence in late
training.

**4. Static tolerance ledger with no runtime self-calibration**  
`params.yaml` values are fixed at commit time. There is no feedback path
where solver residuals can tighten or relax tolerances across runs. If a
new physics regime requires tighter `atol=1e-12` for a specific test, the
entire suite re-runs at that tolerance.

**5. FP8 path is best-effort with version dependency**  
`torch.float8_e4m3fn` is availability-checked via `getattr(torch, ...)`.
If PyTorch < 2.1 or the build lacks FP8 hardware support, the path silently
falls back to FP32 — which is correct behavior, but the `quantization_error`
telemetry then records zeros, giving a false signal of zero quantization
loss.

**6. No proto/schema layer — all inter-module contracts are dict literals**  
Every solver returns a Python `dict` with string keys (`"status"`,
`"Re_alpha_at_M2"`, `"fakeon_virtualized"`, etc.). There is no typed schema,
no serialization contract, no versioning. This is the largest structural
debt against a cross-language refactor.

<!-- VISUAL: Risk matrix — 2D grid: Impact (low→high) vs Likelihood (low→high). Six labeled nodes for weaknesses above, sized by fix complexity. Dark background, amber risk nodes. -->

---

## ◈ Module Deep-Dive

### `SIQGRGESolver` — 2-Loop RGE Integration

Integrates a 9-component coupling vector  
`g = [λ_H, λ_S, λ_HS, y_t, g₁, g₂, g₃, f₂, ξ_H]`  
from the top-mass scale (173.1 GeV) to the fakeon threshold M₂ = 2.4×10²³ GeV
using `scipy.solve_ivp` RK45.

The β-function for the gravity coupling `f₂` is the critical path:

```
β_{f₂} = −(133/20)f₂³ + (5196/5)/16π² · f₂⁵ − 12λ_HS ξ_H² / 16π² · f₂³
```

Two-loop gravitational closure is the hard constraint this solver certifies.

---

### `DiscretizedBootstrapSolver` + `ReggeExtendedBootstrap` — S-Matrix Bootstrap

The bootstrap minimizes a compound loss:

```
ℒ = Σ_{l,s} max(|S_l(s)|² − 1, 0)²  +  10⁻³ · Σ_s |A(s) − A(s)|²
```

Phase shifts `δ_l(s)` are the free variables. L-BFGS-B is the optimizer.
On convergence, `ReggeExtendedBootstrap.track_regge_poles` analytically
continues `S_l` to complex angular momentum and extracts `Re[α(t)]` via
Brentq root-finding on the pole condition:

```
Im[1/(1 − S_c(l_re + iε))] = 0
```

Fakeon virtualization is certified when `Re[α(M₂²)] < 0`.

---

### `JAXHessianEstimator` — Exact Gauss-Newton HVP

Uses the JVP/VJP composition trick for memory-efficient Hessian-vector products:

```python
_, jv = jvp(constraint_fn, (theta,), (v,))     # forward tangent
_, vjp_fn = jax.vjp(constraint_fn, theta)       # reverse cotangent
H_GN·v = vjp_fn(W * jv)[0] + Λ·v
```

No explicit Hessian is materialized. Lanczos iterates `k` times to extract
extremal eigenvalues for the PL-condition certificate.

---

### `ShardedReggeSolver` — `shard_map` Multi-Device Regge Scan

```
Mesh: ("dev",) over all available JAX devices
PartitionSpec: ("dev",) on N_t grid points
Inner kernel: vmap(newton_step) per device shard
```

Newton-Raphson uses `lax.while_loop` with `jvp`-derived analytic gradient
of the pole condition — fully XLA-compilable, zero Python loop overhead.

<!-- VISUAL: Data flow diagram — N_t input grid split into device shards, each feeding vmap(newton), outputs gathered and returned. Show mesh topology with device labels. -->

---

### Callback Layer — Distributed Hessian Telemetry

All eight callback variants share the same core pattern:

```
on_train_batch_end (every monitor_every steps, after warmup)
  │
  ├── _gather_parameters     ← FSDP / ZeRO parameter materialization
  ├── _forward_loss          ← Gradient-checkpointed forward pass
  ├── [HVP]                  ← grad → grad_v → hvp via autograd
  ├── _distributed_lanczos   ← k Lanczos steps, all_reduce eigenvalues
  ├── μ_global, L_global     ← min/max eigenvalue + reg_lambda
  ├── PL check               ← 0.5‖∇L‖²/gap ≥ pl_tol
  ├── η_opt = 1/(L + μ)      ← adaptive LR computation
  ├── dist.broadcast(η_opt)  ← rank-0 broadcasts to all workers
  ├── optimizer LR update    ← applied to all param_groups
  └── log_dict               ← TensorBoard metrics (sync_dist=True)
```

The FP8 variant adds MGS re-orthogonalization in the Lanczos loop and
tracks per-tensor quantization error with a 10-step rolling mean.

---

## ◈ Quickstart

### Local

```bash
python -m pip install -U pip
pip install -r requirements.txt
bash scripts/run_suite.sh
```

### Docker

```bash
docker build -t quft-engine -f docker/Dockerfile .
docker run --rm quft-engine
```

### GCE Preemptible

```bash
export BUCKET="your-verify-bucket"
bash scripts/deploy_gce.sh
# Instance runs suite, uploads junit.xml to gs://$BUCKET/, shuts down.
```

### JAX Sharded Regge

```python
from src.regge_shard_map import ShardedReggeSolver
import jax.numpy as jnp

solver = ShardedReggeSolver(N_t=256)
delta_mock = jnp.zeros(256) + 0.05
trajectory = solver.scan_regge_trajectory_sharded(delta_mock)
certificate = solver.verify_fakeon_virtualization(trajectory)
print(certificate["status"])  # "VERIFIED" | "PENDING"
```

### ZeRO-3 / FSDP Training with Hessian Telemetry

```python
from pytorch_lightning import Trainer
from src.callbacks.zero3_hessian_pl import Zero3CheckpointedHessianPLCallback

trainer = Trainer(
    strategy="fsdp",           # or "deepspeed_stage_3"
    callbacks=[Zero3CheckpointedHessianPLCallback(monitor_every=25)],
    accelerator="gpu",
    devices=4,
)
```

### FP8 Hessian Variant

```python
from src.callbacks.fp8_zero3_hessian_pl import FP8Zero3HessianPLCallback

trainer = Trainer(
    strategy="fsdp",
    callbacks=[FP8Zero3HessianPLCallback(k_lanczos=4, fp8_mode="e4m3fn")],
    accelerator="gpu",
    devices=8,
)
```

---

## ◈ Test Coverage Map

| Test File | Roadmap Claim | Key Assertion |
|---|---|---|
| `test_regression.py` | Baseline stability | Coupling endpoints within bounds |
| `test_flow_fixed_point.py` | ERG flow convergence | Residuals < 1e-8 at IR/UV |
| `test_spectral_rep.py` | Källén-Lehmann | ρ ≥ 0, ∫ρ dμ² = 1 ± 0.01 |
| `test_nonperturbative_unitarity.py` | Bootstrap unitarity | \|S_l\| ≤ 1 across all (l,s) |
| `test_aks_theorem.py` | AKS / BRST | s² = 0 symbolically |
| `test_bootstrap_jax.py` | S-matrix bootstrap | Crossing + unitarity residuals |
| `test_vmap_regge_distributed.py` | Regge vmap | Re[α(M₂²)] < 0 |
| `test_regge_pl_integration.py` | PL + Regge | Joint certification |
| `test_pmap_checkpoint_integration.py` | pmap correctness | Trajectory matches vmap |
| `test_shardmap_zero3_integration.py` | shard_map + ZeRO-3 | End-to-end distributed |
| `test_memory_cpu_fallback.py` | OOM resilience | CPU fallback activates |
| `test_profiler_compression_integration.py` | Profiler + FP8 | Trace & quant error logged |
| `test_gce_fp8_integration.py` | GCE FP8 path | Deploy + profiler + metrics |

<!-- VISUAL: Test coverage sunburst chart — inner ring = test category (symbolic/numerical/distributed/deployment), outer ring = individual test files. Color-coded by pass/pending status. -->

---

## ◈ Dev Message

> **To the engineer about to tear this stack down and rebuild it in Rust/Go/proto/k8s:**  
> Read this before you touch the Python. It is the spec.

---

This engine implements a **physics certification pipeline**, not a training
loop. The distinction matters for every architectural decision in the rewrite.
Here is what you are inheriting, system by system.

### The Physics Contract

The central object is the **fakeon prescription** in SIQG. Fakeons are
off-shell degrees of freedom whose Regge trajectories satisfy `Re[α(M₂²)] < 0`.
This is not a heuristic — it is the hard binary output of the certification
pipeline. Every module is either computing a piece of this certificate or
validating a mathematical constraint that the certificate depends on.

The **nine-coupling RGE vector** `[λ_H, λ_S, λ_HS, y_t, g₁, g₂, g₃, f₂, ξ_H]`
is the state. It flows from 173 GeV to 2.4×10²³ GeV. The β-function for `f₂`
has a two-loop gravitational term `(5196/5)/16π² · f₂⁵` that is the core
closure claim. If you move this to a Rust ODE integrator, your tolerances are
`rtol=1e-8, atol=1e-10` and you need RK45 or Dormand-Prince. Euler will not
close the 14-decade integration.

The **S-matrix bootstrap** is a constrained optimization in `δ_l(s)` space.
The current solver is L-BFGS-B on a flat `(N_l × N_s)` array. In the rewrite
this should be expressed as a **protobuf message** with fields:
`phase_shifts` (repeated double, shape `[N_l, N_s]`), `s_grid` (repeated
double), `froissart_check` (nested message). The bootstrap objective and its
JAX gradients should move to a **Go service** exposing a gRPC endpoint; the
Python caller becomes a thin client.

The **Regge pole extraction** is the most compute-intensive path. The
`lax.while_loop` Newton-Raphson is the right primitive — keep it. What changes
in the Rust core is the **mesh topology management**: replace the Python
`jax.Mesh` with an **eBPF-instrumented XLA device allocator** that maps
physical device UUIDs to mesh axes at the kernel level. The `shard_map` call
signature is stable; what you are replacing is the Python scaffolding that
constructs the `Mesh` object.

### The Hessian Callbacks — What They Actually Do

The eight callback variants are all implementations of the same algorithm:
**online Gauss-Newton Hessian monitoring via distributed Lanczos**. The
relevant mathematical objects are:

- `μ_global`: minimum Lanczos eigenvalue + `reg_lambda` → PL constant estimate
- `L_global`: maximum Lanczos eigenvalue + `reg_lambda` → Lipschitz constant estimate
- `η_opt = 1/(L + μ)` → adaptive learning rate
- PL condition: `0.5‖∇L‖² / (L − L*) ≥ μ_lb = 2.4×10⁻²`

In the k8s rewrite, these callbacks become a **sidecar container** running a
gRPC stream. The `on_train_batch_end` hook posts a `HessianSnapshot` proto
message (eigenvalues, grad_norm_sq, loss_gap, step_id) to the sidecar. The
sidecar computes `η_opt`, pushes it back via a `LRUpdate` proto, and emits
the metrics to a **Prometheus exporter** scraped by a Helm-deployed Grafana
stack. The PyTorch Lightning `log_dict` calls are replaced by a single
`metrics.push(snapshot)` call.

The FP8 quantization error path (`avg_q_err > 1e-3 ∧ μ < pl_tol → BF16 fallback`)
should become a **Helm value** (`hessian.fp8.errorThreshold: 1e-3`) that the
sidecar reads from a ConfigMap and acts on autonomously.

### Serialization — The Zero-Schema Problem

Every inter-module return value is currently a Python `dict`:

```python
{"status": "VERIFIED", "Re_alpha_at_M2": -0.03, "trajectory": [...], ...}
```

In the rewrite, **every one of these dicts has an exact proto3 equivalent**.
The mapping is:

```protobuf
message FakeonCertificate {
  enum Status { PENDING = 0; VERIFIED = 1; }
  Status status = 1;
  double re_alpha_at_m2 = 2;
  double im_alpha_at_m2 = 3;
  bool fakeon_virtualized = 4;
  repeated double trajectory = 5;
  repeated double t_grid = 6;
  bytes signature = 7;  // SHA-256 of (t_grid || trajectory) for audit trail
}

message BootstrapResult {
  bool success = 1;
  double unitarity_residual = 2;
  double crossing_residual = 3;
  FroissartCheck froissart = 4;
  bytes phase_shifts_b64 = 5;  // base64-encoded float64 array
}

message RGEEndpoint {
  repeated double g_uv = 1;
  repeated double g_ir = 2;
  bool success = 3;
  int32 nfev = 4;
}
```

The `bytes signature` field is where SHA-256 comes in. Every `FakeonCertificate`
emitted by the Rust Regge solver should carry a deterministic SHA-256 of the
`(t_grid ‖ trajectory)` byte array so that downstream consumers can verify
they are acting on the same trajectory without re-running the computation. The
Go gRPC service validates this hash before accepting a certificate for storage.

The `phase_shifts_b64` field uses base64 because proto3 repeated double is
expensive for large 2D arrays. Pack the `[N_l, N_s]` float64 array as raw
bytes, base64-encode it, decode on the consumer. The Rust core will emit this
directly from its ndarray output.

### The Cesium Layer

The `t_grid` (Regge t-channel momentum transfer) and `trajectory` (complex
angular momentum) are 1D arrays of physical observables that evolve over
training steps. In the rewrite, the **Cesium time-series database** is the
natural persistence layer for these. Each solver run produces a
`(timestamp, entity_id="regge_trajectory", properties={t_grid, alpha_re, alpha_im})`
Cesium event. The Go service writes this via the Cesium HTTP API; the
Helm chart deploys Cesium alongside the solver workers. The TensorBoard
trajectory plots are replaced by Cesium native queries with time-range filters.

### Rust Core Scope

The Rust core owns:
- Regge pole Newton-Raphson inner loop (port `lax.while_loop` logic)
- Froissart bound check (`spectral_density.py::check_froissart_bound`)
- RGE β-function evaluation (`rge_solver.py::rhs`)
- SHA-256 certificate signing
- Proto serialization/deserialization of all solver outputs

Rust does **not** own:
- JAX shard_map dispatch (keep in Python/JAX, call Rust via PyO3 for inner loops)
- PyTorch Lightning callbacks (keep in Python, communicate via gRPC sidecar)
- Bootstrap L-BFGS-B optimizer (keep in Python until Go gRPC service is stable)

The handoff boundary is the `FakeonCertificate` proto message. Rust produces
it, Go validates and stores it, Python consumes it for test assertions.

### k8s Deployment Shape

```
┌─────────────────────────────────────────────────────────┐
│  Helm release: qft-engine                               │
│                                                         │
│  solver-worker (DaemonSet, GPU nodes)                   │
│    └─ Python/JAX + Rust PyO3 extension                  │
│         └─ emits FakeonCertificate protos to sidecar    │
│                                                         │
│  hessian-sidecar (sidecar container per worker pod)     │
│    └─ Go gRPC service                                    │
│         ├─ validates SHA-256 signatures                  │
│         ├─ writes to Cesium                              │
│         └─ pushes LRUpdate back to worker               │
│                                                         │
│  metrics-exporter (Deployment)                          │
│    └─ Prometheus scrape target                          │
│         └─ Grafana dashboard (Helm dependency)          │
│                                                         │
│  cesium (StatefulSet)                                   │
│    └─ trajectory time-series persistence                │
│                                                         │
│  bootstrap-api (Deployment)                             │
│    └─ Go gRPC, wraps L-BFGS-B bootstrap solver          │
│         └─ returns BootstrapResult proto                 │
└─────────────────────────────────────────────────────────┘
```

The eBPF probe attaches to the XLA device allocator in each `solver-worker`
pod and exports device utilization metrics to the Prometheus exporter without
any application-level instrumentation changes.

<!-- VISUAL: k8s architecture diagram — pod layout matching the ASCII above, rendered as a clean topology diagram. Rust nodes in orange, Go nodes in blue, Python/JAX nodes in teal, Cesium in violet. -->

---

## ◈ CI / Workflow

The GitHub Actions workflow (`.github/workflows/quft-verify.yml`) performs:

1. Docker build
2. The verification/test steps defined in the workflow
3. Upload of `results_*.xml` test result artifacts

All 11 test files must pass. Physics tolerance failures are reported as
`FAILED` with the residual value and the `params.yaml` bound that was violated.

---

<!-- VISUAL: Footer banner — narrow dark strip with "QFT-Engine · QÜFT Verification Suite" left-aligned, repo URL right-aligned, version / commit hash center. -->

*Physics predicates as code. Certification as CI.*
