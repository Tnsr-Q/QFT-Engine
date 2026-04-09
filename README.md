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
| Schema contracts | Pydantic `PhysicsPredicate` + `@enforce_schema` (`src/proto/`) |
| Mesh abstraction | `UnifiedMesh` singleton with JAX/PyTorch adapters (`src/mesh/`) |
| Tolerance governance | Dynamic ledger (`src/tolerance/`) + regime detector |
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
│   ├── hessian_qjax.py            ← Quantized JAX Hessian (FP8 precision)
│   ├── regge_jax_solver.py        ← JAX-native Regge pole tracker (JAXReggePoleTracker)
│   ├── regge_vmap_solver.py       ← Single-device vmap Regge solver
│   ├── regge_pmap_solver.py       ← Multi-device pmap Regge solver
│   ├── regge_shard_map.py         ← Multi-device shard_map Regge solver (ShardedReggeSolver)
│   ├── regge_shard_map_profiler.py← Profiled shard_map + jax.profiler traces
│   ├── regge_shard_map_memory_snapshot.py ← Memory snapshot instrumentation
│   ├── regge_shard_map_tensorboard.py     ← TensorBoard event writer for trajectories
│   ├── unified_topology.py        ← Hybrid classical/quantum topology manager
│   ├── rl_conjecture_loop.py      ← RL conjecture-refinement loop (PPO agent)
│   ├── proto/                     ← Schema system — validation & serialization
│   │   ├── constraint_schema.py   ← PhysicsPredicate contract (StatusLevel, AssumptionTag)
│   │   ├── registry.py            ← Versioned predicate registry + dependency graph
│   │   ├── return_schemas.py      ← FakeonCertification, UnifiedMeshResults, MeshExecutionScheme
│   │   ├── schema_enforcer.py     ← @enforce_schema decorator for return validation
│   │   ├── serializer.py          ← Multi-format serializer (JSON, protobuf, Arrow, pickle)
│   │   └── orbax_atomic.py        ← Atomic checkpoint I/O (Orbax fallback)
│   ├── mesh/                      ← Distributed device topology abstraction
│   │   ├── topology.py            ← DeviceTopology ABC, JAXMeshAdapter, PyTorchMeshAdapter
│   │   ├── unified_mesh.py        ← UnifiedMesh singleton (auto-detect JAX/PyTorch)
│   │   └── schemes.py             ← FSDP detection + JAX-FSDP scheme unification
│   ├── tolerance/                 ← Dynamic tolerance governance
│   │   ├── dynamic_ledger.py      ← Self-calibrating tolerance updates + audit logs
│   │   └── regime_detector.py     ← UV/IR/stiffness/PL regime classification
│   ├── callbacks/                 ← PyTorch-Lightning distributed callbacks
│   │   ├── hessian_pl_callback.py             ← Base PL Hessian callback
│   │   ├── checkpointed_hessian_pl.py         ← Gradient-checkpointed HVP
│   │   ├── distributed_hessian_pl.py          ← DDP-aware Lanczos callback
│   │   ├── zero3_hessian_pl.py                ← ZeRO-3/FSDP Hessian + adaptive LR
│   │   ├── zero3_compressed_hessian_pl.py     ← ZeRO-3 + 1-bit gradient compression
│   │   ├── fp8_zero3_hessian_pl.py            ← FP8 (e4m3fn) Hessian + MGS re-orthog
│   │   ├── zeroinfinity_fp8_hessian_pl.py     ← ZeRO-∞ + FP8 combined
│   │   ├── zeroinfinity_cpu_fallback_pl.py    ← CPU offload fallback under OOM
│   │   └── precision_controller.py             ← Runtime precision state controller
│   ├── truth/                     ← Universality kernel & epistemic boundary
│   │   ├── universality_kernel.py ← f₂-space scan for unique fixed points
│   │   └── epistemic_guard.py     ← Claim-level epistemic boundary enforcement
│   ├── discovery/                 ← Theory-space exploration
│   │   └── theory_space.py        ← Symbolic-numeric hybrid theory generation
│   └── spectral/                  ← Spectral density & robust estimation
│       └── robust_estimator.py    ← Distributed Hessian spectral estimator
├── tests/                         ← Pytest suite covering roadmap claims (20 files)
├── configs/
│   ├── params.yaml                ← Physics + solver baseline parameters
│   └── tolerance_priors.yaml      ← Dynamic tolerance priors (runtime-adapted)
├── scripts/
│   ├── run_suite.sh               ← Local / CI entrypoint (ledger-aware, timeout-guarded)
│   ├── deploy_gce.sh              ← Preemptible GCE launcher + GCS upload
│   ├── deploy_profiler_gce.sh     ← Profiler GCE variant
│   ├── deploy_universal.sh        ← Universal multi-cloud deployment orchestrator
│   ├── diagnose_precision.py      ← Precision diagnostics (detect backend, FP8 effects)
│   └── launch_tensorboard_proxy.sh← Proxy for remote TensorBoard
└── docker/Dockerfile              ← Reproducible build image
```

---

## ◈ Core Physics Parameters

```yaml
# configs/params.yaml — baseline physical + numerical constants
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

```yaml
# configs/tolerance_priors.yaml — adaptive ledger priors
rge_atol:
  base_tol: 1.0e-10
  min_tol: 1.0e-14
  max_tol: 1.0e-6
  adaptation_rate: 0.15
```

`scripts/run_suite.sh` executes pytest with `--tolerance-ledger=configs/tolerance_priors.yaml`
and supports `--freeze` (freeze updates) plus `--audit-verify` (print deterministic checksum).

<!-- VISUAL: Physics constants card grid — 7 parameter tiles, each with symbol, value, physical meaning, and which test it gates. Dark background, monochrome with teal accent on value. -->

---

## ◈ Computational Environment — Strengths & Weaknesses

### Strengths

**1. Strong numerical core for physics-grade stability**
The stack is explicitly designed around high-precision numerical behavior:
FP64-enabled JAX kernels for sensitive derivatives, robust SciPy integration
for long-scale RG running, and deterministic tolerance governance that keeps
residuals explainable instead of “magic-number tuned.”

**2. Multi-backend execution strategy is practical, not ornamental**
Having vmap, pmap, and shard_map implementations provides a credible path from
single-device debugging to multi-device throughput. This reduces migration
friction when moving from local validation to cluster-scale experiments.

**3. Built-in spectral/Hessian diagnostics improve training trust**
The callback layer does more than logging — it estimates curvature and checks
PL-style conditions online, then updates optimization behavior from measured
signals. This is a meaningful reliability feature for stiff, non-convex
objectives.

**4. Resilience-first precision and memory fallbacks**
The precision cascade and ZeRO/CPU-offload style fallbacks prioritize job
continuity under hardware pressure (OOM, unsupported low-precision modes),
which is critical for long-running verification pipelines.

**5. Clear physics-to-test traceability**
The repository structure maps physical claims to concrete checks and tolerances.
That claim → module → test pattern is one of the strongest maintainability
characteristics in the project.

**6. Deployment and observability are already production-aware**
Dockerization, remote profiling hooks, TensorBoard support, and cloud scripts
mean operational concerns are considered early, not retrofitted after research
code stabilizes.

---

### Weaknesses

**1. Mixed execution model increases complexity at boundaries**
The project combines SciPy, JAX, and PyTorch Lightning effectively, but this
also creates orchestration seams (device placement, distributed runtime
assumptions, and data handoff conventions) that are easy to regress.

**2. Some hot paths remain Python-loop bound**
Where root-finding and trajectory tracking stay in Python control flow,
parallel scaling is constrained and difficult to optimize with accelerator
compilation alone.

**3. Distributed topology unification is in progress**
The `UnifiedMesh` singleton (`src/mesh/unified_mesh.py`) and the
`DeviceTopology` ABC (`src/mesh/topology.py`) provide a shared contract
for JAX and PyTorch device coordination. However, not all solver entry
points route through the unified mesh yet — some still construct
framework-specific meshes directly.

**4. Curvature estimation is lightweight relative to model scale risk**
Short Lanczos runs are computationally efficient but can under-sample sharp or
ill-conditioned spectra, making adaptive step-size decisions overconfident in
late training.

**5. Adaptive tolerances are present but not fully pervasive**
The tolerance ledger exists and is valuable, yet enforcement still depends on
specific execution paths. Consistent adoption across all solver entry points is
not complete.

**6. Schema enforcement coverage is growing but incomplete**
The `src/proto/` schema layer (`@enforce_schema`, `PhysicsPredicate`,
`FakeonCertification`) now provides typed Pydantic contracts for key
solver outputs. However, some internal helper functions still return
untyped dicts. Full schema coverage across every inter-module boundary
remains a work in progress.


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

Nine callback modules are included (`hessian_*`, `zero3_*`, `zeroinfinity_*`, and
`precision_controller.py`). The Hessian-focused variants share this core pattern:

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

## ◈ Schemas · Mesh · Ledgers — The Governance Triad

Three subsystems — the **schema layer** (`src/proto/`), the **mesh layer**
(`src/mesh/`), and the **tolerance ledger** (`src/tolerance/`) — form a
closed governance loop that every physics computation passes through.
No solver output escapes without schema validation, no device sharding
occurs outside the mesh contract, and no tolerance drifts without a
ledger entry. This section explains each subsystem, then traces how
they compose into a single verification pipeline.

---

### Schema Layer — `src/proto/`

The schema layer enforces type safety, versioned identity, and
serialization contracts for every physics assertion the engine produces.

#### Core Contract: `PhysicsPredicate` (`constraint_schema.py`)

Every verifiable physics claim is a `PhysicsPredicate` instance — a
Pydantic `BaseModel` carrying:

| Field | Type | Purpose |
|---|---|---|
| `predicate_id` | `str` | Unique identifier (e.g. `"C_ghost"`) |
| `version` | `int` | Monotonic version for registry tracking |
| `statement` | `str` | Human-readable claim text |
| `mathematical_form` | `str` | LaTeX or SymPy expression |
| `assumptions` | `List[AssumptionTag]` | Tagged dependency list |
| `dependencies` | `List[str]` | IDs of upstream predicates |
| `tolerance` | `float` | Maximum acceptable residual |
| `residual` | `Optional[float]` | Measured residual after evaluation |
| `status` | `StatusLevel` | PROVED / VERIFIED / CALCULATED / DEMONSTRATED / PENDING |
| `boundary_checks` | `Dict` | Results from assumption boundary evaluation |
| `metadata` | `Dict` | Free-form context (regime, solver, etc.) |

Two Pydantic validators enforce invariants at construction time:

1. **`residual_must_respect_tolerance`** — if a residual is supplied, it
   must not exceed the declared tolerance.
2. **`status_requires_residual`** — a predicate cannot claim `VERIFIED`
   or `CALCULATED` status without a numeric residual.

Methods:
- `to_json()` → JSON string for audit logs and test fixtures.
- `to_protobuf()` → protobuf-compatible bytes for cross-language consumers.
- `check_assumption_boundaries()` → evaluates each `AssumptionTag` against
  the predicate's metadata and returns a `Dict[AssumptionTag, bool]` map.

Enums that scope the contract:

| Enum | Values | Role |
|---|---|---|
| `StatusLevel` | PROVED, VERIFIED, CALCULATED, DEMONSTRATED, PENDING | Epistemic grade |
| `ConstraintRole` | SELECTOR, CLOSURE, CONSISTENCY | How the predicate is used |
| `AssumptionTag` | A1\_PERTURBATIVE, A2\_FAKEON\_VALID, A3\_PALATINI\_COMPAT, A4\_SCALE\_INVARIANT, A5\_PORTAL\_DOMINANCE | Physics assumption labels |

#### Registry: `PredicateRegistry` (`registry.py`)

A global in-memory store for versioned predicates with dependency
indexing. Every predicate registered here is reachable by ID and
version, and its dependency edges form a directed acyclic graph.

| Method | Signature | Purpose |
|---|---|---|
| `register` | `(predicate: PhysicsPredicate) → bool` | Insert with version dedup |
| `get_latest` | `(predicate_id: str) → Optional[PhysicsPredicate]` | Fetch highest version |
| `propagate_assumption_failure` | `(failed_tag: AssumptionTag) → List[str]` | Return IDs affected by a broken assumption |
| `predicate_ids` | `() → List[str]` | List all registered IDs |
| `dependency_graph` | `(predicate_id: str, depth: int = 3) → Dict[str, List[str]]` | Traverse dependency tree to given depth |

When a physics assumption fails (e.g. perturbativity breakdown),
`propagate_assumption_failure` identifies every predicate that
transitively depends on the broken tag — the registry becomes a
failure-propagation bus.

#### Return Schemas (`return_schemas.py`)

Three Pydantic models constrain solver output shapes:

| Schema | Fields | Used by |
|---|---|---|
| `FakeonCertification` | `Re_alpha_at_M2`, `fakeon_virtualized`, `trajectory`, `status`, `t_grid` | `ShardedReggeSolver`, `JAXReggePoleTracker` |
| `UnifiedMeshResults` | `jax`, `torch` | `UnifiedMesh.co_schedule` |
| `MeshExecutionScheme` | `mesh_axes`, `fsdp_enabled`, `fsdp_sharding_strategy`, `checkpoint_backend` | `UnifiedMesh.get_execution_scheme` |

#### Enforcement: `@enforce_schema` (`schema_enforcer.py`)

A decorator that wraps any function returning a `dict` and validates the
payload against a Pydantic schema before returning:

```python
@enforce_schema(FakeonCertification)
def verify_fakeon_virtualization(self, trajectory, ...):
    ...
    return {"Re_alpha_at_M2": alpha, "status": "VERIFIED", ...}
```

The decorator constructs a schema instance from the dict, triggering all
Pydantic validators. If validation passes, the original dict is returned
(preserving backward compatibility). If validation fails, the Pydantic
`ValidationError` propagates unmodified. This guarantees that every
decorated solver output has been structurally certified without requiring
callers to change their consumption patterns.

#### Serialization: `Serializer` (`serializer.py`)

Multi-format serialization for predicates and arbitrary dicts:

| Method | Signature | Formats |
|---|---|---|
| `serialize` | `(obj, format="json", version="1.0.0") → str \| bytes` | json, protobuf, arrow, pickle |
| `deserialize` | `(data, format="json", target_cls=PhysicsPredicate) → Any` | json, protobuf, arrow, pickle |
| `compute_checksum` | `(data) → str` | SHA-256 hex digest |

The checksum method underpins audit-trail integrity — every serialized
predicate or tolerance snapshot can be fingerprinted for deterministic
replay verification.

#### Atomic Persistence: `OrbaxAtomicStateIO` (`orbax_atomic.py`)

Provides crash-safe checkpoint I/O. Uses Orbax when available; falls
back to a temp-file-rename pattern otherwise:

- `save(state: dict) → None` — atomic write to disk.
- `restore() → dict` — read last committed checkpoint.

---

### Mesh Layer — `src/mesh/`

The mesh layer provides a framework-agnostic distributed device
abstraction so that physics solvers can shard tensors and synchronize
results without coupling to JAX or PyTorch internals.

#### Abstract Contract: `DeviceTopology` (`topology.py`)

An abstract base class defining the minimal device coordination API:

| Method | Return | Purpose |
|---|---|---|
| `get_device_count()` | `int` | Number of available accelerators |
| `get_device_type()` | `DeviceType` | CPU / GPU / TPU / IPU |
| `shard_tensor(tensor, axis, partition_spec)` | sharded tensor | Distribute data across mesh axis |
| `all_reduce(tensor, op="sum")` | tensor | Collective reduction |
| `barrier()` | `None` | Global synchronization fence |
| `get_rank()` | `int` | Current process rank |
| `get_world_size()` | `int` | Total process count |

Two enums scope the topology:

| Enum | Values |
|---|---|
| `DeviceType` | CPU, GPU, TPU, IPU |
| `MeshAxis` | DATA, MODEL, PIPELINE, EXPERT, SPECTRAL |

#### JAX Adapter: `JAXMeshAdapter` (`topology.py`)

Implements `DeviceTopology` using `jax.sharding.Mesh`, `NamedSharding`,
and `PartitionSpec`. All-reduce delegates to `jax.lax.psum` / `pmean`.
The reshape spec is derived from the `mesh_axes` constructor parameter,
which maps logical axis names to physical device ordinals.

```python
adapter = JAXMeshAdapter(mesh_axes=("data", "model"), devices=jax.devices())
sharded = adapter.shard_tensor(tensor, MeshAxis.DATA, ("data", None))
```

#### PyTorch Adapter: `PyTorchMeshAdapter` (`topology.py`)

Implements `DeviceTopology` using `torch.distributed`. Supports `"gloo"`
and `"nccl"` backends. All-reduce uses `dist.all_reduce`.

```python
adapter = PyTorchMeshAdapter(backend="nccl")
adapter.all_reduce(grad_tensor, op="sum")
```

#### Singleton Coordinator: `UnifiedMesh` (`unified_mesh.py`)

A singleton that bridges JAX and PyTorch mesh topologies behind a single
interface. On `initialize(backend="auto")`, it auto-detects framework
availability and constructs the appropriate adapter(s). Framework-
specific kwargs are filtered via `_JAX_PARAMS` / `_TORCH_PARAMS`
introspection sets.

| Method | Signature | Purpose |
|---|---|---|
| `initialize` | `(backend="auto", **kwargs)` | Detect & construct adapters |
| `get_topology` | `(framework=None) → DeviceTopology` | Fetch active adapter |
| `co_schedule` | `(jax_fn, torch_fn, shared_data) → Dict` | Execute both frameworks with barrier synchronization |
| `shard_across_frameworks` | `(data, axis, jax_spec, torch_spec) → Dict` | Dual-framework tensor sharding |
| `get_execution_scheme` | `() → dict` | Return `MeshExecutionScheme`-compatible dict |

`co_schedule` is the critical interop method: it runs a JAX function
and a PyTorch function in sequence, inserting barriers before and after
each, and returns a `UnifiedMeshResults`-validated dict containing both
outputs.

#### FSDP Helpers: `schemes.py`

Two utility functions unify FSDP detection across frameworks:

- `detect_fsdp(model) → bool` — checks whether a model is wrapped in
  `FullyShardedDataParallel`.
- `unify_jax_fsdp_scheme(mesh_axes, model) → dict` — returns a
  normalized scheme dict that merges JAX mesh config with FSDP state.

---

### Tolerance Ledger — `src/tolerance/`

The tolerance ledger is a self-calibrating precision manager.
It tracks per-solver tolerance bounds, adapts them from measured
residuals via an EWMA feedback rule, classifies the active physics
regime, and writes an immutable audit trail.

#### Dynamic Tolerance Manager: `DynamicToleranceLedger` (`dynamic_ledger.py`)

Initialized from `configs/tolerance_priors.yaml`, the ledger holds a
`ToleranceConfig` dataclass per solver key:

| Field | Type | Purpose |
|---|---|---|
| `base_tol` | `float` | Starting tolerance before adaptation |
| `min_tol` | `float` | Hard floor (precision cannot tighten past this) |
| `max_tol` | `float` | Hard ceiling (precision cannot relax past this) |
| `adaptation_rate` | `float` | EWMA smoothing constant α |
| `reference_residual` | `float` | Equilibrium residual for the update rule |
| `regime` | `str` | Physics regime label |
| `last_updated` | `str` | ISO timestamp of last mutation |

The adaptation formula is:

```
tol_{k+1} = clip( tol_k · exp(−α · (r_ewma / r_ref − 1)),  [min_tol, max_tol] )
```

where `r_ewma` is the exponentially weighted moving average of recent
residuals. This rule tightens tolerance when residuals are consistently
below the reference and relaxes it when residuals exceed it — always
staying within the bounded interval.

| Method | Signature | Purpose |
|---|---|---|
| `get_tolerance` | `(key, regime=None) → float` | Fetch current tolerance for a solver key |
| `update_from_residual` | `(key, residual, solver_id="unknown") → float` | Adapt tolerance from measured residual |
| `flush_audit` | `() → None` | Persist audit buffer to Parquet (or YAML fallback) |
| `export_snapshot` | `() → Dict[str, Any]` | Serialize all ledger state |

A `freeze_mode` flag (settable at init or via `--freeze` CLI flag)
disables all `update_from_residual` mutations, producing deterministic
replay runs where tolerances remain at their prior values.

The audit trail logs every mutation as a `(timestamp, key, old_tol,
new_tol, residual, solver_id)` tuple. `flush_audit()` writes the buffer
to Parquet when `pyarrow` is available, falling back to YAML otherwise.

**Tolerance keys** loaded from `configs/tolerance_priors.yaml`:

| Key | Base | Bounds | Regime |
|---|---|---|---|
| `rge_atol` | 1e-10 | [1e-14, 1e-6] | STIFF\_ODE |
| `hessian_pl` | 2.4e-2 | [1e-3, 5e-2] | HESSIAN\_PL |
| `bootstrap_unitarity` | 1e-4 | [1e-8, 5e-3] | NONPERT\_UNITARITY |
| `regge_pole` | 1e-8 | [1e-12, 1e-5] | UV\_FAKEON |

#### Regime Classification: `RegimeDetector` (`regime_detector.py`)

Classifies the active physics regime from solver state and residual
maps, selecting the tolerance band appropriate for the current
computation:

| Regime | Condition |
|---|---|
| `UV_FAKEON` | Energy scale μ ≥ 0.1 × M₂ and \|f₂ − target\| < 1e-6 |
| `IR_SM` | μ < 1 kGeV and residuals < 1e-4 |
| `STIFF_ODE` | Jacobian condition number > 50 or step size < 1e-6 |
| `HESSIAN_PL` | Hessian μ > 1e-3 |
| `NONPERT_UNITARITY` | Inelasticity or crossing residuals present |
| `DEFAULT` | No specific regime detected |

```python
detector = RegimeDetector(M2_GeV=2.4e23, f2_target=1e-8)
regime = detector.classify(solver_state={"mu": 1e22, "f2": 1e-8},
                           residuals={"pole": 3e-9})
ledger.get_tolerance("regge_pole", regime=regime)
```

---

### How They Compose: The Governance Loop

When a solver executes, the three subsystems form a closed loop:

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                        SOLVER EXECUTION                         │
 │  (RGE, Bootstrap, Regge, Hessian)                               │
 │                                                                  │
 │  1. Mesh Layer allocates devices                                 │
 │     UnifiedMesh.get_topology() → JAXMeshAdapter                  │
 │     ShardedReggeSolver distributes t-grid via shard_map          │
 │                                                                  │
 │  2. Solver computes result dict                                  │
 │     {"Re_alpha_at_M2": -0.03, "status": "VERIFIED", ...}        │
 │                                                                  │
 │  3. Schema Layer validates output                                │
 │     @enforce_schema(FakeonCertification) ─────────────────┐      │
 │     Pydantic validators fire → residual ≤ tolerance       │      │
 │                                                           │      │
 │  4. Ledger updates tolerance from residual                │      │
 │     DynamicToleranceLedger.update_from_residual()  ◄──────┘      │
 │     RegimeDetector.classify() selects tolerance band             │
 │     Audit entry appended                                         │
 │                                                                  │
 │  5. Registry tracks the predicate                                │
 │     PredicateRegistry.register(predicate)                        │
 │     Assumption failure → propagate_assumption_failure()          │
 │                                                                  │
 │  6. Serializer persists the result                               │
 │     Serializer.serialize(result, format="json")                  │
 │     Serializer.compute_checksum(data) → SHA-256 fingerprint      │
 └──────────────────────────────────────────────────────────────────┘
```

**Concrete example — Hessian PL certification on multi-device:**

1. `UnifiedMesh.initialize(backend="auto")` detects JAX, constructs
   `JAXMeshAdapter` with `mesh_axes=("dev",)`.
2. `HessianPLCallback.on_train_batch_end` invokes
   `JAXHessianEstimator.lanczos_eigenvalues(theta, k)`, which calls
   `hessian_vector_product` inside a `shard_map` over the device mesh.
3. Eigenvalues are all-reduced via `adapter.all_reduce(eigvals, "sum")`.
4. The callback computes `μ_global`, `L_global`, checks the PL condition,
   and returns a result dict.
5. `@enforce_schema(FakeonCertification)` validates the dict — status,
   trajectory shape, and residual-within-tolerance are all checked.
6. `DynamicToleranceLedger.update_from_residual("hessian_pl", residual)`
   adapts the next run's tolerance. `RegimeDetector.classify()` confirms
   the `HESSIAN_PL` regime, so tolerance stays in [1e-3, 5e-2].
7. `PredicateRegistry.register(predicate)` stores the versioned result.
   If any `AssumptionTag` later fails, `propagate_assumption_failure`
   marks this predicate's dependents.
8. `Serializer.compute_checksum(trajectory_bytes)` produces a SHA-256
   fingerprint that downstream consumers verify before acting on the
   trajectory.

This loop is the reason every solver in the engine can make certified
claims: the mesh guarantees correct distribution, the schema guarantees
structural validity, and the ledger guarantees that numerical precision
was governed, bounded, and auditable.

---

### Key Functions Reference — Governance Triad

| Function / Method | Location | Purpose |
|---|---|---|
| `PhysicsPredicate.to_json()` | `src/proto/constraint_schema.py` | Serialize predicate to JSON |
| `PhysicsPredicate.check_assumption_boundaries()` | `src/proto/constraint_schema.py` | Evaluate assumptions against metadata |
| `PredicateRegistry.register(predicate)` | `src/proto/registry.py` | Insert versioned predicate |
| `PredicateRegistry.propagate_assumption_failure(tag)` | `src/proto/registry.py` | Identify affected predicates |
| `PredicateRegistry.dependency_graph(id, depth)` | `src/proto/registry.py` | Traverse dependency DAG |
| `@enforce_schema(schema)` | `src/proto/schema_enforcer.py` | Decorator: validate return dict |
| `Serializer.serialize(obj, format)` | `src/proto/serializer.py` | Multi-format output |
| `Serializer.compute_checksum(data)` | `src/proto/serializer.py` | SHA-256 fingerprint |
| `OrbaxAtomicStateIO.save(state)` | `src/proto/orbax_atomic.py` | Crash-safe checkpoint write |
| `JAXMeshAdapter.shard_tensor(tensor, axis, spec)` | `src/mesh/topology.py` | JAX device sharding |
| `PyTorchMeshAdapter.all_reduce(tensor, op)` | `src/mesh/topology.py` | PyTorch collective reduce |
| `UnifiedMesh.initialize(backend, **kwargs)` | `src/mesh/unified_mesh.py` | Auto-detect & construct adapters |
| `UnifiedMesh.co_schedule(jax_fn, torch_fn, data)` | `src/mesh/unified_mesh.py` | Cross-framework execution |
| `UnifiedMesh.get_execution_scheme()` | `src/mesh/unified_mesh.py` | Return MeshExecutionScheme dict |
| `detect_fsdp(model)` | `src/mesh/schemes.py` | Check for FSDP wrapper |
| `DynamicToleranceLedger.get_tolerance(key, regime)` | `src/tolerance/dynamic_ledger.py` | Fetch current tolerance |
| `DynamicToleranceLedger.update_from_residual(key, r)` | `src/tolerance/dynamic_ledger.py` | Adapt tolerance from residual |
| `DynamicToleranceLedger.flush_audit()` | `src/tolerance/dynamic_ledger.py` | Write audit trail to Parquet/YAML |
| `RegimeDetector.classify(state, residuals)` | `src/tolerance/regime_detector.py` | Select physics regime |

---

## ◈ Quickstart

### Local

```bash
python -m pip install -U pip
pip install -r requirements.txt
pip install sympy
bash scripts/run_suite.sh
# Optional: freeze adaptive tolerances for deterministic replay
bash scripts/run_suite.sh --freeze --audit-verify
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
# Instance runs suite, uploads results/results.xml to gs://$BUCKET/, shuts down.
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
| `test_robust_spectral.py` | Spectral robustness | Stable estimator under perturbations |
| `test_tolerance_ledger.py` | Dynamic tolerance ledger | Bounded adaptation + auditability |
| `test_pr23_review_fixes.py` | Schema + mesh fixes | Registry IDs, kwargs filtering, IR consistency, JAX reshape |
| `test_hybrid_discovery_architecture.py` | Theory-space discovery | Hybrid symbolic-numeric exploration loop |
| `test_w_fixes_integration.py` | Integration fixes | Cross-module integration correctness |

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

### Serialization — Schema Contracts and the Proto3 Path

The `src/proto/` layer has introduced Pydantic-validated return schemas
(`FakeonCertification`, `UnifiedMeshResults`, `MeshExecutionScheme`) and
the `@enforce_schema` decorator, which now guards key solver outputs.
Inter-module return values are still Python `dict` objects at the
transport layer, but they are structurally validated before leaving the
decorated function:

```python
@enforce_schema(FakeonCertification)
def verify_fakeon_virtualization(self, ...):
    return {"status": "VERIFIED", "Re_alpha_at_M2": -0.03, "trajectory": [...], ...}
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

All test files under `tests/` must pass. Physics tolerance failures are reported as
`FAILED` with the residual value and the `params.yaml` bound that was violated.

---

<!-- VISUAL: Footer banner — narrow dark strip with "QFT-Engine · QÜFT Verification Suite" left-aligned, repo URL right-aligned, version / commit hash center. -->

*Physics predicates as code. Certification as CI.*
