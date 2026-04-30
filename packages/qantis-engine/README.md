# qantis-engine

> **Status: dry-run research prototype, 2026-04-27.** Companion to the
> four-layer roadmap stated in `qantis-paper/sections/07-conclusion.tex`.
> *Not* hardware-validated end to end. No "we beat Gurobi" claim is made
> on the basis of this prototype. See
> `qantis-paper/notes/decision-engine-vision-2026-04-27.md` for the full
> vision document and the explicit list of validations still owed.

QANTIS Decision Engine — a four-pillar API for stochastic decision problems
under partial observability, rare observations, and chance constraints.

## Slogan

> Gurobi solves deterministic models. QANTIS solves the belief-to-action loop
> under fixed decision and sample budgets.

## Four pillars

| Module | Purpose | Key 2026 anchors |
|--------|---------|------------------|
| `qantis_engine.infer`    | Belief update / posterior conditioning              | arXiv:2603.00785 (QANTIS), Quantum 10:1962 (BIQAE) |
| `qantis_engine.risk`     | Rare-event probability / CVaR / chance constraints  | arXiv:2602.09059, 2602.09847, 2603.15664, 2604.20088 |
| `qantis_engine.optimize` | Constraint-native quantum/hybrid optimizer          | arXiv:2601.01516, 2604.02083, 2604.07218, 2603.21283 |
| `qantis_engine.verify`   | Feasibility, dual bound, closed-loop regret         | arXiv:2509.11040, 2511.19501 |

## Quickstart

```python
import numpy as np
from qantis_engine import infer, risk, optimize, verify

# 1) Posterior under a noisy observation
prior = np.array([0.5, 0.5])
obs_model = np.array([[0.85, 0.15], [0.15, 0.85]])
post = infer.posterior(prior, observation=0, obs_model=obs_model, mode="classical")

# 2) Rare-event tail probability via BIQAE
def world(rng): return bool(rng.uniform() < 0.07)
def collision(s): return s
rep = risk.tail_probability(world, collision, mode="biqae",
                            n_samples=1024, target_amplitude=0.07, seed=0)
print(rep.estimate, rep.credible_interval)

# 3) Constraint-native optimization with chance constraint
cost_matrix = np.random.default_rng(0).uniform(0, 5, size=(5, 7))
plan = optimize.solve(cost_matrix=cost_matrix, deadline_s=0.05, seed=0)

# 4) Verify
chk = verify.check_assignment(plan.assignments, n_tracks=5, n_meas=7,
                              missed_detections=plan.missed_detections,
                              false_alarms=plan.false_alarms)
print(chk.feasible, chk.violations)

# 5) Closed-loop benchmark vs Gurobi-style baselines
from qantis_engine.bench import (BenchConfig, generate_rare_event_mtda,
                                 run_benchmark)
from qantis_engine.bench.pipelines import run_qantis_full
cfg = BenchConfig("rare_event_mtda", "qantis_full",
                  deadline_ms=100.0, seed=0, n_steps=20)
result = run_benchmark(
    scenario_factory=lambda n, s: generate_rare_event_mtda(
        n_steps=n, n_targets=4, p_miss=0.15, seed=s),
    pipeline_factory=run_qantis_full, config=cfg)
print(result.metrics.cumulative_regret)
```

## Benchmark v0: QANTIS vs Gurobi-centred pipelines

Four pipelines, same scenario stream, same seed, same deadline:

- **A** `classical_pf_gurobi`   — particle filter + Hungarian assignment
- **B** `smc_gurobi`            — SMC + Hungarian
- **C** `qantis_infer_gurobi`   — QANTIS-Infer (BIQAE) + Hungarian
- **D** `qantis_full`           — QANTIS-Infer + QANTIS-Optimize hybrid B&B

Closed-loop metrics: ID switches, missed/false tracks, deadline-miss rate,
sample cost, posterior Hellinger, cumulative regret.

Hungarian (scipy `linear_sum_assignment`) substitutes for Gurobi at the
assignment frontier — the LP relaxation of the assignment polytope has the
integrality property, so for pure assignment Hungarian is provably equal to
Gurobi (Kuhn 1955). For chance-constrained extensions a Gurobi adapter slots
behind the same `pipeline_factory` interface.

## Defending against critics

- arXiv:2604.20180 (Watanabe-Sels-Tindall, Apr 2026) shows shallow QAOA can
  be matched by 2D tensor-network surrogates on heavy-hex / square Ising.
  We deliberately live in three regimes where TN surrogates are weak:
  rare-event posteriors (P ≪ 1), deeper FPC-/IWS-XY mixers, and
  closed-loop deadline-bounded metrics.

## Run tests

```bash
uv sync --all-packages
.venv/bin/pytest packages/qantis-engine/tests -q --no-header
```

Expect **35 passed, 0 failed** (qiskit not required for the engine layer).

## Open validations

Listed in detail in
`qantis-paper/notes/decision-engine-vision-2026-04-27.md`. Headlines:

- Pipelines A/B/C use `scipy.linear_sum_assignment` (Hungarian) as the
  Gurobi stand-in. A real Gurobi adapter is needed before any
  "Gurobi-vs-QANTIS" framing leaves dry-run mode.
- Pipeline D's quantum sampler falls back to classical Hamming-Weight
  Gibbs. Closed-loop hardware integration with `quantum_mht.solvers.
  XYMixerQAOASolver` on IBM Heron R3 is not done.
- A TN-surrogate baseline (per arXiv:2604.20180 Watanabe-Sels-Tindall,
  Apr 2026) is *not* in the engine yet; required to defend any
  closed-loop advantage claim.
- BIQAE classical emulation aliases on mid-amplitude probabilities; the
  fix is to route through `CalibratedBIQAEEstimator` (boundary-aware
  Beta prior).
