# QANTIS

**Quantum Autonomous Navigation, Tracking & Intelligence System**

Quantum computing platform for autonomous systems: **POMDP belief-state estimation** and **multi-hypothesis tracking (MHT)** using quantum annealing, QAOA, and amplitude estimation — with hardware validation on IBM Heron processors.

Built by **Neura Parse Ltd** as a [uv](https://docs.astral.sh/uv/) workspace monorepo.

---

## Packages

| Package | Description | Key Algorithms |
|---------|-------------|----------------|
| **quantum-common** | Shared infrastructure: backend abstraction, error mitigation, config, benchmarking | ZNE, PEC, Pauli twirling, readout mitigation, noise modelling |
| **quantum-pomdp** | Quantum-enhanced POMDP planning for autonomous navigation | QBRL, BIQAE, Grover-AA belief oracle, quantum belief update circuits |
| **quantum-mht** | Quantum multi-hypothesis tracking for drone swarm surveillance | QUBO formulation, D-Wave annealing, QAOA, FPC-QAOA, Kalman filtering |

---

## Architecture

```
qantis/
  packages/
    quantum-common/    # backends, mitigation, config — shared library
    quantum-pomdp/     # POMDP belief estimation  (depends on quantum-common)
    quantum-mht/       # multi-hypothesis tracking (depends on quantum-common)
  configs/             # backend + experiment YAML configs
  scripts/hardware/    # IBM/D-Wave hardware run scripts
  output/hardware/     # hardware job results (JSON, timestamped)
```

Heavyweight quantum SDKs (Qiskit, D-Wave Ocean, PennyLane, Mitiq) are **optional dependencies** — the core packages install cleanly with only NumPy/SciPy.

---

## Requirements

- **Python 3.12+**
- **uv** package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))

---

## Installation

```bash
git clone https://github.com/neuraparse/qantis.git
cd qantis
uv sync
```

### Optional quantum backends

```bash
uv pip install "quantum-common[ibm]"        # Qiskit + Aer simulator
uv pip install "quantum-common[dwave]"      # D-Wave Ocean
uv pip install "quantum-common[pennylane]"  # PennyLane / QAOA
uv pip install "quantum-common[mitiq]"      # ZNE, PEC error mitigation
uv pip install "quantum-common[viz]"        # matplotlib, plotly
uv pip install "quantum-common[all]"        # everything
```

### Hardware credentials

```bash
cp .env.example .env
# Set IBM_QUANTUM_TOKEN and/or DWAVE_API_TOKEN in .env
```

---

## Quick Start

### POMDP — Hybrid quantum–classical planning

```python
from quantum_pomdp.scenarios.tiger_problem import TigerProblem
from quantum_pomdp.models.belief_state import BeliefState
from quantum_pomdp.algorithms.qbrl import QBRLPlanner, QBRLConfig

tiger = TigerProblem.create()
belief = BeliefState.uniform(tiger.num_states)
planner = QBRLPlanner(tiger, QBRLConfig(horizon=3, num_simulations=100))
action = planner.select_action(belief)
print(f"Recommended action: {action}")
```

### MHT — QUBO-based multi-target data association

```python
import numpy as np
from quantum_mht.formulation.cost_matrix import CostMatrixBuilder
from quantum_mht.formulation.mtda_qubo_builder import MTDAQuboBuilder
from quantum_mht.solvers.solver_factory import create_solver

cost_matrix = CostMatrixBuilder().build(
    track_predictions=np.array([[1.0, 2.0], [3.0, 4.0]]),
    measurements=np.array([[1.1, 2.1], [3.1, 3.9], [5.0, 6.0]]),
    gate_threshold=9.21,
)
qubo = MTDAQuboBuilder().build(cost_matrix)
solution = create_solver("hungarian").solve(qubo)
print(f"Assignments: {solution.assignments}")
```

---

## Tests

```bash
uv run pytest                              # all unit tests
uv run pytest packages/quantum-pomdp/     # single package
uv run pytest --cov=quantum_common --cov=quantum_pomdp --cov=quantum_mht
uv run pytest -m integration              # requires live backend credentials
uv run pytest -m benchmark
```

Tests requiring optional SDKs are automatically skipped if not installed.

---

## Running on IBM Hardware

```bash
export IBM_QUANTUM_TOKEN=<your-token>

uv run python scripts/hardware/run_fpc_qaoa_ibm.py
# Results saved to output/hardware/<experiment>_ibm_<timestamp>.json
```

Backend settings: `configs/backends/ibm_quantum.yaml`

---

## Project Structure

```
packages/
  quantum-common/src/quantum_common/
    config/              # Pydantic schemas, YAML loader, credentials (SecretStr)
    backends/            # IBM, D-Wave, PennyLane, Azure, local simulator
    mitigation/          # ZNE, PEC, Pauli twirling, readout, noise pipeline
    benchmarking/        # metrics, runner, storage, reproducibility
    visualization/       # circuit, benchmark, style plotting
  quantum-pomdp/src/quantum_pomdp/
    models/              # POMDPModel, BeliefState, BayesianNetwork
    quantum_circuits/    # belief update, register map, unitaries, amplitude amplification
    algorithms/          # QBRL planner, BIQAE estimator, lookahead tree
    scenarios/           # Tiger, 4-state Tiger, GPS-denied, grid navigation
    pipeline/            # hybrid quantum-classical pipeline
    classical_baselines/ # POMCP, DESPOT, PBVI
  quantum-mht/src/quantum_mht/
    formulation/         # cost matrix, QUBO builder, constraints, variables
    solvers/             # QAOA, FPC-QAOA, annealing, hybrid, Hungarian, JPDA
    tracking/            # Kalman, EKF, gating, track lifecycle
    simulation/          # drone swarm, dynamics, targets, world model
    pipeline/            # 5-stage pipeline (predict→gate→associate→update→manage)
```

---

## Supported Backends

| Backend | Extra | Hardware |
|---------|-------|----------|
| IBM Qiskit | `[ibm]` | ibm\_marrakesh (156q Heron R2), ibm\_torino (133q Heron R1), Aer simulator |
| D-Wave Ocean | `[dwave]` | Advantage2 (4400+ qubits, Zephyr) |
| PennyLane | `[pennylane]` | Variational circuits, QAOA, multiple plugins |
| Azure Quantum | `[ibm]` | IonQ Aria-2 (25q), Quantinuum H2 (56q) |
| Local simulator | *(included)* | CPU — development and testing |

---

## Development

```bash
uv run ruff check packages/
uv run ruff format packages/
uv run mypy packages/
uv run pre-commit run --all-files
```

---

## License

MIT — see [LICENSE](LICENSE).

**Neura Parse Ltd.**
