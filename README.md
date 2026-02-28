# QANTIS

**Quantum Autonomous Navigation, Tracking & Intelligence System**

Quantum computing algorithms for autonomous systems: **POMDP belief-state estimation** and **multi-hypothesis tracking (MHT)** using quantum annealing, QAOA, and amplitude estimation.

Built by **Neura Parse Ltd** as a [uv](https://docs.astral.sh/uv/) workspace monorepo with three independent packages sharing a common infrastructure layer.

---

## Packages

| Package | Description | Key Algorithms |
|---------|-------------|----------------|
| **quantum-common** | Shared infrastructure: backend abstraction, error mitigation, config, benchmarking, visualization | ZNE, PEC, readout mitigation, noise modelling |
| **quantum-pomdp** | Quantum-enhanced POMDP planning for autonomous navigation under uncertainty | QBRL, BIQAE, quantum belief update circuits, lookahead tree |
| **quantum-mht** | Quantum multi-hypothesis tracking for drone swarm surveillance | QUBO formulation, D-Wave annealing, QAOA, FPC-QAOA, Kalman filtering |

## Architecture

```
qantis/                               # workspace root
  packages/
    quantum-common/                   # shared library (backends, mitigation, config)
    quantum-pomdp/                    # POMDP belief estimation  (depends on quantum-common)
    quantum-mht/                      # multi-hypothesis tracking (depends on quantum-common)
```

Both `quantum-pomdp` and `quantum-mht` depend on `quantum-common` as a workspace dependency. Heavyweight quantum SDKs (Qiskit, D-Wave Ocean, PennyLane, Mitiq) are **optional dependencies** so the core packages install cleanly with only NumPy/SciPy.

---

## Requirements

- **Python 3.12+** (tested on 3.13)
- **uv** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))

## Installation

### 1. Clone and sync

```bash
git clone <repo-url> qantis
cd qantis
uv sync
```

This installs the three workspace packages in editable mode along with dev dependencies (pytest, ruff, mypy).

### 2. Install optional quantum backends

Install only the backends you need:

```bash
# IBM Qiskit (gate-based circuits, Aer simulator)
uv pip install "quantum-common[ibm]"

# D-Wave Ocean (quantum annealing, simulated annealing)
uv pip install "quantum-common[dwave]"

# PennyLane (variational circuits, QAOA)
uv pip install "quantum-common[pennylane]"

# Mitiq (error mitigation: ZNE, PEC)
uv pip install "quantum-common[mitiq]"

# Visualization (matplotlib, plotly)
uv pip install "quantum-common[viz]"

# Everything
uv pip install "quantum-common[all]"
```

### 3. Verify installation

```bash
uv run python -c "
import quantum_common; print('quantum-common', quantum_common.__version__)
import quantum_pomdp;  print('quantum-pomdp',  quantum_pomdp.__version__)
import quantum_mht;    print('quantum-mht',    quantum_mht.__version__)
"
```

---

## Quick Start

### POMDP Belief-State Estimation

```python
import numpy as np
from quantum_pomdp.scenarios.tiger_problem import TigerProblem
from quantum_pomdp.models.belief_state import BeliefState
from quantum_pomdp.algorithms.qbrl import QBRLPlanner, QBRLConfig

# Create the classic Tiger POMDP
tiger = TigerProblem.create()
belief = BeliefState.uniform(tiger.num_states)

# Configure QBRL planner (quantum Bayesian reinforcement learning)
config = QBRLConfig(horizon=3, num_simulations=100)
planner = QBRLPlanner(tiger, config)

# Select the best action given current belief
action = planner.select_action(belief)
print(f"Recommended action: {action}")
```

### Multi-Hypothesis Tracking (QUBO formulation)

```python
import numpy as np
from quantum_mht.formulation.cost_matrix import CostMatrixBuilder
from quantum_mht.formulation.mtda_qubo_builder import MTDAQuboBuilder
from quantum_mht.solvers.solver_factory import create_solver

# Build a cost matrix from tracks and measurements
cost_builder = CostMatrixBuilder()
cost_matrix = cost_builder.build(
    track_predictions=np.array([[1.0, 2.0], [3.0, 4.0]]),
    measurements=np.array([[1.1, 2.1], [3.1, 3.9], [5.0, 6.0]]),
    gate_threshold=9.21,
)

# Convert to QUBO and solve
qubo_builder = MTDAQuboBuilder()
qubo = qubo_builder.build(cost_matrix)

# Solve with classical Hungarian (or "annealing" with D-Wave installed)
solver = create_solver("hungarian")
solution = solver.solve(qubo)
print(f"Assignments: {solution.assignments}")
```

### GPS-Denied Navigation Scenario

```python
from quantum_pomdp.scenarios.gps_denied import GPSDeniedScenario

# Create a GPS-denied navigation POMDP
scenario = GPSDeniedScenario.create(grid_size=8, sensor_noise=0.15)
pomdp = scenario.to_pomdp()
print(f"States: {pomdp.num_states}, Actions: {pomdp.num_actions}")
```

---

## Running Tests

```bash
# Run all unit tests (excludes integration/benchmark by default)
uv run pytest

# Verbose output
uv run pytest -v

# Run tests for a specific package
uv run pytest packages/quantum-common/tests/
uv run pytest packages/quantum-pomdp/tests/
uv run pytest packages/quantum-mht/tests/

# Run with coverage
uv run pytest --cov=quantum_common --cov=quantum_pomdp --cov=quantum_mht

# Run integration tests (requires live backend credentials)
uv run pytest -m integration

# Run benchmarks
uv run pytest -m benchmark
```

Tests that require optional dependencies (qiskit, dwave, mitiq) are automatically **skipped** when those packages are not installed.

---

## Project Structure

```
qantis/
  pyproject.toml                         # workspace root config
  conftest.py                            # shared pytest fixtures
  packages/
    quantum-common/
      src/quantum_common/
        config/                          # Pydantic schemas, YAML loader, credentials
        backends/                        # IBM, D-Wave, PennyLane, Azure, simulator
        mitigation/                      # ZNE, PEC, readout, noise models, pipeline
        benchmarking/                    # metrics, runner, storage, reproducibility
        visualization/                   # circuit, benchmark, and style plotting
        logging/                         # structured logging (structlog)
        types.py                         # shared type definitions
        exceptions.py                    # custom exception hierarchy
      tests/
    quantum-pomdp/
      src/quantum_pomdp/
        models/                          # POMDPModel, BeliefState, BayesianNetwork
        quantum_circuits/                # belief update, register map, unitaries, AA
        algorithms/                      # QBRL planner, BIQAE estimator, lookahead tree
        scenarios/                       # Tiger, grid navigation, GPS-denied
        pipeline/                        # hybrid quantum-classical pipeline
        classical_baselines/             # POMCP, DESPOT, PBVI
        sensing/                         # IMU fusion, sensor interface
        analysis/                        # metrics, scalability analysis
      tests/
    quantum-mht/
      src/quantum_mht/
        formulation/                     # cost matrix, QUBO builder, constraints, variables
        solvers/                         # QAOA, FPC-QAOA, annealing, hybrid
          classical_solvers/             # Hungarian, Reid MHT, JPDA, GNN
        tracking/                        # Kalman, EKF, gating, track lifecycle
        fusion/                          # covariance intersection, sensor models
        simulation/                      # dynamics, drones, targets, world, swarm
        pipeline/                        # tracking pipeline with 5 stages
          stages/                        # predict, gate, associate, update, manage
        experiments/                     # QUBO scaling, annealing vs QAOA, etc.
        benchmarking/                    # benchmark runner, scenario suite
      tests/
```

---

## Configuration

Experiments are configured via YAML files or Pydantic models:

```yaml
# experiment.yaml
name: tiger_qbrl
backend:
  backend_type: local_aer
  use_simulator: true
  shots: 10000
mitigation:
  zne_enabled: true
  zne_scale_factors: [1, 2, 3]
  readout_mitigation: true
```

```python
from quantum_common.config.loader import load_experiment_config

config = load_experiment_config("experiment.yaml")
```

Environment variables override YAML values (prefixed with `QAR_`).

---

## Supported Backends

| Backend | Package | Hardware | Notes |
|---------|---------|----------|-------|
| IBM Qiskit | `quantum-common[ibm]` | Heron R3 (156q), Eagle (127q) | Gate-based circuits, Aer simulator |
| D-Wave Ocean | `quantum-common[dwave]` | Advantage2 (4400+ qubits, Zephyr 20-way) | Quantum annealing, LeapHybrid |
| PennyLane | `quantum-common[pennylane]` | Multiple via plugins | Variational circuits, QAOA |
| Azure Quantum | `quantum-common[ibm]` | IonQ Aria-2 (25q), Quantinuum H2 (56q) | Cloud access |
| Local simulator | (included) | CPU | Development and testing |

---

## Key Academic References

This codebase implements and extends algorithms from the following primary sources:

- **QBRL** (arXiv:2507.18606, Jul 2025) - Hybrid quantum-classical POMDP planning via quantum Bayesian reinforcement learning
- **BIQAE** (Quantum 10:1962, Jan 2026) - Bayesian iterative quantum amplitude estimation with adaptive K-schedule
- **FPC-QAOA** (arXiv:2512.21181, Dec 2025) - Fixed-parameter-count QAOA for combinatorial optimization
- **Quantum MHT** (Ihara 2025, Sci. Rep. 15:24294) - Multi-object tracking via quantum annealing with reverse annealing warm-start
- **QUBO MHT** (arXiv:2110.08346) - Fraunhofer FKIE multi-target data association as QUBO
- **ZNE/PEC extensions** (Quantum q-2026-02-10-2003, Feb 2026) - Error mitigation generalized to non-Clifford gates
- **Q-CTRL Ironstone** (2025) - 111x accuracy improvement, 4m precision over 700km for GPS-denied navigation

Full annotated bibliography with 900+ references is embedded in source docstrings throughout the codebase.

---

## Development

```bash
# Lint
uv run ruff check packages/

# Format
uv run ruff format packages/

# Type check
uv run mypy

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

Neura Parse Ltd.
