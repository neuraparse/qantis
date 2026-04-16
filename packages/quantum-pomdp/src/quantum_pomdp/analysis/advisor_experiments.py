"""Publication-oriented experiment helpers for the QCE26 case-study paper.

These helpers are intentionally conservative:
- they provide simulator and classical-baseline evidence that directly answers
  reviewer/advisor questions,
- they avoid overstating runtime claims,
- and they keep the paper's validated core centred on belief inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate

from quantum_common.backends.base import ExecutionRequest
from quantum_common.backends.simulator import AerSimulatorBackend
from quantum_pomdp.classical_baselines.despot import DESPOTSolver
from quantum_pomdp.classical_baselines.pbvi import PBVISolver
from quantum_pomdp.classical_baselines.pomcp import POMCPSolver
from quantum_pomdp.models.belief_state import BeliefState
from quantum_pomdp.models.pomdp import POMDPModel
from quantum_pomdp.scenarios.gps_denied import create_gps_denied_pomdp
from quantum_pomdp.scenarios.grid_navigation import create_grid_navigation_pomdp
from quantum_pomdp.scenarios.tiger_problem import create_tiger_pomdp

TIGER_ACTION_NAMES = ["listen", "open-left", "open-right"]
CORRIDOR_LEFT_OBS = np.array([0.85, 0.70, 0.30, 0.15], dtype=float)


def create_corridor_tiger_pomdp() -> POMDPModel:
    """Create a 4-state Tiger-style POMDP used as the first step beyond |S|=2."""
    num_states = 4
    num_actions = 3
    num_observations = 2

    transition = np.zeros((num_actions, num_states, num_states), dtype=float)
    transition[0] = np.eye(num_states)
    transition[1] = np.full((num_states, num_states), 0.25, dtype=float)
    transition[2] = np.full((num_states, num_states), 0.25, dtype=float)

    observation = np.zeros((num_actions, num_states, num_observations), dtype=float)
    observation[0, :, 0] = CORRIDOR_LEFT_OBS
    observation[0, :, 1] = 1.0 - CORRIDOR_LEFT_OBS
    observation[1] = 0.5
    observation[2] = 0.5

    reward = np.full((num_states, num_actions), -1.0, dtype=float)
    reward[:, 1] = np.array([-100.0, -30.0, 5.0, 10.0], dtype=float)
    reward[:, 2] = np.array([10.0, 5.0, -30.0, -100.0], dtype=float)

    return POMDPModel(
        num_states=num_states,
        num_actions=num_actions,
        num_observations=num_observations,
        transition_tensor=transition,
        observation_tensor=observation,
        reward_matrix=reward,
        discount_factor=0.95,
    )


def _pbvi_action_from_belief(
    solver: PBVISolver,
    belief: np.ndarray,
    alphas: list[np.ndarray],
    model: POMDPModel,
) -> int:
    values = []
    for action in range(model.num_actions):
        alpha = solver._backup(belief, action, alphas, model)
        values.append(float(alpha @ belief))
    return int(np.argmax(values))


def tiger_classical_baseline_report() -> list[dict[str, Any]]:
    """Return action recommendations from standard classical POMDP baselines."""
    model = create_tiger_pomdp()
    beliefs = {
        "uniform": np.array([0.5, 0.5], dtype=float),
        "moderately_left": np.array([0.85, 0.15], dtype=float),
        "strongly_left": np.array([0.97, 0.03], dtype=float),
        "strongly_right": np.array([0.03, 0.97], dtype=float),
    }

    pbvi = PBVISolver(num_belief_points=50, max_iterations=20)
    alphas = pbvi.solve(model)

    report: list[dict[str, Any]] = []
    for label, belief in beliefs.items():
        pomcp = POMCPSolver(num_simulations=2000, max_depth=6, seed=42)
        despot = DESPOTSolver(num_scenarios=1000, max_depth=6, seed=42)

        solver_actions = {
            "POMCP": int(pomcp.select_action(belief, model)),
            "DESPOT": int(despot.select_action(belief, model)),
            "PBVI": int(_pbvi_action_from_belief(pbvi, belief, alphas, model)),
        }
        report.append(
            {
                "belief_label": label,
                "belief": belief.tolist(),
                "recommended_actions": solver_actions,
                "recommended_action_names": {
                    name: TIGER_ACTION_NAMES[action]
                    for name, action in solver_actions.items()
                },
            }
        )

    return report


def _build_corridor_tiger_circuit(prior: list[float]) -> QuantumCircuit:
    """3-qubit simulator circuit for the 4-state Corridor Tiger sensor update."""
    p = [float(x) for x in prior]
    p01 = float(np.clip(p[0] + p[1], 1e-9, 1.0))
    p23 = float(np.clip(p[2] + p[3], 1e-9, 1.0))

    qc = QuantumCircuit(3, 3, name="corridor_tiger_4state")

    theta_q0 = 2.0 * float(np.arccos(np.sqrt(np.clip(p01, 1e-9, 1.0))))
    qc.ry(theta_q0, 0)

    cond_p0 = p[0] / p01
    theta_q1_given_0 = 2.0 * float(np.arccos(np.sqrt(np.clip(cond_p0, 1e-9, 1.0))))
    qc.x(0)
    qc.cry(theta_q1_given_0, 0, 1)
    qc.x(0)

    cond_p2 = p[2] / p23
    theta_q1_given_1 = 2.0 * float(np.arccos(np.sqrt(np.clip(cond_p2, 1e-9, 1.0))))
    qc.cry(theta_q1_given_1, 0, 1)

    theta_obs = [
        2.0 * float(np.arccos(np.sqrt(np.clip(prob, 1e-9, 1.0))))
        for prob in CORRIDOR_LEFT_OBS
    ]
    qc.x(0)
    qc.x(1)
    qc.append(RYGate(theta_obs[0]).control(2), [0, 1, 2])
    qc.x(0)
    qc.x(1)

    qc.x(0)
    qc.append(RYGate(theta_obs[1]).control(2), [0, 1, 2])
    qc.x(0)

    qc.x(1)
    qc.append(RYGate(theta_obs[2]).control(2), [0, 1, 2])
    qc.x(1)

    qc.append(RYGate(theta_obs[3]).control(2), [0, 1, 2])
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def _corridor_posterior_from_counts(
    counts: dict[str, int],
    target_observation: int,
) -> list[float]:
    filtered = np.zeros(4, dtype=float)
    for bitstring, count in counts.items():
        cleaned = bitstring.replace(" ", "")
        if len(cleaned) != 3 or int(cleaned[0]) != target_observation:
            continue
        state_idx = 2 * int(cleaned[2]) + int(cleaned[1])
        filtered[state_idx] += count

    total = filtered.sum()
    if total <= 0:
        return [0.25, 0.25, 0.25, 0.25]
    return (filtered / total).tolist()


def corridor_tiger_4state_simulator_report(
    prior: list[float] | None = None,
    observation: int = 0,
    shots: int = 8192,
) -> dict[str, Any]:
    """Simulator validation for the first step beyond the 2-state Tiger unit cell."""
    prior = prior or [0.25, 0.25, 0.25, 0.25]
    model = create_corridor_tiger_pomdp()

    classical = BeliefState(np.array(prior, dtype=float)).classical_update(
        action=0,
        observation=observation,
        transition_tensor=model.transition_tensor,
        observation_tensor=model.observation_tensor,
    )

    circuit = _build_corridor_tiger_circuit(prior)
    backend = AerSimulatorBackend()
    counts = backend.execute(ExecutionRequest(circuits=[circuit], shots=shots)).counts[0]
    simulator = BeliefState(np.array(_corridor_posterior_from_counts(counts, observation), dtype=float))

    return {
        "prior": [float(x) for x in prior],
        "observation": observation,
        "shots": shots,
        "num_qubits": circuit.num_qubits,
        "logical_depth": circuit.depth(),
        "classical_posterior": classical.probabilities.tolist(),
        "simulator_posterior": simulator.probabilities.tolist(),
        "hellinger_distance": simulator.hellinger_distance(classical),
    }


def scenario_resource_pathway_report() -> list[dict[str, Any]]:
    """Summarize circuit width and evidence-conditioning difficulty by scenario."""
    tiger = create_tiger_pomdp()
    corridor = create_corridor_tiger_pomdp()
    grid4 = create_grid_navigation_pomdp(grid_size=4)
    gps4 = create_gps_denied_pomdp(grid_size=4)
    gps6 = create_gps_denied_pomdp(grid_size=6)

    scenarios = [
        {
            "label": "Tiger-2",
            "model": tiger,
            "belief": np.array([0.97, 0.03], dtype=float),
            "action": 0,
            "observation": 1,
            "context": "contradicting listen observation under concentrated prior",
        },
        {
            "label": "CorridorTiger-4",
            "model": corridor,
            "belief": np.array([0.70, 0.20, 0.07, 0.03], dtype=float),
            "action": 0,
            "observation": 1,
            "context": "contradicting corridor-hear-right observation",
        },
        {
            "label": "GridNavigation-4x4",
            "model": grid4,
            "belief": np.ones(grid4.num_states, dtype=float) / grid4.num_states,
            "action": 4,
            "observation": grid4.num_observations - 1,
            "context": "single-cell observation under uniform prior",
        },
        {
            "label": "GPSDenied-4x4",
            "model": gps4,
            "belief": np.ones(gps4.num_states, dtype=float) / gps4.num_states,
            "action": 4,
            "observation": gps4.num_observations - 1,
            "context": "terrain signature observation under uniform prior",
        },
        {
            "label": "GPSDenied-6x6",
            "model": gps6,
            "belief": np.ones(gps6.num_states, dtype=float) / gps6.num_states,
            "action": 4,
            "observation": gps6.num_observations - 1,
            "context": "terrain signature observation under uniform prior",
        },
    ]

    report: list[dict[str, Any]] = []
    for entry in scenarios:
        model = entry["model"]
        belief = entry["belief"]
        evidence_probability = model.observation_probability(
            belief, entry["action"], entry["observation"]
        )
        classical_queries = 1.0 / max(evidence_probability, 1e-12)
        quantum_query_scale = 1.0 / np.sqrt(max(evidence_probability, 1e-12))
        report.append(
            {
                "label": entry["label"],
                "context": entry["context"],
                "num_states": model.num_states,
                "num_actions": model.num_actions,
                "num_observations": model.num_observations,
                "state_qubits": model.state_qubits,
                "action_qubits": model.action_qubits,
                "observation_qubits": model.observation_qubits,
                "total_circuit_qubits": model.total_circuit_qubits,
                "representative_evidence_probability": float(evidence_probability),
                "classical_queries_per_accept": float(classical_queries),
                "quantum_query_scale_only": float(quantum_query_scale),
                "asymptotic_sample_efficiency_gain": float(
                    classical_queries / max(quantum_query_scale, 1e-12)
                ),
            }
        )

    return report
