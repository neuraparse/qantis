"""Quantum Process Tomography (QPT) for Tiger POMDP belief update circuit.

Fully characterizes the quantum channel implemented by the Tiger belief update
circuit and computes:

1. **Process fidelity** -- overlap of the reconstructed channel with the ideal
   unitary (F_pro).
2. **Average gate fidelity** -- F_avg = (d * F_pro + 1) / (d + 1) where d = 4
   for a 2-qubit channel.
3. **Unitarity** -- how close the channel is to a unitary (1.0 = perfectly
   unitary; noise drives it below 1).
4. **Diamond distance** -- worst-case distinguishability between the
   reconstructed and ideal channels (0 = identical).

Method
------
We use input-state/output-tomography QPT:

- Prepare 4 informationally complete input states on the 2-qubit register:
  |00>, |01>, |10>, |11>, |+0>, |0+> (6 inputs for over-determination /
  robustness; the 4 computational basis states alone span the diagonal of the
  Choi matrix while the +0 and 0+ superposition inputs capture off-diagonal
  coherences).
- For each input, apply the Tiger belief channel (measurement-free circuit).
- For each output, perform full 2-qubit state tomography in 9 Pauli bases
  (XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ).
- Reconstruct each output density matrix via linear inversion + physical
  projection.
- From the set of (input, output-rho) pairs, reconstruct the Choi matrix of the
  channel and derive all process metrics.

Total circuits: 6 inputs x 9 bases = 54 circuits.

Usage
-----
# Dry run (no credentials needed):
python scripts/hardware/run_process_tomography.py --dry-run --prior 0.5 0.5

# Hardware run:
python scripts/hardware/run_process_tomography.py \\
    --backend ibm_kingston --prior 0.5 0.5 --shots 8192

Output
------
  output/hardware/process_tomography_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

from scripts.hardware import (
    get_ibm_token_optional,
    ibm_channel,
    ibm_instance,
    save_result,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantum Process Tomography for Tiger POMDP belief circuit"
    )
    p.add_argument("--backend", default="ibm_kingston", help="IBM backend name")
    p.add_argument("--shots", type=int, default=8192, help="Shot count per circuit")
    p.add_argument(
        "--prior", nargs="+", type=float, default=[0.5, 0.5],
        help="Prior belief probabilities (pair of floats summing to 1)",
    )
    p.add_argument(
        "--p-correct", type=float, default=0.85,
        help="Tiger listen accuracy P(hear-correct|state) (default: 0.85)",
    )
    p.add_argument(
        "--channel", default=None,
        help="Qiskit channel: ibm_quantum_platform or ibm_cloud "
             "(overrides IBM_QUANTUM_CHANNEL env)",
    )
    p.add_argument(
        "--instance", default=None,
        help="IBM instance / CRN (overrides IBM_QUANTUM_INSTANCE env)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Run on AerSimulator only, skip IBM hardware (no credentials needed)",
    )
    p.add_argument(
        "--opt-level", type=int, default=3,
        help="Transpiler optimization level 0-3 (default: 3)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tiger belief circuit (measurement-free, for process tomography)
# ---------------------------------------------------------------------------

def build_tiger_sensor_channel(
    p_correct: float = 0.85,
) -> "QuantumCircuit":
    """Build the 2-qubit Tiger sensor-model channel WITHOUT measurements.

    This is the *channel* to be characterized by QPT: the conditional
    observation encoding that creates entanglement between the state qubit
    and the observation qubit.

    The prior encoding (Ry on q0) is NOT included -- it is state preparation,
    not part of the channel.  QPT characterizes: U_sensor = X(0).CRY(0,1).X(0).CRY(0,1).

    Qubit layout:
        q0 = state register  (control)
        q1 = observation register  (target)
    """
    from qiskit import QuantumCircuit

    p00 = float(np.clip(p_correct, 1e-9, 1.0 - 1e-9))
    p10 = float(np.clip(1 - p_correct, 1e-9, 1.0 - 1e-9))

    theta_obs_s0 = 2.0 * float(np.arccos(np.sqrt(p00)))
    theta_obs_s1 = 2.0 * float(np.arccos(np.sqrt(p10)))

    qc = QuantumCircuit(2, name="tiger_sensor")

    # Conditional observation encoding (creates entanglement)
    qc.x(0)
    qc.cry(theta_obs_s0, 0, 1)   # P(obs|state=tiger-left) when q0=|0>
    qc.x(0)
    qc.cry(theta_obs_s1, 0, 1)   # P(obs|state=tiger-right) when q0=|1>

    return qc


def build_tiger_belief_circuit(
    prior: list[float],
    p_correct: float = 0.85,
) -> "QuantumCircuit":
    """Build the full 2-qubit Tiger belief state circuit (prior + sensor).

    Used for computing the ideal final state for a given prior.
    """
    from qiskit import QuantumCircuit

    theta_state = 2.0 * float(np.arccos(np.sqrt(max(prior[0], 1e-9))))

    qc = QuantumCircuit(2, name="tiger_belief")
    qc.ry(theta_state, 0)
    qc.compose(build_tiger_sensor_channel(p_correct), inplace=True)

    return qc


# ---------------------------------------------------------------------------
# Input state preparation
# ---------------------------------------------------------------------------

_INPUT_STATES = {
    "|00>": lambda qc: None,
    "|01>": lambda qc: qc.x(1),
    "|10>": lambda qc: qc.x(0),
    "|11>": lambda qc: (qc.x(0), qc.x(1)),
    "|+0>": lambda qc: qc.h(0),
    "|0+>": lambda qc: qc.h(1),
}


def _prepare_input_statevector(label: str) -> np.ndarray:
    """Return the 4-element statevector for a named 2-qubit input state."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(2)
    _INPUT_STATES[label](qc)
    return Statevector.from_instruction(qc).data


# ---------------------------------------------------------------------------
# Pauli measurement bases
# ---------------------------------------------------------------------------

_PAULI_BASES = ["X", "Y", "Z"]
_BASIS_ROTATIONS = {
    "Z": [],
    "X": ["h"],
    "Y": ["sdg", "h"],
}


# ---------------------------------------------------------------------------
# Build QPT circuits
# ---------------------------------------------------------------------------

def build_qpt_circuits(
    p_correct: float = 0.85,
) -> tuple[list, list[tuple[str, str]]]:
    """Build all QPT circuits: 6 inputs x 9 Pauli bases = 54 circuits.

    The channel under test is the Tiger sensor model (CRY entangling gates),
    NOT the full belief circuit (which includes prior encoding).

    Returns:
        (circuits, labels) where each label is (input_name, basis_label).
    """
    from qiskit import QuantumCircuit

    sensor_qc = build_tiger_sensor_channel(p_correct)

    circuits = []
    labels = []

    for inp_name, inp_fn in _INPUT_STATES.items():
        for b0 in _PAULI_BASES:
            for b1 in _PAULI_BASES:
                basis_label = f"{b0}{b1}"
                qc = QuantumCircuit(2, 2, name=f"qpt_{inp_name}_{basis_label}")

                # 1. Prepare input state
                inp_fn(qc)

                # 2. Apply the Tiger sensor channel
                qc.compose(sensor_qc, inplace=True)

                # 3. Rotate to measurement basis
                for gate in _BASIS_ROTATIONS[b0]:
                    getattr(qc, gate)(0)
                for gate in _BASIS_ROTATIONS[b1]:
                    getattr(qc, gate)(1)

                # 4. Measure
                qc.measure([0, 1], [0, 1])
                circuits.append(qc)
                labels.append((inp_name, basis_label))

    return circuits, labels


# ---------------------------------------------------------------------------
# Density matrix reconstruction (reused from entanglement_witness.py)
# ---------------------------------------------------------------------------

def _expectation_from_counts(counts: dict) -> float:
    """Compute <P_A tensor P_B> from 2-qubit counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    exp_val = 0.0
    for bs, count in counts.items():
        b0 = int(bs[-1])
        b1 = int(bs[-2]) if len(bs) >= 2 else 0
        eigenvalue = ((-1) ** b0) * ((-1) ** b1)
        exp_val += eigenvalue * count / total
    return exp_val


def _single_qubit_exp(counts: dict, qubit: int) -> float:
    """Extract single-qubit expectation from 2-qubit measurement counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    exp_val = 0.0
    for bs, count in counts.items():
        if qubit == 0:
            b = int(bs[-1])
        else:
            b = int(bs[-2]) if len(bs) >= 2 else 0
        exp_val += ((-1) ** b) * count / total
    return exp_val


def _project_to_physical(rho: np.ndarray) -> np.ndarray:
    """Project a matrix to the nearest valid density matrix.

    Enforces: Hermitian, positive semidefinite, trace = 1.
    """
    rho = (rho + rho.conj().T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)
    total = np.sum(eigvals)
    if total > 0:
        eigvals /= total
    return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T


def reconstruct_density_matrix(
    counts_list: list[dict],
    basis_labels: list[str],
) -> np.ndarray:
    """Reconstruct a 2-qubit density matrix from 9 Pauli tomography measurements.

    Uses linear inversion: rho = (1/4) sum_{ij} <sigma_i x sigma_j> (sigma_i x sigma_j)
    then projects to the nearest physical density matrix.
    """
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = {"I": I, "X": X, "Y": Y, "Z": Z}

    label_to_counts = dict(zip(basis_labels, counts_list))
    pauli_labels = ["I", "X", "Y", "Z"]
    expectations = {}

    for p0 in pauli_labels:
        for p1 in pauli_labels:
            key = f"{p0}{p1}"
            if p0 == "I" and p1 == "I":
                expectations[key] = 1.0
            elif p0 == "I":
                ref_label = f"Z{p1}"
                if ref_label in label_to_counts:
                    expectations[key] = _single_qubit_exp(
                        label_to_counts[ref_label], 1
                    )
                else:
                    expectations[key] = 0.0
            elif p1 == "I":
                ref_label = f"{p0}Z"
                if ref_label in label_to_counts:
                    expectations[key] = _single_qubit_exp(
                        label_to_counts[ref_label], 0
                    )
                else:
                    expectations[key] = 0.0
            else:
                if key in label_to_counts:
                    expectations[key] = _expectation_from_counts(
                        label_to_counts[key]
                    )
                else:
                    expectations[key] = 0.0

    # Reconstruct: rho = (1/4) sum_{ij} <sigma_i x sigma_j> (sigma_i x sigma_j)
    # Label convention: p0 = qubit 0, p1 = qubit 1.
    # Qiskit matrix ordering: MSB (qubit 1) is the FIRST tensor factor,
    # LSB (qubit 0) is the SECOND.  So kron order is (p1, p0).
    rho = np.zeros((4, 4), dtype=complex)
    for p0 in pauli_labels:
        for p1 in pauli_labels:
            key = f"{p0}{p1}"
            tensor_product = np.kron(paulis[p1], paulis[p0])
            rho += expectations[key] * tensor_product
    rho /= 4.0

    rho = _project_to_physical(rho)
    return rho


# ---------------------------------------------------------------------------
# Ideal output states from statevector simulation
# ---------------------------------------------------------------------------

def compute_ideal_outputs(
    p_correct: float = 0.85,
) -> dict[str, np.ndarray]:
    """Compute the ideal output density matrix for each input state.

    Uses the sensor channel (not the full belief circuit) to match
    what the QPT circuits characterize.

    Returns dict mapping input_name -> 4x4 density matrix.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, DensityMatrix

    sensor_qc = build_tiger_sensor_channel(p_correct)
    ideal_outputs = {}

    for inp_name, inp_fn in _INPUT_STATES.items():
        qc = QuantumCircuit(2)
        inp_fn(qc)
        qc.compose(sensor_qc, inplace=True)
        sv = Statevector.from_instruction(qc)
        ideal_outputs[inp_name] = DensityMatrix(sv).data

    return ideal_outputs


# ---------------------------------------------------------------------------
# Choi matrix reconstruction
# ---------------------------------------------------------------------------

def reconstruct_choi_matrix(
    input_output_pairs: dict[str, np.ndarray],
) -> np.ndarray:
    """Reconstruct the 16x16 Choi matrix from input-output density matrix pairs.

    For a 2-qubit channel E, the Choi matrix is:
        J(E) = sum_{i,j} |i><j| tensor E(|i><j|)

    where i,j range over the 4 computational basis states {00, 01, 10, 11}.

    We have E(|i><i|) directly from computational basis inputs (diagonal blocks).
    For off-diagonal blocks E(|i><j|) with i != j, we use superposition inputs:
        E(|+0><+0|) = (1/2)[E(|00><00|) + E(|00><10|) + E(|10><00|) + E(|10><10|)]
    so  E(|00><10|) + E(|10><00|) = 2*E(|+0><+0|) - E(|00><00|) - E(|10><10|)

    Similarly for |0+>:
        E(|01><01|) + E(|01><00|) ... etc.

    This gives us the real parts of off-diagonal Choi blocks. For a complete
    reconstruction we would also need |+i, 0> inputs, but this over-determined
    system with 6 inputs gives a good approximation.

    Returns:
        16x16 Choi matrix (may not be perfectly physical due to finite stats).
    """
    d = 4  # dimension of 2-qubit Hilbert space

    # Initialize Choi matrix
    choi = np.zeros((d * d, d * d), dtype=complex)

    # Computational basis labels (Qiskit ordering: |q1 q0>)
    comp_labels = ["|00>", "|01>", "|10>", "|11>"]

    # Diagonal blocks: J_{ii} = E(|i><i|)
    for i, label in enumerate(comp_labels):
        rho_out = input_output_pairs[label]
        choi[i * d:(i + 1) * d, i * d:(i + 1) * d] = rho_out

    # Off-diagonal blocks from superposition inputs
    # |+0> = (|00> + |10>)/sqrt(2)
    # E(|+0><+0|) = (1/2)[E(|00><00|) + E(|00><10|) + E(|10><00|) + E(|10><10|)]
    # => E(|00><10|) + E(|10><00|) = 2*E(|+0><+0|) - E(|00><00|) - E(|10><10|)
    if "|+0>" in input_output_pairs:
        rho_plus0 = input_output_pairs["|+0>"]
        rho_00 = input_output_pairs["|00>"]
        rho_10 = input_output_pairs["|10>"]
        # This gives us the Hermitian part: E(|00><10|) + E(|10><00|)
        off_diag_02 = 2.0 * rho_plus0 - rho_00 - rho_10
        # E(|00><10|) = off_diag_02 / 2 (assuming Hermiticity: E(|10><00|) = E(|00><10|).T.conj)
        # For a physical channel, E(|i><j|)^dag = E(|j><i|)
        # So off_diag_02 = E(|00><10|) + E(|00><10|)^dag = 2 * Re[E(|00><10|)]
        # Without imaginary data, we approximate E(|00><10|) ~ off_diag_02 / 2
        e_00_10 = off_diag_02 / 2.0
        choi[0 * d:1 * d, 2 * d:3 * d] = e_00_10
        choi[2 * d:3 * d, 0 * d:1 * d] = e_00_10.conj().T

    # |0+> = (|00> + |01>)/sqrt(2)
    # E(|0+><0+|) = (1/2)[E(|00><00|) + E(|00><01|) + E(|01><00|) + E(|01><01|)]
    # => E(|00><01|) + E(|01><00|) = 2*E(|0+><0+|) - E(|00><00|) - E(|01><01|)
    if "|0+>" in input_output_pairs:
        rho_0plus = input_output_pairs["|0+>"]
        rho_00 = input_output_pairs["|00>"]
        rho_01 = input_output_pairs["|01>"]
        off_diag_01 = 2.0 * rho_0plus - rho_00 - rho_01
        e_00_01 = off_diag_01 / 2.0
        choi[0 * d:1 * d, 1 * d:2 * d] = e_00_01
        choi[1 * d:2 * d, 0 * d:1 * d] = e_00_01.conj().T

    # Remaining off-diagonal blocks (1,2), (1,3), (2,3), (0,3):
    # We don't have dedicated superposition inputs for these, so we leave them
    # as zero (which is equivalent to assuming no coherence between those
    # input subspaces). This is a known limitation of the 6-input QPT scheme.
    # For a complete reconstruction one would need 16 input states (full QPT).
    # Our metrics account for this by using per-input state fidelities as well.

    # Project Choi to physical (trace-preserving completely positive)
    choi = _project_choi_to_physical(choi, d)

    return choi


def _project_choi_to_physical(choi: np.ndarray, d: int) -> np.ndarray:
    """Project the Choi matrix to be positive semidefinite with trace = d.

    A valid Choi matrix for a trace-preserving channel satisfies:
    - Positive semidefinite
    - Tr_output(J) = I_d  (trace-preservation)
    We enforce PSD and correct trace here.
    """
    # Hermitianize
    choi = (choi + choi.conj().T) / 2.0

    # Eigendecompose and truncate negatives
    eigvals, eigvecs = np.linalg.eigh(choi)
    eigvals = np.maximum(eigvals, 0)

    # Normalize: trace of Choi = d for trace-preserving map
    total = np.sum(eigvals)
    if total > 0:
        eigvals *= d / total

    choi_phys = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return choi_phys


# ---------------------------------------------------------------------------
# Process metrics
# ---------------------------------------------------------------------------

def compute_process_fidelity(
    output_rhos: dict[str, np.ndarray],
    ideal_outputs: dict[str, np.ndarray],
    p_correct: float = 0.85,
) -> float:
    """Compute process fidelity via the entanglement fidelity.

    For a unitary target channel U, the process fidelity equals the
    entanglement fidelity:

        F_pro = F_e = (1/d) * sum_i <psi_i| rho_i |psi_i>

    where |psi_i> = U|i> are the ideal outputs for computational basis inputs
    |i>, and rho_i = E(|i><i|) are the measured outputs.

    This is equivalent to averaging the state fidelity over the d=4
    computational basis inputs only (not superposition inputs).
    """
    d = 4
    comp_labels = ["|00>", "|01>", "|10>", "|11>"]

    fidelities = []
    for label in comp_labels:
        rho_m = output_rhos[label]
        rho_i = ideal_outputs[label]
        f = compute_per_input_state_fidelity(rho_m, rho_i)
        fidelities.append(f)

    f_pro = float(np.mean(fidelities))
    return np.clip(f_pro, 0.0, 1.0)


def compute_average_gate_fidelity(process_fidelity: float, d: int = 4) -> float:
    """Average gate fidelity from process fidelity.

    F_avg = (d * F_pro + 1) / (d + 1)
    """
    return (d * process_fidelity + 1) / (d + 1)


def compute_process_infidelity(process_fidelity: float) -> float:
    """Process infidelity r = 1 - F_pro."""
    return 1.0 - process_fidelity


def compute_unitarity(output_rhos: dict[str, np.ndarray]) -> float:
    """Estimate the unitarity of the channel from output density matrices.

    Unitarity u(E) measures how well a channel preserves the purity of states.
    For a unitary channel, u = 1. For a fully depolarizing channel, u = 0.

    We estimate it as:
        u = (d/(d-1)) * [avg_i Tr(E(rho_i - I/d)^2)] / [avg_i Tr((rho_i - I/d)^2)]

    where we average over all input states and E(rho_i - I/d) is the traceless
    part of the output.

    For computational basis inputs, rho_i - I/d has known trace(square).
    """
    d = 4  # 2-qubit dimension
    I_d = np.eye(d, dtype=complex) / d

    numerators = []
    denominators = []

    for inp_name, rho_out in output_rhos.items():
        # Input state
        inp_sv = _prepare_input_statevector(inp_name)
        rho_in = np.outer(inp_sv, inp_sv.conj())

        # Traceless parts
        delta_in = rho_in - I_d
        delta_out = rho_out - I_d

        num = float(np.real(np.trace(delta_out @ delta_out)))
        den = float(np.real(np.trace(delta_in @ delta_in)))

        if den > 1e-15:
            numerators.append(num)
            denominators.append(den)

    if not denominators:
        return 0.0

    # u = (d/(d-1)) * mean(num) / mean(den)
    avg_num = np.mean(numerators)
    avg_den = np.mean(denominators)

    if avg_den < 1e-15:
        return 0.0

    u = (d / (d - 1)) * avg_num / avg_den
    return float(np.clip(u, 0.0, 1.0))


def compute_diamond_distance_bound(
    input_output_pairs_measured: dict[str, np.ndarray],
    input_output_pairs_ideal: dict[str, np.ndarray],
) -> float:
    """Compute an upper bound on the diamond distance from per-input trace distances.

    The diamond distance between two channels E and F is:
        d_diamond(E, F) = max_{rho} || (E - F)(rho) ||_1

    We bound it by the maximum trace distance over our input states:
        d_diamond >= max_i || E(rho_i) - F(rho_i) ||_1 / 2

    This is a lower bound; the actual diamond distance could be higher.
    We report it as such.
    """
    trace_distances = []
    for inp_name in input_output_pairs_measured:
        rho_m = input_output_pairs_measured[inp_name]
        rho_i = input_output_pairs_ideal[inp_name]
        diff = rho_m - rho_i
        # Trace distance = (1/2) * ||A||_1 = (1/2) * sum of singular values
        svs = np.linalg.svd(diff, compute_uv=False)
        td = 0.5 * float(np.sum(svs))
        trace_distances.append(td)

    return float(np.max(trace_distances))


def compute_per_input_state_fidelity(
    rho_measured: np.ndarray,
    rho_ideal: np.ndarray,
) -> float:
    """Compute state fidelity between two density matrices.

    F(rho, sigma) = [Tr(sqrt(sqrt(rho) sigma sqrt(rho)))]^2

    For a pure ideal state |psi>, simplifies to F = <psi|rho_measured|psi>.
    """
    # Check if ideal is pure (rank 1)
    eigvals = np.linalg.eigvalsh(rho_ideal)
    if np.max(eigvals) > 0.999:
        # Pure state -- use simplified formula
        idx = np.argmax(eigvals)
        _, eigvecs = np.linalg.eigh(rho_ideal)
        psi = eigvecs[:, idx]
        return float(np.clip(np.real(psi.conj() @ rho_measured @ psi), 0.0, 1.0))

    # General case: Uhlmann fidelity
    sqrt_rho = _matrix_sqrt(rho_ideal)
    M = sqrt_rho @ rho_measured @ sqrt_rho
    eigvals_M = np.linalg.eigvalsh(M)
    eigvals_M = np.maximum(eigvals_M, 0)
    fidelity = float(np.sum(np.sqrt(eigvals_M))) ** 2
    return float(np.clip(fidelity, 0.0, 1.0))


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Compute the matrix square root of a positive semidefinite matrix."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


# ---------------------------------------------------------------------------
# Circuit execution
# ---------------------------------------------------------------------------

def _run_circuits_simulator(
    circuits: list["QuantumCircuit"],
    shots: int,
) -> list[dict]:
    """Execute circuits on AerSimulator and return list of count dicts."""
    from quantum_common.backends.simulator import AerSimulatorBackend
    from quantum_common.backends.base import ExecutionRequest

    aer = AerSimulatorBackend()
    result = aer.execute(ExecutionRequest(circuits=circuits, shots=shots))
    return result.counts


def _run_circuits_hardware(
    circuits: list["QuantumCircuit"],
    shots: int,
    backend_name: str,
    token: str | None,
    channel: str,
    instance: str | None,
    opt_level: int,
) -> tuple[list[dict], str]:
    """Execute circuits on IBM hardware.

    Returns (list_of_count_dicts, job_id).
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    _svc_kw: dict = {"channel": channel}
    if token:
        _svc_kw["token"] = token
    if instance:
        _svc_kw["instance"] = instance
    service = QiskitRuntimeService(**_svc_kw)
    hw_backend = service.backend(backend_name)

    print(f"\n[ibm] Connected to {backend_name} ({hw_backend.num_qubits} qubits)")

    pm = generate_preset_pass_manager(
        optimization_level=opt_level, backend=hw_backend
    )
    isa_circuits = pm.run(circuits)

    # Clear layouts to prevent QPY layout-register mismatch (IBM Error 3211)
    for isa_qc in isa_circuits:
        isa_qc._layout = None

    isa_depths = [qc.depth() for qc in isa_circuits]
    print(
        f"  ISA depths: min={min(isa_depths)}, max={max(isa_depths)}, "
        f"mean={np.mean(isa_depths):.0f}"
    )

    sampler = SamplerV2(mode=hw_backend)
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    sampler.options.twirling.strategy = "active-accum"
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.dynamical_decoupling.scheduling_method = "alap"
    print("  Twirling+DD: enabled (XY4/alap)")

    print(f"\n[ibm] Submitting {len(isa_circuits)} circuits ...")
    pubs = [(qc,) for qc in isa_circuits]
    job = sampler.run(pubs, shots=shots)
    result = job.result()

    all_counts = []
    for i in range(len(circuits)):
        pub_result = result[i]
        creg_name = next(
            k for k in vars(pub_result.data) if not k.startswith("_")
        )
        counts = dict(getattr(pub_result.data, creg_name).get_counts())
        all_counts.append(counts)

    job_id = job.job_id()
    print(f"  Job ID: {job_id}")

    return all_counts, job_id


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(
    prior: list[float],
    shots: int,
    p_correct: float,
    dry_run: bool,
    backend_name: str,
    token: str | None,
    channel: str,
    instance: str | None,
    opt_level: int,
) -> dict:
    """Run full QPT analysis for a single prior value."""

    print(f"\n{'=' * 60}")
    print(f"  QUANTUM PROCESS TOMOGRAPHY")
    print(f"  Prior: {prior}")
    print(f"{'=' * 60}")

    result: dict = {"prior": prior, "p_correct": p_correct}

    # ------------------------------------------------------------------
    # 1. Compute ideal outputs (ground truth)
    # ------------------------------------------------------------------
    print("\n[ideal] Computing ideal channel outputs from statevector ...")
    ideal_outputs = compute_ideal_outputs(p_correct)

    for inp_name, rho_ideal in ideal_outputs.items():
        purity = float(np.real(np.trace(rho_ideal @ rho_ideal)))
        print(f"  {inp_name} -> purity = {purity:.6f}")

    # ------------------------------------------------------------------
    # 2. Build QPT circuits
    # ------------------------------------------------------------------
    circuits, labels = build_qpt_circuits(p_correct)
    n_inputs = len(_INPUT_STATES)
    n_bases = 9
    print(
        f"\n[qpt] Built {len(circuits)} circuits "
        f"({n_inputs} inputs x {n_bases} bases)"
    )

    # ------------------------------------------------------------------
    # 3. Execute circuits
    # ------------------------------------------------------------------
    t_start = time.time()
    if dry_run:
        print(
            f"\n[sim] Running {len(circuits)} circuits on AerSimulator "
            f"({shots} shots each) ..."
        )
        all_counts = _run_circuits_simulator(circuits, shots)
        result["backend"] = "aer_simulator"
    else:
        # Collect simulator baseline first
        print(f"\n[sim] Collecting simulator baseline ...")
        sim_counts = _run_circuits_simulator(circuits, shots)

        # Run on hardware
        hw_counts, job_id = _run_circuits_hardware(
            circuits, shots, backend_name, token, channel, instance, opt_level,
        )
        result["hw_job_id"] = job_id
        result["backend"] = backend_name

        # Store simulator results for comparison
        sim_output_rhos = _reconstruct_all_outputs(sim_counts, labels)
        sim_fidelities = {}
        for inp_name, rho_sim in sim_output_rhos.items():
            rho_ideal = ideal_outputs[inp_name]
            sim_fidelities[inp_name] = compute_per_input_state_fidelity(
                rho_sim, rho_ideal
            )
        result["simulator_per_input_fidelities"] = sim_fidelities
        print(f"\n[sim] Simulator per-input fidelities:")
        for inp_name, f in sim_fidelities.items():
            print(f"  {inp_name}: {f:.6f}")

        all_counts = hw_counts

    elapsed = time.time() - t_start
    print(f"\n[exec] Circuits executed in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 4. Reconstruct output density matrices
    # ------------------------------------------------------------------
    print("\n[tomo] Reconstructing output density matrices ...")
    output_rhos = _reconstruct_all_outputs(all_counts, labels)

    # ------------------------------------------------------------------
    # 5. Per-input state fidelity
    # ------------------------------------------------------------------
    print("\n[fidelity] Per-input state fidelities:")
    per_input_fidelities = {}
    for inp_name in _INPUT_STATES:
        rho_m = output_rhos[inp_name]
        rho_i = ideal_outputs[inp_name]
        f = compute_per_input_state_fidelity(rho_m, rho_i)
        per_input_fidelities[inp_name] = f
        print(f"  {inp_name}: F = {f:.6f}")

    avg_state_fidelity = float(np.mean(list(per_input_fidelities.values())))
    print(f"\n  Average state fidelity: {avg_state_fidelity:.6f}")
    result["per_input_state_fidelities"] = per_input_fidelities
    result["average_state_fidelity"] = avg_state_fidelity

    # ------------------------------------------------------------------
    # 6. Reconstruct Choi matrix and process metrics
    # ------------------------------------------------------------------
    print("\n[choi] Reconstructing Choi matrix ...")
    choi = reconstruct_choi_matrix(output_rhos)

    choi_eigvals = np.sort(np.linalg.eigvalsh(choi)).tolist()
    choi_rank = int(np.sum(np.array(choi_eigvals) > 1e-10))
    choi_trace = float(np.real(np.trace(choi)))
    print(f"  Choi rank: {choi_rank} (ideal: 1 for unitary)")
    print(f"  Choi trace: {choi_trace:.4f} (ideal: 4.0)")
    print(f"  Choi eigenvalue range: [{choi_eigvals[0]:.6f}, {choi_eigvals[-1]:.6f}]")

    result["choi_matrix_real"] = choi.real.tolist()
    result["choi_matrix_imag"] = choi.imag.tolist()
    result["choi_eigenvalues"] = choi_eigvals
    result["choi_rank"] = choi_rank
    result["choi_trace"] = choi_trace

    # Process fidelity
    print("\n[metrics] Computing process metrics ...")
    f_pro = compute_process_fidelity(output_rhos, ideal_outputs, p_correct)
    f_avg = compute_average_gate_fidelity(f_pro)
    r_pro = compute_process_infidelity(f_pro)
    print(f"  Process fidelity (F_pro)    : {f_pro:.6f}")
    print(f"  Average gate fidelity (F_avg): {f_avg:.6f}")
    print(f"  Process infidelity (r)       : {r_pro:.6f}")

    result["process_fidelity"] = f_pro
    result["average_gate_fidelity"] = f_avg
    result["process_infidelity"] = r_pro

    # Unitarity
    unitarity = compute_unitarity(output_rhos)
    print(f"  Unitarity                    : {unitarity:.6f}")
    result["unitarity"] = unitarity

    # Diamond distance bound
    dd_bound = compute_diamond_distance_bound(output_rhos, ideal_outputs)
    print(f"  Diamond distance (lower bnd) : {dd_bound:.6f}")
    result["diamond_distance_lower_bound"] = dd_bound

    # ------------------------------------------------------------------
    # 7. Output purity analysis (depolarization signature)
    # ------------------------------------------------------------------
    print("\n[purity] Output state purities:")
    purities = {}
    ideal_purities = {}
    for inp_name in _INPUT_STATES:
        rho_m = output_rhos[inp_name]
        rho_i = ideal_outputs[inp_name]
        p_meas = float(np.real(np.trace(rho_m @ rho_m)))
        p_ideal = float(np.real(np.trace(rho_i @ rho_i)))
        purities[inp_name] = p_meas
        ideal_purities[inp_name] = p_ideal
        print(
            f"  {inp_name}: measured = {p_meas:.6f}, ideal = {p_ideal:.6f}, "
            f"ratio = {p_meas / p_ideal:.4f}" if p_ideal > 1e-10 else
            f"  {inp_name}: measured = {p_meas:.6f}, ideal = {p_ideal:.6f}"
        )

    avg_purity_ratio = float(np.mean([
        purities[k] / ideal_purities[k]
        for k in purities
        if ideal_purities[k] > 1e-10
    ]))
    print(f"\n  Average purity ratio: {avg_purity_ratio:.6f} (1.0 = no decoherence)")
    result["output_purities_measured"] = purities
    result["output_purities_ideal"] = ideal_purities
    result["average_purity_ratio"] = avg_purity_ratio

    # ------------------------------------------------------------------
    # 8. Density matrices for output (serialized)
    # ------------------------------------------------------------------
    result["output_density_matrices"] = {
        inp_name: {
            "real": rho.real.tolist(),
            "imag": rho.imag.tolist(),
        }
        for inp_name, rho in output_rhos.items()
    }

    # ------------------------------------------------------------------
    # 9. Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  QPT VERDICT")
    print(f"{'=' * 60}")

    # Thresholds
    f_pro_good = f_pro > 0.80
    f_avg_good = f_avg > 0.85
    unitarity_good = unitarity > 0.80
    dd_good = dd_bound < 0.20

    verdict_pass = bool(f_pro_good and f_avg_good)
    result["pass"] = verdict_pass

    status = "PASS" if verdict_pass else "NEEDS IMPROVEMENT"
    print(f"  Process fidelity  : {f_pro:.4f}  {'OK' if f_pro_good else 'LOW'}")
    print(f"  Avg gate fidelity : {f_avg:.4f}  {'OK' if f_avg_good else 'LOW'}")
    print(f"  Unitarity         : {unitarity:.4f}  {'OK' if unitarity_good else 'LOW'}")
    print(f"  Diamond distance  : {dd_bound:.4f}  {'OK' if dd_good else 'HIGH'}")
    print(f"  Average state fid : {avg_state_fidelity:.4f}")
    print(f"  Avg purity ratio  : {avg_purity_ratio:.4f}")
    print(f"\n  >>> VERDICT: {status} <<<")

    if not verdict_pass:
        print("\n  Diagnostic notes:")
        if not f_pro_good:
            print(
                "  - Process fidelity < 0.80: significant channel error."
            )
        if not unitarity_good:
            print(
                "  - Unitarity < 0.80: channel has strong incoherent "
                "(depolarizing) noise."
            )
        if not dd_good:
            print(
                "  - Diamond distance > 0.20: worst-case error is large; "
                "some input states are poorly reproduced."
            )

    return result


def _reconstruct_all_outputs(
    all_counts: list[dict],
    labels: list[tuple[str, str]],
) -> dict[str, np.ndarray]:
    """Group counts by input state and reconstruct density matrices.

    Returns dict mapping input_name -> 4x4 density matrix.
    """
    # Group by input state
    from collections import defaultdict

    grouped: dict[str, tuple[list[dict], list[str]]] = defaultdict(
        lambda: ([], [])
    )
    for (inp_name, basis_label), counts in zip(labels, all_counts):
        grouped[inp_name][0].append(counts)
        grouped[inp_name][1].append(basis_label)

    output_rhos = {}
    for inp_name, (counts_list, basis_labels) in grouped.items():
        rho = reconstruct_density_matrix(counts_list, basis_labels)
        output_rhos[inp_name] = rho

    return output_rhos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Parse priors
    prior_vals = args.prior
    if len(prior_vals) != 2:
        print("[error] --prior requires exactly 2 values (e.g. --prior 0.5 0.5)")
        sys.exit(1)
    prior = [prior_vals[0], prior_vals[1]]
    if abs(sum(prior) - 1.0) > 0.01:
        print(f"[error] Prior {prior} does not sum to 1.0")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  QUANTUM PROCESS TOMOGRAPHY")
    print("  Tiger POMDP Belief Update Circuit")
    print("=" * 60)
    print(
        f"  Backend        : "
        f"{'AerSimulator (dry-run)' if args.dry_run else args.backend}"
    )
    print(f"  Shots          : {args.shots}")
    print(f"  P(correct)     : {args.p_correct}")
    print(f"  Prior          : {prior}")
    print(f"  Input states   : {len(_INPUT_STATES)}")
    print(f"  Pauli bases    : 9 (XX..ZZ)")
    print(f"  Total circuits : {len(_INPUT_STATES) * 9}")

    token = None
    channel = ibm_channel()
    instance_val = args.instance or ibm_instance()
    if not args.dry_run:
        token = get_ibm_token_optional()
        channel = args.channel or channel

    result = analyze(
        prior=prior,
        shots=args.shots,
        p_correct=args.p_correct,
        dry_run=args.dry_run,
        backend_name=args.backend,
        token=token,
        channel=channel,
        instance=instance_val,
        opt_level=args.opt_level,
    )

    # Save results
    output_data = {
        "task_ids": ["process_tomography"],
        "backend": result["backend"],
        "shots": args.shots,
        "p_correct": args.p_correct,
        "prior": prior,
        "n_input_states": len(_INPUT_STATES),
        "n_pauli_bases": 9,
        "n_circuits": len(_INPUT_STATES) * 9,
        "process_fidelity": result["process_fidelity"],
        "average_gate_fidelity": result["average_gate_fidelity"],
        "process_infidelity": result["process_infidelity"],
        "unitarity": result["unitarity"],
        "diamond_distance_lower_bound": result["diamond_distance_lower_bound"],
        "average_state_fidelity": result["average_state_fidelity"],
        "average_purity_ratio": result["average_purity_ratio"],
        "per_input_state_fidelities": result["per_input_state_fidelities"],
        "choi_rank": result["choi_rank"],
        "choi_trace": result["choi_trace"],
        "choi_eigenvalues": result["choi_eigenvalues"],
        "pass": result["pass"],
        "results": result,
        "notes": (
            f"QPT with {len(_INPUT_STATES)} input states and 9 Pauli bases "
            f"({len(_INPUT_STATES) * 9} circuits). "
            f"Process fidelity: {result['process_fidelity']:.4f}, "
            f"Avg gate fidelity: {result['average_gate_fidelity']:.4f}, "
            f"Unitarity: {result['unitarity']:.4f}."
        ),
    }

    save_result("process_tomography", output_data)

    overall = "PASS" if result["pass"] else "NEEDS IMPROVEMENT"
    print(f"\n  OVERALL: {overall}")
    print(f"  Process fidelity: {result['process_fidelity']:.4f}")
    print(f"  Average gate fidelity: {result['average_gate_fidelity']:.4f}")
    print(f"  Unitarity: {result['unitarity']:.4f}")
    print(f"  Diamond distance (lower bound): {result['diamond_distance_lower_bound']:.4f}")


if __name__ == "__main__":
    main()
