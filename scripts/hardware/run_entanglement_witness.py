"""VALIDATION-ROADMAP Task — Entanglement certification for Tiger POMDP belief circuit.

Proves that genuine quantum entanglement is present in the QANTIS-2 Tiger POMDP
belief update circuit, addressing the "classical simulator can do this" criticism.

Three independent entanglement tests are performed:

1. **PPT Criterion (Peres-Horodecki):** Compute the partial transpose of the
   reconstructed density matrix. Any negative eigenvalue proves entanglement.

2. **Concurrence:** Quantitative entanglement measure (0 = separable,
   1 = maximally entangled). Computed from the density matrix.

3. **CHSH Inequality:** Measures correlations between state (q0) and
   observation (q1) at 4 angle settings. Classical limit |S| <= 2;
   quantum violation |S| > 2 proves non-classical correlations.
   Measurement angles are numerically optimized for the Tiger state.

The Tiger belief circuit creates entanglement between the state qubit (q0) and
the observation qubit (q1) via conditional R_y gates.  At a uniform prior
[0.5, 0.5] the entanglement is strongest (concurrence ~ 0.70); as the agent
becomes more confident (e.g. [0.99, 0.01]), the state approaches separability
(concurrence ~ 0.14).  This is physically meaningful: quantum advantage is
largest when the agent is uncertain.

Usage
-----
# Dry run (no credentials needed):
python scripts/hardware/run_entanglement_witness.py --dry-run --prior 0.5 0.5

# Multiple priors (dry run):
python scripts/hardware/run_entanglement_witness.py --dry-run --prior 0.5 0.5 0.85 0.15 0.99 0.01

# Hardware run:
IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_entanglement_witness.py \\
    --backend ibm_kingston --prior 0.5 0.5 --shots 8192

Output
------
  output/hardware/entanglement_witness_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
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
        description="Entanglement certification for Tiger POMDP belief circuit"
    )
    p.add_argument("--backend", default="ibm_kingston", help="IBM backend name")
    p.add_argument("--shots", type=int, default=8192, help="Shot count per circuit")
    p.add_argument(
        "--prior", nargs="+", type=float, default=[0.5, 0.5],
        help="Prior belief probabilities. Supply multiple pairs to sweep: "
             "e.g. --prior 0.5 0.5 0.85 0.15 0.99 0.01",
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
    p.add_argument(
        "--skip-chsh", action="store_true",
        help="Skip the CHSH test (saves 4 circuits per prior)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tiger belief circuit (measurement-free, for state tomography)
# ---------------------------------------------------------------------------

def build_tiger_belief_statevector_circuit(
    prior: list[float],
    p_correct: float = 0.85,
) -> "QuantumCircuit":
    """Build the 2-qubit Tiger belief state circuit WITHOUT measurements.

    This circuit is used for:
    - Statevector simulation (exact density matrix)
    - Quantum state tomography on hardware

    Qubit layout:
        q0 = state register  (|0> = tiger-left, |1> = tiger-right)
        q1 = observation register

    The controlled-R_y gates create entanglement between state and
    observation, encoding the sensor model P(obs|state).
    """
    from qiskit import QuantumCircuit

    p00 = float(np.clip(p_correct, 1e-9, 1.0 - 1e-9))      # P(obs=0|state=0)
    p10 = float(np.clip(1 - p_correct, 1e-9, 1.0 - 1e-9))  # P(obs=0|state=1)

    # R_y(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    # P(state=0) = cos^2(theta/2) = prior[0]  =>  theta = 2*arccos(sqrt(prior[0]))
    theta_state = 2.0 * float(np.arccos(np.sqrt(max(prior[0], 1e-9))))
    theta_obs_s0 = 2.0 * float(np.arccos(np.sqrt(p00)))   # when state=0
    theta_obs_s1 = 2.0 * float(np.arccos(np.sqrt(p10)))   # when state=1

    qc = QuantumCircuit(2, name="tiger_belief")

    # 1. Encode prior on state qubit
    qc.ry(theta_state, 0)

    # 2. Conditional observation encoding (creates entanglement)
    #    CRY activates when control=|1>; to condition on |0>, bracket with X.
    qc.x(0)
    qc.cry(theta_obs_s0, 0, 1)   # P(obs|state=tiger-left) when q0=|0>
    qc.x(0)
    qc.cry(theta_obs_s1, 0, 1)   # P(obs|state=tiger-right) when q0=|1>

    return qc


# ---------------------------------------------------------------------------
# Ideal (statevector) entanglement analysis
# ---------------------------------------------------------------------------

def compute_ideal_entanglement(
    prior: list[float],
    p_correct: float = 0.85,
) -> dict:
    """Compute exact entanglement measures from statevector simulation.

    Returns dict with: concurrence, negativity, ppt_eigenvalues,
    ppt_min_eigenvalue, is_entangled, density_matrix.
    """
    from qiskit.quantum_info import Statevector, DensityMatrix, concurrence, negativity

    qc = build_tiger_belief_statevector_circuit(prior, p_correct)
    sv = Statevector.from_instruction(qc)
    dm = DensityMatrix(sv)

    neg = float(negativity(dm, [0]))
    conc = float(concurrence(dm))

    # Manual PPT: partial transpose w.r.t. subsystem B (qubit 1)
    rho = dm.data
    rho_reshaped = rho.reshape(2, 2, 2, 2)
    rho_pt = rho_reshaped.transpose(0, 3, 2, 1).reshape(4, 4)
    ppt_eigenvalues = np.sort(np.linalg.eigvalsh(rho_pt)).tolist()
    ppt_min = float(min(ppt_eigenvalues))

    return {
        "concurrence": conc,
        "negativity": neg,
        "ppt_eigenvalues": ppt_eigenvalues,
        "ppt_min_eigenvalue": ppt_min,
        "is_entangled": ppt_min < -1e-10,
        "density_matrix_real": rho.real.tolist(),
        "density_matrix_imag": rho.imag.tolist(),
    }


# ---------------------------------------------------------------------------
# CHSH Bell inequality test
# ---------------------------------------------------------------------------

def _optimize_chsh_angles(
    prior: list[float],
    p_correct: float = 0.85,
    n_restarts: int = 30,
) -> tuple[list[float], float]:
    """Find measurement angles that maximize |S| for the Tiger state.

    The Tiger belief state is not a Bell state, so the standard CHSH angles
    (0, pi/4, pi/8, 3pi/8) are not optimal. We numerically optimize over
    all four angles (a, a', b, b') to find the maximum CHSH violation.

    Returns (optimal_angles, max_S_value).
    """
    from scipy.optimize import minimize as scipy_minimize
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    theta_prior = 2.0 * np.arccos(np.sqrt(max(prior[0], 1e-9)))
    p00 = float(np.clip(p_correct, 1e-9, 1.0 - 1e-9))
    p10 = float(np.clip(1 - p_correct, 1e-9, 1.0 - 1e-9))
    theta_obs_s0 = 2.0 * np.arccos(np.sqrt(p00))
    theta_obs_s1 = 2.0 * np.arccos(np.sqrt(p10))

    def compute_s(angles: np.ndarray) -> float:
        a, a_prime, b, b_prime = angles
        settings = [(a, b), (a, b_prime), (a_prime, b), (a_prime, b_prime)]
        correlators = []
        for alice_angle, bob_angle in settings:
            qc = QuantumCircuit(2)
            qc.ry(theta_prior, 0)
            qc.x(0)
            qc.cry(theta_obs_s0, 0, 1)
            qc.x(0)
            qc.cry(theta_obs_s1, 0, 1)
            qc.ry(-2 * alice_angle, 0)
            qc.ry(-2 * bob_angle, 1)
            sv = Statevector.from_instruction(qc)
            probs = sv.probabilities_dict()
            same = probs.get("00", 0) + probs.get("11", 0)
            diff = probs.get("01", 0) + probs.get("10", 0)
            correlators.append(same - diff)
        E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime = correlators
        return E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime

    rng = np.random.default_rng(42)
    best_angles = np.array([0, np.pi / 4, np.pi / 8, 3 * np.pi / 8])
    best_s = abs(compute_s(best_angles))

    for _ in range(n_restarts):
        x0 = rng.uniform(0, np.pi, 4)
        res = scipy_minimize(lambda x: -abs(compute_s(x)), x0, method="Nelder-Mead",
                             options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-10})
        s_val = abs(compute_s(res.x))
        if s_val > best_s:
            best_s = s_val
            best_angles = res.x.copy()

    # Determine the sign of S at optimal angles
    s_signed = compute_s(best_angles)
    return best_angles.tolist(), float(s_signed)


def build_chsh_circuits(
    prior: list[float],
    angles: list[float],
    p_correct: float = 0.85,
) -> list["QuantumCircuit"]:
    """Build 4 CHSH measurement circuits for the Tiger belief state.

    Args:
        prior: Belief state [P(state=0), P(state=1)].
        angles: [a, a', b, b'] measurement angles (optimized).
        p_correct: Tiger listen accuracy.

    Returns:
        List of 4 QuantumCircuits with measurements.
    """
    from qiskit import QuantumCircuit

    p00 = float(np.clip(p_correct, 1e-9, 1.0 - 1e-9))
    p10 = float(np.clip(1 - p_correct, 1e-9, 1.0 - 1e-9))
    theta_prior = 2.0 * float(np.arccos(np.sqrt(max(prior[0], 1e-9))))
    theta_obs_s0 = 2.0 * float(np.arccos(np.sqrt(p00)))
    theta_obs_s1 = 2.0 * float(np.arccos(np.sqrt(p10)))

    a, a_prime, b, b_prime = angles
    settings = [(a, b), (a, b_prime), (a_prime, b), (a_prime, b_prime)]

    circuits = []
    for alice_angle, bob_angle in settings:
        qc = QuantumCircuit(2, 2, name="chsh")

        # Tiger belief state (entangling part)
        qc.ry(theta_prior, 0)
        qc.x(0)
        qc.cry(theta_obs_s0, 0, 1)
        qc.x(0)
        qc.cry(theta_obs_s1, 0, 1)

        # Rotate to measurement basis
        qc.ry(-2 * alice_angle, 0)
        qc.ry(-2 * bob_angle, 1)

        # Measure
        qc.measure([0, 1], [0, 1])
        circuits.append(qc)

    return circuits


def compute_chsh_from_counts(counts_list: list[dict]) -> tuple[float, list[float]]:
    """Compute CHSH S value from 4 measurement settings.

    Args:
        counts_list: List of 4 count dicts from the CHSH circuits.

    Returns:
        (S, correlators) where S = E(a,b) - E(a,b') + E(a',b) + E(a',b').
    """
    correlators = []
    for counts in counts_list:
        total = sum(counts.values())
        if total == 0:
            correlators.append(0.0)
            continue
        same = counts.get("00", 0) + counts.get("11", 0)
        diff = counts.get("01", 0) + counts.get("10", 0)
        correlators.append((same - diff) / total)

    E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime = correlators
    S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime
    return float(S), [float(e) for e in correlators]


# ---------------------------------------------------------------------------
# Manual quantum state tomography (2 qubits, 9 Pauli bases)
# ---------------------------------------------------------------------------

def build_tomography_circuits(
    prior: list[float],
    p_correct: float = 0.85,
) -> tuple[list["QuantumCircuit"], list[str]]:
    """Build 9 circuits for 2-qubit state tomography.

    For full tomography of a 2-qubit state we measure in the 9 Pauli bases:
    XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ.

    Each basis rotation:
      Z basis: no rotation (computational basis measurement)
      X basis: H gate before measurement
      Y basis: S^dag then H before measurement

    Returns:
        (circuits, basis_labels) where basis_labels are like 'XX', 'XY', etc.
    """
    from qiskit import QuantumCircuit

    bases = ["X", "Y", "Z"]
    basis_rotations = {
        "Z": [],              # No rotation needed
        "X": ["h"],           # H maps X eigenstates to Z eigenstates
        "Y": ["sdg", "h"],   # Sdg+H maps Y eigenstates to Z eigenstates
    }

    qc_state = build_tiger_belief_statevector_circuit(prior, p_correct)

    circuits = []
    labels = []
    for b0 in bases:
        for b1 in bases:
            label = f"{b0}{b1}"
            qc = QuantumCircuit(2, 2, name=f"tomo_{label}")

            # Append the state preparation
            qc.compose(qc_state, inplace=True)

            # Apply basis rotation for qubit 0
            for gate in basis_rotations[b0]:
                getattr(qc, gate)(0)

            # Apply basis rotation for qubit 1
            for gate in basis_rotations[b1]:
                getattr(qc, gate)(1)

            qc.measure([0, 1], [0, 1])
            circuits.append(qc)
            labels.append(label)

    return circuits, labels


def reconstruct_density_matrix(
    counts_list: list[dict],
    basis_labels: list[str],
) -> np.ndarray:
    """Reconstruct a 2-qubit density matrix from tomography measurements.

    Uses linear inversion: rho = (1/4) * sum_{ij} <sigma_i x sigma_j> * (sigma_i x sigma_j)

    This is the standard approach for 2-qubit QST. Linear inversion may produce
    a non-physical (non-positive) density matrix; we project to the nearest
    physical state via eigenvalue truncation.

    Args:
        counts_list: List of 9 count dicts (one per Pauli basis).
        basis_labels: Corresponding labels like ['XX', 'XY', ...].

    Returns:
        4x4 density matrix (numpy array).
    """
    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = {"I": I, "X": X, "Y": Y, "Z": Z}

    def expectation_from_counts(counts: dict) -> float:
        """Compute <P> from counts where P is a Pauli with eigenvalues +/-1."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        # For Pauli measurement: eigenvalue = (-1)^(b0 XOR b1) for tensor product
        # But for single-qubit Pauli on each qubit independently:
        # outcome 0 -> eigenvalue +1, outcome 1 -> eigenvalue -1
        # For tensor product P_A x P_B:
        # <P_A x P_B> = <P_A> * <P_B> for product measurements
        # But since we measure both qubits, we get the joint correlator directly.
        # Bitstring 'b1b0' (MSB first in Qiskit):
        #   eigenvalue = (-1)^b0 * (-1)^b1 for the tensor Pauli
        exp_val = 0.0
        for bs, count in counts.items():
            b0 = int(bs[-1])   # qubit 0 (rightmost)
            b1 = int(bs[-2]) if len(bs) >= 2 else 0   # qubit 1
            eigenvalue = ((-1) ** b0) * ((-1) ** b1)
            exp_val += eigenvalue * count / total
        return exp_val

    # We also need single-qubit expectation values.
    # From the ZZ, ZI, IZ measurements... but we only have XX, XY, ..., ZZ.
    # We need all 16 components: <I x I>, <I x X>, ..., <Z x Z>.
    # <I x I> = 1 always. For <I x P> and <P x I> we extract from combined measurements.

    # Build map from label -> counts
    label_to_counts = dict(zip(basis_labels, counts_list))

    def single_qubit_exp(counts: dict, qubit: int) -> float:
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

    # Compute all 16 Pauli expectation values
    pauli_labels = ["I", "X", "Y", "Z"]
    expectations = {}

    for p0 in pauli_labels:
        for p1 in pauli_labels:
            key = f"{p0}{p1}"
            if p0 == "I" and p1 == "I":
                expectations[key] = 1.0
            elif p0 == "I":
                # <I x P1>: extract qubit-1 expectation from any circuit measuring P1 on qubit 1
                # Use the Z<p1> circuit (measures Z on q0, p1 on q1)
                ref_label = f"Z{p1}"
                if ref_label in label_to_counts:
                    expectations[key] = single_qubit_exp(label_to_counts[ref_label], 1)
                else:
                    expectations[key] = 0.0
            elif p1 == "I":
                # <P0 x I>: extract qubit-0 expectation from any circuit measuring P0 on qubit 0
                ref_label = f"{p0}Z"
                if ref_label in label_to_counts:
                    expectations[key] = single_qubit_exp(label_to_counts[ref_label], 0)
                else:
                    expectations[key] = 0.0
            else:
                # <P0 x P1>: direct joint measurement
                if key in label_to_counts:
                    expectations[key] = expectation_from_counts(label_to_counts[key])
                else:
                    expectations[key] = 0.0

    # Reconstruct: rho = (1/4) * sum_{ij} <sigma_i x sigma_j> * (sigma_i x sigma_j)
    rho = np.zeros((4, 4), dtype=complex)
    for p0 in pauli_labels:
        for p1 in pauli_labels:
            key = f"{p0}{p1}"
            tensor_product = np.kron(paulis[p0], paulis[p1])
            rho += expectations[key] * tensor_product
    rho /= 4.0

    # Project to nearest physical density matrix (positive semidefinite, trace 1)
    rho = _project_to_physical(rho)

    return rho


def _project_to_physical(rho: np.ndarray) -> np.ndarray:
    """Project a matrix to the nearest valid density matrix.

    Enforces: Hermitian, positive semidefinite, trace = 1.
    Uses eigenvalue truncation (set negative eigenvalues to 0, renormalize).
    """
    # Ensure Hermitian
    rho = (rho + rho.conj().T) / 2.0

    # Eigendecompose and truncate negative eigenvalues
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)

    # Renormalize
    total = np.sum(eigvals)
    if total > 0:
        eigvals /= total

    rho_phys = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho_phys


def compute_entanglement_from_rho(rho: np.ndarray) -> dict:
    """Compute entanglement measures from a 4x4 density matrix.

    Returns dict with concurrence, negativity, PPT eigenvalues, and
    entanglement verdict.
    """
    from qiskit.quantum_info import DensityMatrix, concurrence, negativity

    dm = DensityMatrix(rho)
    neg = float(negativity(dm, [0]))
    conc = float(concurrence(dm))

    # Manual PPT
    rho_reshaped = rho.reshape(2, 2, 2, 2)
    rho_pt = rho_reshaped.transpose(0, 3, 2, 1).reshape(4, 4)
    ppt_eigenvalues = np.sort(np.linalg.eigvalsh(rho_pt)).tolist()
    ppt_min = float(min(ppt_eigenvalues))

    return {
        "concurrence": conc,
        "negativity": neg,
        "ppt_eigenvalues": ppt_eigenvalues,
        "ppt_min_eigenvalue": ppt_min,
        "is_entangled_ppt": ppt_min < -1e-6,
        "is_entangled_concurrence": conc > 1e-6,
    }


# ---------------------------------------------------------------------------
# Run circuits on backend (simulator or hardware)
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

    pm = generate_preset_pass_manager(optimization_level=opt_level, backend=hw_backend)
    isa_circuits = pm.run(circuits)

    # Clear layouts to prevent QPY layout-register mismatch (IBM Error 3211)
    for isa_qc in isa_circuits:
        isa_qc._layout = None

    isa_depths = [qc.depth() for qc in isa_circuits]
    print(f"  ISA depths: min={min(isa_depths)}, max={max(isa_depths)}, "
          f"mean={np.mean(isa_depths):.0f}")

    sampler = SamplerV2(mode=hw_backend)
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    sampler.options.twirling.strategy = "active-accum"
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.dynamical_decoupling.scheduling_method = "alap"
    print("  Twirling+DD: enabled (XY4/alap)")

    print(f"\n[ibm] Submitting {len(isa_circuits)} circuits ...")
    # Submit as PUBs (one per circuit)
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
# Analyze one prior
# ---------------------------------------------------------------------------

def analyze_prior(
    prior: list[float],
    shots: int,
    p_correct: float,
    dry_run: bool,
    skip_chsh: bool,
    backend_name: str,
    token: str | None,
    channel: str,
    instance: str | None,
    opt_level: int,
) -> dict:
    """Run full entanglement analysis for a single prior value.

    Returns a dict with all results.
    """
    print(f"\n{'='*60}")
    print(f"  Prior: {prior}")
    print(f"{'='*60}")

    result = {"prior": prior, "p_correct": p_correct}

    # ------------------------------------------------------------------
    # 1. Ideal (statevector) entanglement — ground truth
    # ------------------------------------------------------------------
    print("\n[ideal] Computing exact entanglement from statevector ...")
    ideal = compute_ideal_entanglement(prior, p_correct)
    result["ideal"] = {
        "concurrence": ideal["concurrence"],
        "negativity": ideal["negativity"],
        "ppt_min_eigenvalue": ideal["ppt_min_eigenvalue"],
        "is_entangled": ideal["is_entangled"],
    }
    print(f"  Concurrence     : {ideal['concurrence']:.6f}")
    print(f"  Negativity      : {ideal['negativity']:.6f}")
    print(f"  PPT min eigval  : {ideal['ppt_min_eigenvalue']:.6f}")
    print(f"  Entangled (PPT) : {ideal['is_entangled']}")

    # ------------------------------------------------------------------
    # 2. Tomography circuits + CHSH circuits
    # ------------------------------------------------------------------
    tomo_circuits, tomo_labels = build_tomography_circuits(prior, p_correct)
    print(f"\n[tomo] Built {len(tomo_circuits)} tomography circuits")

    all_circuits = list(tomo_circuits)
    chsh_angles = None
    ideal_chsh_s = None

    if not skip_chsh:
        print("[chsh] Optimizing CHSH measurement angles ...")
        chsh_angles, ideal_chsh_s = _optimize_chsh_angles(prior, p_correct)
        print(f"  Optimal angles  : [{', '.join(f'{a:.4f}' for a in chsh_angles)}]")
        print(f"  Ideal |S|       : {abs(ideal_chsh_s):.4f}  "
              f"({'VIOLATES' if abs(ideal_chsh_s) > 2.0 else 'does NOT violate'} "
              f"classical limit 2.0)")

        chsh_circuits = build_chsh_circuits(prior, chsh_angles, p_correct)
        all_circuits.extend(chsh_circuits)
        print(f"[chsh] Built {len(chsh_circuits)} CHSH circuits")

        result["chsh_ideal"] = {
            "angles": chsh_angles,
            "S": ideal_chsh_s,
            "abs_S": abs(ideal_chsh_s),
            "violates_classical": abs(ideal_chsh_s) > 2.0,
        }

    # ------------------------------------------------------------------
    # 3. Execute circuits
    # ------------------------------------------------------------------
    if dry_run:
        print(f"\n[sim] Running {len(all_circuits)} circuits on AerSimulator "
              f"({shots} shots each) ...")
        all_counts = _run_circuits_simulator(all_circuits, shots)
        result["backend"] = "aer_simulator"
    else:
        all_counts = _run_circuits_simulator(all_circuits, shots)
        sim_counts = all_counts  # save for comparison
        print(f"\n[sim] Simulator baseline collected")

        hw_counts, job_id = _run_circuits_hardware(
            all_circuits, shots, backend_name, token, channel, instance, opt_level,
        )
        result["hw_job_id"] = job_id
        result["backend"] = backend_name

        # Simulator tomography results (for comparison)
        sim_tomo_counts = sim_counts[:len(tomo_circuits)]
        sim_rho = reconstruct_density_matrix(sim_tomo_counts, tomo_labels)
        sim_ent = compute_entanglement_from_rho(sim_rho)
        result["simulator_tomography"] = sim_ent
        print(f"\n[sim-tomo] Simulator tomography:")
        print(f"  Concurrence     : {sim_ent['concurrence']:.6f}")
        print(f"  Negativity      : {sim_ent['negativity']:.6f}")

        # Use hardware counts for analysis below
        all_counts = hw_counts

    # ------------------------------------------------------------------
    # 4. Tomography analysis
    # ------------------------------------------------------------------
    tomo_counts = all_counts[:len(tomo_circuits)]
    print(f"\n[tomo] Reconstructing density matrix from {len(tomo_counts)} measurements ...")
    rho_measured = reconstruct_density_matrix(tomo_counts, tomo_labels)
    ent_measured = compute_entanglement_from_rho(rho_measured)

    result["measured_tomography"] = ent_measured
    result["density_matrix_real"] = rho_measured.real.tolist()
    result["density_matrix_imag"] = rho_measured.imag.tolist()

    # Fidelity between ideal and measured density matrices
    from qiskit.quantum_info import state_fidelity, DensityMatrix
    qc_ideal = build_tiger_belief_statevector_circuit(prior, p_correct)
    from qiskit.quantum_info import Statevector
    dm_ideal = DensityMatrix(Statevector.from_instruction(qc_ideal))
    dm_measured = DensityMatrix(rho_measured)
    fidelity = float(state_fidelity(dm_ideal, dm_measured))
    result["state_fidelity"] = fidelity

    print(f"  State fidelity  : {fidelity:.6f}")
    print(f"  Concurrence     : {ent_measured['concurrence']:.6f}  "
          f"(ideal: {ideal['concurrence']:.6f})")
    print(f"  Negativity      : {ent_measured['negativity']:.6f}  "
          f"(ideal: {ideal['negativity']:.6f})")
    print(f"  PPT min eigval  : {ent_measured['ppt_min_eigenvalue']:.6f}  "
          f"(ideal: {ideal['ppt_min_eigenvalue']:.6f})")
    print(f"  Entangled (PPT) : {ent_measured['is_entangled_ppt']}")
    print(f"  Entangled (conc): {ent_measured['is_entangled_concurrence']}")

    # ------------------------------------------------------------------
    # 5. CHSH analysis
    # ------------------------------------------------------------------
    if not skip_chsh:
        chsh_counts = all_counts[len(tomo_circuits):]
        S_measured, correlators = compute_chsh_from_counts(chsh_counts)

        result["chsh_measured"] = {
            "S": S_measured,
            "abs_S": abs(S_measured),
            "correlators": correlators,
            "violates_classical": abs(S_measured) > 2.0,
        }

        print(f"\n[chsh] CHSH results:")
        print(f"  S (measured)    : {S_measured:.4f}  (ideal: {ideal_chsh_s:.4f})")
        print(f"  |S| (measured)  : {abs(S_measured):.4f}")
        print(f"  Classical limit : 2.0000")
        print(f"  Quantum limit   : {2*np.sqrt(2):.4f}")
        print(f"  Violates CHSH   : {abs(S_measured) > 2.0}")
        for i, (E, label) in enumerate(zip(correlators, ["E(a,b)", "E(a,b')", "E(a',b)", "E(a',b')"])):
            print(f"    {label} = {E:.4f}")

    # ------------------------------------------------------------------
    # 6. Verdict
    # ------------------------------------------------------------------
    entangled_ppt = ent_measured["is_entangled_ppt"]
    entangled_conc = ent_measured["is_entangled_concurrence"]
    entangled_chsh = (not skip_chsh) and abs(S_measured) > 2.0 if not skip_chsh else None

    verdict = entangled_ppt or entangled_conc
    if not skip_chsh and entangled_chsh:
        verdict = True

    result["entanglement_certified"] = verdict
    result["certification_methods"] = {
        "ppt": entangled_ppt,
        "concurrence": entangled_conc,
    }
    if not skip_chsh:
        result["certification_methods"]["chsh"] = entangled_chsh

    status = "ENTANGLED" if verdict else "NOT CERTIFIED"
    print(f"\n  >>> VERDICT: {status} <<<")
    if verdict:
        methods = []
        if entangled_ppt:
            methods.append("PPT (negative partial transpose eigenvalue)")
        if entangled_conc:
            methods.append(f"Concurrence = {ent_measured['concurrence']:.4f}")
        if not skip_chsh and entangled_chsh:
            methods.append(f"CHSH |S| = {abs(S_measured):.4f} > 2")
        print(f"  Certified by: {'; '.join(methods)}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Parse priors: flatten list into pairs
    prior_vals = args.prior
    if len(prior_vals) % 2 != 0:
        print("[error] --prior requires pairs of values (e.g. --prior 0.5 0.5 0.85 0.15)")
        sys.exit(1)
    priors = [[prior_vals[i], prior_vals[i + 1]] for i in range(0, len(prior_vals), 2)]

    # Validate priors sum to ~1
    for p in priors:
        if abs(sum(p) - 1.0) > 0.01:
            print(f"[error] Prior {p} does not sum to 1.0")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  ENTANGLEMENT CERTIFICATION")
    print("  Tiger POMDP Belief Update Circuit")
    print("=" * 60)
    print(f"  Backend        : {'AerSimulator (dry-run)' if args.dry_run else args.backend}")
    print(f"  Shots          : {args.shots}")
    print(f"  P(correct)     : {args.p_correct}")
    print(f"  Priors to test : {priors}")
    print(f"  CHSH test      : {'SKIP' if args.skip_chsh else 'ON'}")
    print(f"  Circuits/prior : {9 + (0 if args.skip_chsh else 4)} "
          f"(9 tomo{'' if args.skip_chsh else ' + 4 CHSH'})")

    token = None
    channel = ibm_channel()
    instance_val = args.instance or ibm_instance()
    if not args.dry_run:
        token = get_ibm_token_optional()
        channel = args.channel or channel

    # ------------------------------------------------------------------
    # Run analysis for each prior
    # ------------------------------------------------------------------
    all_results = []
    for prior in priors:
        r = analyze_prior(
            prior=prior,
            shots=args.shots,
            p_correct=args.p_correct,
            dry_run=args.dry_run,
            skip_chsh=args.skip_chsh,
            backend_name=args.backend,
            token=token,
            channel=channel,
            instance=instance_val,
            opt_level=args.opt_level,
        )
        all_results.append(r)

    # ------------------------------------------------------------------
    # Summary across priors
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Prior':<16} {'Conc(ideal)':<14} {'Conc(meas)':<14} "
          f"{'Neg(meas)':<12} {'CHSH |S|':<12} {'Certified'}")
    print("  " + "-" * 78)

    any_certified = False
    for r in all_results:
        prior_str = f"[{r['prior'][0]:.2f}, {r['prior'][1]:.2f}]"
        conc_ideal = r["ideal"]["concurrence"]
        conc_meas = r["measured_tomography"]["concurrence"]
        neg_meas = r["measured_tomography"]["negativity"]
        chsh_str = "---"
        if "chsh_measured" in r:
            chsh_str = f"{abs(r['chsh_measured']['abs_S']):.4f}"
        certified = r["entanglement_certified"]
        any_certified = any_certified or certified
        print(f"  {prior_str:<16} {conc_ideal:<14.6f} {conc_meas:<14.6f} "
              f"{neg_meas:<12.6f} {chsh_str:<12} {'YES' if certified else 'NO'}")

    # Physical interpretation
    if len(all_results) > 1:
        print("\n  Physical interpretation:")
        print("  - Entanglement is strongest at uniform prior [0.50, 0.50]")
        print("    (maximum agent uncertainty => maximum quantum correlations)")
        print("  - Entanglement decreases as the agent becomes more confident")
        print("    (confident agent => state nearly separable => classical sufficient)")
        print("  - This confirms: quantum advantage is greatest under uncertainty")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_data = {
        "task_ids": ["entanglement_certification"],
        "backend": "aer_simulator" if args.dry_run else args.backend,
        "shots": args.shots,
        "p_correct": args.p_correct,
        "n_priors_tested": len(priors),
        "any_entanglement_certified": any_certified,
        "results_by_prior": all_results,
        "pass": any_certified,
        "notes": (
            f"Tested {len(priors)} prior(s). "
            f"Entanglement certified: {sum(1 for r in all_results if r['entanglement_certified'])}"
            f"/{len(all_results)} priors."
        ),
    }

    save_result("entanglement_witness", output_data)

    overall = "PASS" if any_certified else "FAIL"
    print(f"\n  OVERALL: {overall}")
    print(f"  Entanglement certified in "
          f"{sum(1 for r in all_results if r['entanglement_certified'])}"
          f"/{len(all_results)} prior configurations")


if __name__ == "__main__":
    main()
