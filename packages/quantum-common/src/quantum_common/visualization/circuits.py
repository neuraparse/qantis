"""Circuit drawing utilities."""
from __future__ import annotations
from typing import Any

def draw_circuit(circuit: Any, output: str = "mpl", save_path: str | None = None) -> Any:
    """Draw a Qiskit QuantumCircuit."""
    fig = circuit.draw(output=output)
    if save_path and hasattr(fig, "savefig"):
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

def circuit_summary(circuit: Any) -> dict[str, int]:
    """Get summary statistics for a circuit."""
    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "num_clbits": circuit.num_clbits,
        "count_ops": dict(circuit.count_ops()),
    }
