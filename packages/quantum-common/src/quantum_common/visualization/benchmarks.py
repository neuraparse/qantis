"""Benchmark visualization utilities."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np

def plot_speedup_curve(
    problem_sizes: Sequence[int],
    quantum_times: Sequence[float],
    classical_times: Sequence[float],
    title: str = "Quantum vs Classical Speedup",
    save_path: str | None = None,
) -> Any:
    """Plot quantum speedup as a function of problem size."""
    import matplotlib.pyplot as plt
    from quantum_common.visualization.styles import apply_publication_style, QUANTUM_COLORS

    apply_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time comparison
    ax1.semilogy(problem_sizes, classical_times, "o-", color=QUANTUM_COLORS["classical"], label="Classical")
    ax1.semilogy(problem_sizes, quantum_times, "s-", color=QUANTUM_COLORS["quantum"], label="Quantum")
    ax1.set_xlabel("Problem Size")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Execution Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Speedup
    speedups = [c / q if q > 0 else 0 for c, q in zip(classical_times, quantum_times)]
    ax2.plot(problem_sizes, speedups, "D-", color=QUANTUM_COLORS["hybrid"])
    ax2.axhline(y=1.0, linestyle="--", color="gray", alpha=0.5)
    ax2.set_xlabel("Problem Size")
    ax2.set_ylabel("Speedup (Classical / Quantum)")
    ax2.set_title("Quantum Speedup")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_scalability(
    problem_sizes: Sequence[int],
    metrics: dict[str, Sequence[float]],
    title: str = "Scalability Analysis",
    ylabel: str = "Value",
    save_path: str | None = None,
) -> Any:
    """Plot scalability curves for multiple approaches."""
    import matplotlib.pyplot as plt
    from quantum_common.visualization.styles import apply_publication_style, QUANTUM_COLORS

    apply_publication_style()
    fig, ax = plt.subplots()
    colors = list(QUANTUM_COLORS.values())
    for i, (label, values) in enumerate(metrics.items()):
        color = colors[i % len(colors)]
        ax.plot(problem_sizes, values, "o-", color=color, label=label)

    ax.set_xlabel("Problem Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
