"""Publication-ready matplotlib styles."""
from __future__ import annotations
from typing import Any

QUANTUM_COLORS = {
    "quantum": "#1f77b4",
    "classical": "#ff7f0e",
    "hybrid": "#2ca02c",
    "ibm": "#6929C4",
    "dwave": "#0062FF",
    "azure": "#00BCF2",
    "pennylane": "#64B5F6",
}

PUBLICATION_RCPARAMS: dict[str, Any] = {
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

def apply_publication_style() -> None:
    """Apply publication-ready style to matplotlib."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(PUBLICATION_RCPARAMS)

def get_color(label: str) -> str:
    """Get a consistent color for a given label."""
    return QUANTUM_COLORS.get(label, "#333333")
