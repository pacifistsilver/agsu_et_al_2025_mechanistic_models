"""Shared figure style and output location.

Every figure script calls :func:`use_paper_style` instead of carrying its own
copy of the rcParams block, so a change to the paper style is a one-line edit.
"""

import os

import matplotlib as mpl

#: Categorical palette used throughout the figures.
PALETTE = ["#2a6f97", "#c1440e", "#3d8168", "#8a5cb8", "#b08000"]

#: Per-architecture colours, shared by the logic and architecture figures.
ARCH_COLOURS = {
    "OR": "#2a6f97",
    "AND": "#c1440e",
    "ADD": "#3d8168",
    "DIM": "#8a5cb8",
}

_PAPER_STYLE = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "legend.frameon": False,
    "axes.grid": False,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
}

#: Directory the figure scripts write into, overridable for out-of-tree builds.
OUTPUT_DIR = os.environ.get(
    "STOCHTF_FIGURE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "figures", "output"),
)


def use_paper_style(sans_serif=None):
    """Apply the shared paper rcParams.

    Parameters
    ----------
    sans_serif : str, optional
        Font family to request. Arial is what the submitted figures used, but it
        is not present on every machine, so it is opt-in per figure rather than
        a global default that would emit font-fallback warnings everywhere.
    """
    mpl.rcParams.update(_PAPER_STYLE)
    if sans_serif is not None:
        mpl.rcParams["font.sans-serif"] = sans_serif


def output_path(filename):
    """Absolute path for a figure output, creating the directory on first use."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, filename)
