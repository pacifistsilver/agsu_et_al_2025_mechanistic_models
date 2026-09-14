"""Figure style and output location, shared by every figure script.

Scripts call use_paper_style() rather than each carrying their own rcParams
block, so restyling the whole paper is one edit here.
"""

import os

import matplotlib as mpl

# Categorical palette used throughout the figures.
PALETTE = ["#2a6f97", "#c1440e", "#3d8168", "#8a5cb8", "#b08000"]

# One colour per promoter architecture, kept the same across figures.
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

# Set STOCHTF_FIGURE_DIR to build somewhere other than figures/output.
OUTPUT_DIR = os.environ.get(
    "STOCHTF_FIGURE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "figures", "output"),
)


def use_paper_style(sans_serif=None):
    """Apply the paper rcParams.

    sans_serif is opt-in rather than a default: the submitted figures used Arial,
    but plenty of machines don't have it and matplotlib then warns on every call.
    """
    mpl.rcParams.update(_PAPER_STYLE)
    if sans_serif is not None:
        mpl.rcParams["font.sans-serif"] = sans_serif


def output_path(filename):
    """Absolute path for a figure output. Makes the directory on first use."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, filename)
