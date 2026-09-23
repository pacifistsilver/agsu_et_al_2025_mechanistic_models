import os

import matplotlib as mpl

PALETTE = ["#2a6f97", "#c1440e", "#3d8168", "#8a5cb8", "#b08000"]

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

OUTPUT_DIR = os.environ.get(
    "STOCHTF_FIGURE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "figures", "output"),
)


def use_paper_style(sans_serif=None):
    mpl.rcParams.update(_PAPER_STYLE)
    if sans_serif is not None:
        mpl.rcParams["font.sans-serif"] = sans_serif


def output_path(filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, filename)
