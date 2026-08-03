"""Project directory locations, resolved from the installed package.
"""

import os

#: Repository root (three levels up from src/stochtf/paths.py).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "synthetic")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURE_DIR = os.path.join(ROOT, "figures", "output")


def processed(name):
    """Path to a processed per-gene count array, e.g. ``processed('sox2.npy')``."""
    return os.path.join(PROCESSED_DATA_DIR, name)

def synthetic(name):
    """Path to a processed per-gene count array, e.g. ``processed('sox2.npy')``."""
    return os.path.join(SYNTHETIC_DATA_DIR, name)


def raw(name):
    """Path to a raw downloaded file under ``data/raw/``."""
    return os.path.join(RAW_DATA_DIR, name)


def results(name):
    """Path under ``results/``, creating the directory on first use."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)
