import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "synthetic")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURE_DIR = os.path.join(ROOT, "figures", "output")


def processed(name):
    return os.path.join(PROCESSED_DATA_DIR, name)

def synthetic(name):
    return os.path.join(SYNTHETIC_DATA_DIR, name)

def raw(name):
    return os.path.join(RAW_DATA_DIR, name)

def results(name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)
