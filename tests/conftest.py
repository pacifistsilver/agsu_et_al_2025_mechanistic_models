"""Pytest configuration and fixtures for expression model tests."""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config_loader import load_yaml_config


@pytest.fixture
def sample_config():
    """Load sample configuration for testing."""
    config_path = PROJECT_ROOT / "src" / "config" / "hetmer_excl.yaml"
    if config_path.exists():
        return load_yaml_config(str(config_path))
    return {
        "experiment_name": "test_experiment",
        "sim_time": 1000,
        "runs": 10,
        "binding_sites": 10,
    }


@pytest.fixture
def project_root():
    """Return project root path."""
    return PROJECT_ROOT
