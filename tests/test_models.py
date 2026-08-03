"""Checks on the PyMC layer.

Skipped unless the optional [inference] extra is installed.
"""

import numpy as np
import pytest

pytest.importorskip("pymc")
pytest.importorskip("pymc_extras")

import cloudpickle  # noqa: E402
import pytensor  # noqa: E402
import pytensor.tensor as pt  # noqa: E402

from stochtf.inference.models import (  # noqa: E402
    MODELS,
    StationaryLogLike,
)

COUNTS = np.array([0, 1, 5, 30, 42, 17])
ARGS = (0.5, 0.3, 0.05, 0.2, 30.0)  # alpha_s, alpha_n, beta_s, beta_n, k_y


def test_op_pickles_with_its_data():
    """Regression: sample_smc ships the kernel to worker processes.

    The Op was previously built by ``as_op``/``wrap_py`` inside a factory, so it
    reduced to a name lookup for a function that only existed as a closure, and
    sampling died with "Can't pickle wrap_py(), not found as
    stochtf.inference.models._logp".
    """
    op = StationaryLogLike(COUNTS, "dimer")
    restored = cloudpickle.loads(cloudpickle.dumps(op))
    assert np.array_equal(restored.counts, op.counts)
    assert restored.model_name == "dimer"


def test_compiled_function_pickles_and_agrees():
    """It is the compiled graph, not the bare Op, that SMC actually pickles."""
    op = StationaryLogLike(COUNTS, "dimer")
    inputs = [pt.dscalar(n) for n in ("a_s", "a_n", "b_s", "b_n", "k_y")]
    fn = pytensor.function(inputs, op(*inputs))
    restored = cloudpickle.loads(cloudpickle.dumps(fn))
    assert float(restored(*ARGS)) == pytest.approx(float(fn(*ARGS)), rel=1e-12)


def test_op_matches_the_underlying_likelihood():
    from stochtf.inference.likelihood import log_likelihood

    op = StationaryLogLike(COUNTS, "dimer")
    inputs = [pt.dscalar(n) for n in ("a_s", "a_n", "b_s", "b_n", "k_y")]
    fn = pytensor.function(inputs, op(*inputs))
    a_s, a_n, b_s, b_n, k_y = ARGS
    expected = log_likelihood(COUNTS, a_s, b_s, a_n, b_n, k_y, 1.0, model="dimer")
    assert float(fn(*ARGS)) == pytest.approx(expected, rel=1e-12)


def test_distinct_datasets_never_compare_equal():
    """No __props__, so graph merging cannot swap one dataset's Op for another."""
    a = StationaryLogLike(COUNTS, "dimer")
    b = StationaryLogLike(COUNTS + 1, "dimer")
    assert a != b
    assert StationaryLogLike(COUNTS, "dimer") != StationaryLogLike(COUNTS, "monomer")


@pytest.mark.parametrize("name", sorted(MODELS))
def test_build_model_produces_five_free_parameters(name):
    model = MODELS[name]()
    built = model.build_model(y=COUNTS)
    free = {v.name for v in built.free_RVs}
    assert free == {"alpha_s", "alpha_n", "beta_s", "beta_n", "k_y"}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_likelihood_is_grouped_with_the_data_logp(name):
    """SMC tempers model.datalogp; a Potential must land there, not in the prior."""
    model = MODELS[name]()
    built = model.build_model(y=COUNTS)
    # The priors are log-transformed, so use the model's own initial point
    # rather than trying to name the untransformed variables.
    point = built.initial_point()
    data_logp = float(built.compile_fn(built.datalogp)(point))
    assert np.isfinite(data_logp) and data_logp != 0.0


def test_build_model_requires_observed_counts():
    with pytest.raises(ValueError, match="observed counts are required"):
        MODELS["dimer"]().build_model()
