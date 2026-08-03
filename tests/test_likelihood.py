"""Checks on the exact stationary likelihood that replaced the ABC layer."""

import numpy as np
import pytest

from stochtf.analytical import pgf
from stochtf.inference.likelihood import log_likelihood, prepare_counts

TRUE = dict(a_s=0.5, b_s=0.05, a_n=0.3, b_n=0.2, k_y=30.0, gamma=1.0)


@pytest.fixture(scope="module")
def counts():
    """Draw iid counts from the exact stationary distribution."""
    p = pgf.stationary_pmf(TRUE["a_s"], TRUE["b_s"], TRUE["a_n"], TRUE["b_n"],
                           TRUE["k_y"], TRUE["gamma"], "OR")
    rng = np.random.default_rng(0)
    return rng.choice(len(p), size=600, p=p / p.sum())


def test_likelihood_is_maximised_near_the_truth(counts):
    base = log_likelihood(counts, **TRUE, model="dimer")
    for name, factor in [("k_y", 1.3), ("k_y", 0.7), ("b_s", 3.0), ("a_s", 0.3)]:
        perturbed = dict(TRUE)
        perturbed[name] = perturbed[name] * factor
        assert log_likelihood(counts, **perturbed, model="dimer") < base


def test_likelihood_equals_the_sum_of_log_pmf(counts):
    """It must be exactly sum_i log P(y_i), with no summary statistic."""
    p = pgf.model_pmf("dimer", TRUE["a_s"], TRUE["b_s"], TRUE["a_n"],
                      TRUE["b_n"], TRUE["k_y"], TRUE["gamma"],
                      y_max=max(int(counts.max()), 400))
    expected = np.log(p[counts]).sum()
    assert log_likelihood(counts, **TRUE, model="dimer") == pytest.approx(
        expected, rel=1e-9)


@pytest.mark.parametrize("bad", [
    dict(a_s=-1.0), dict(gamma=0.0), dict(k_y=np.nan), dict(b_n=-0.5),
])
def test_invalid_parameters_give_minus_inf_not_an_exception(counts, bad):
    params = dict(TRUE)
    params.update(bad)
    assert log_likelihood(counts, **params, model="dimer") == -np.inf


def test_grid_reaches_the_largest_observation():
    """An outlier far above the mean must still get a finite log-probability."""
    y = np.array([0, 1, 2, 500])
    value = log_likelihood(y, **TRUE, model="dimer")
    assert np.isfinite(value)


def test_monomer_and_dimer_give_different_likelihoods(counts):
    assert (log_likelihood(counts, **TRUE, model="monomer")
            != log_likelihood(counts, **TRUE, model="dimer"))


def test_prepare_counts_rejects_non_integer_and_negative():
    with pytest.raises(ValueError, match="integers"):
        prepare_counts(np.array([1.0, 2.5]))
    with pytest.raises(ValueError, match="non-negative"):
        prepare_counts(np.array([1, -2]))
    with pytest.raises(ValueError, match="non-finite"):
        prepare_counts(np.array([1.0, np.nan]))
    assert prepare_counts(np.array([[1, 2], [3, 4]])).shape == (4,)
