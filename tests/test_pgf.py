"""Checks on the generating-function machinery.

Three independent routes to the same stationary distribution are compared: the
closed-form PGF inverted by FFT, the block-tridiagonal CME solve, and the
sparse FSP already in ``analytical.gates``. A fourth, the ODE integration of the
defining equation, is checked separately because it is slow.
"""

import numpy as np
import pytest

from stochtf.analytical import gates as G
from stochtf.analytical import heterodimer as ts
from stochtf.analytical import pgf

PARAMS = (0.5, 0.05, 0.3, 0.2)
GATES = ["OR", "AND", "ADD"]


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("k_y,gamma", [(1.0, 0.1), (30.0, 1.0)])
def test_stationary_pmf_matches_sparse_fsp(gate, k_y, gamma):
    """The CME route must reproduce the FSP distribution already in the repo."""
    y_max = 500
    p = pgf.stationary_pmf(*PARAMS, k_y, gamma, gate, y_max=y_max)
    p_fsp = G.fsp(*PARAMS, k_y, gamma, gate, ymax=y_max)
    assert np.abs(p - p_fsp).max() < 1e-12
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(p >= 0)


@pytest.mark.parametrize("gate", GATES)
def test_factorial_moments_match_the_distribution(gate):
    """Exact moments must agree with moments taken from the PMF."""
    k_y, gamma = 20.0, 1.0
    mean, var, fano = pgf.moments(*PARAMS, k_y, gamma, gate)
    p = pgf.stationary_pmf(*PARAMS, k_y, gamma, gate, y_max=800)
    y = np.arange(p.size)
    assert (y * p).sum() == pytest.approx(mean, rel=1e-9)
    assert (y**2 * p).sum() - mean**2 == pytest.approx(var, rel=1e-7)
    assert fano == pytest.approx(var / mean, rel=1e-12)


def test_add_gate_closed_form_matches_the_cme():
    """The two-Kummer product is exact where hyp1f1 is well conditioned."""
    k_y, gamma = 1.0, 0.1  # k_y/gamma = 10, inside the safe range
    p_pgf = pgf.stationary_pmf(*PARAMS, k_y, gamma, "ADD", y_max=400,
                               method="pgf")
    p_cme = pgf.stationary_pmf(*PARAMS, k_y, gamma, "ADD", y_max=400)
    assert np.abs(p_pgf - p_cme).max() < 1e-12


def test_telegraph_pgf_is_a_valid_generating_function():
    """G(1) = 1, and G'(1) is the mean."""
    a, b, k_y, gamma = 0.5, 0.05, 1.0, 0.1
    assert pgf.telegraph_pgf(1.0, a, b, k_y, gamma) == pytest.approx(1.0, abs=1e-12)
    h = 1e-6
    dG = (pgf.telegraph_pgf(1.0, a, b, k_y, gamma)
          - pgf.telegraph_pgf(1.0 - h, a, b, k_y, gamma)) / h
    # single site: <y> = (k_y/gamma) * p_on
    assert dG == pytest.approx((k_y / gamma) * a / (a + b), rel=1e-4)


def test_pgf_closed_form_refuses_gates_that_do_not_separate():
    for gate in ["OR", "AND"]:
        with pytest.raises(ValueError, match="no closed-form PGF"):
            pgf.pgf(0.5, *PARAMS, 1.0, 0.1, gate)


def test_pgf_fft_refuses_when_hyp1f1_is_unreliable():
    """Rather than quietly wrong numbers, or NaN at k_y/gamma ~ 500."""
    with pytest.raises(ValueError, match="exceeds"):
        pgf.stationary_pmf(*PARAMS, 250.0, 1.0, "ADD", y_max=800, method="pgf")


def test_moments_agree_with_gates_and_expose_the_fano_bug():
    """A fourth independent confirmation of the heterodimer.fano discrepancy.

    The PGF factorial-moment recursion agrees with gates.analytic and with the
    exact FSP; heterodimer.fano does not, except when k_y == gamma.
    """
    k_y, gamma = 2.0, 0.5
    _, _, fano_pgf = pgf.moments(*PARAMS, k_y, gamma, "OR")
    _, fano_gates = G.analytic(*PARAMS, k_y, gamma, "OR")
    assert fano_pgf == pytest.approx(fano_gates, rel=1e-10)

    excess_ratio = (fano_pgf - 1) / (ts.fano(*PARAMS, k_y, gamma) - 1)
    assert excess_ratio == pytest.approx(k_y / gamma, rel=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("gate", ["OR", "ADD"])
def test_ode_pgf_matches_the_closed_form_and_the_cme(gate):
    """The ODE integration validates the other two routes for any gate."""
    k_y, gamma = 1.0, 0.1
    n = 512
    z = np.exp(2j * np.pi * np.arange(n) / n)
    g_ode = pgf.pgf_ode(z - 1.0, *PARAMS, k_y, gamma, gate)
    p_ode = pgf.pmf_from_pgf(g_ode, n)
    p_cme = pgf.stationary_pmf(*PARAMS, k_y, gamma, gate, y_max=n - 1)
    assert np.abs(p_ode - p_cme).max() < 1e-5
