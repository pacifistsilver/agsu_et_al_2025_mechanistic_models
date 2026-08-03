"""Consistency checks for the two-site OR promoter results.

Converted from the former ``src/analytical/plots/validate.py``, which printed
these quantities for a human to eyeball. Each section of that script asserted an
identity implicitly; here the identities are checked.

The parameter set is the same deliberately awkward asymmetric one.
"""

import numpy as np
import pytest

from stochtf.analytical import gates as G
from stochtf.analytical import heterodimer as ts

A_S, B_S, A_N, B_N = 0.5, 0.05, 0.3, 0.2
K_Y, GAMMA = 0.1, 0.01

PARAMS = (A_S, B_S, A_N, B_N)


def test_burst_frequency_is_inverse_mean_cycle_time():
    """f = 1 / (<T_on> + <T_off>): every burst is one OFF->ON crossing."""
    f = ts.burst_frequency(*PARAMS)
    t_on = ts.t_on(*PARAMS)
    t_off = ts.t_off(*PARAMS)
    assert f == pytest.approx(1.0 / (t_on + t_off), rel=1e-10)


@pytest.mark.xfail(
    reason="tbound() disagrees with the stationary bound probability: "
           "pbound() != t_on/(t_on+t_off), by 0.4% at these rates and up to "
           "2.5% elsewhere, so <y> = b f / gamma does not close. "
           "mean_y and pbound are the exact stationary results; the MFPT in "
           "tbound() is the suspect term. See PR discussion.",
    strict=True,
)
def test_mean_expression_matches_burst_decomposition():
    """<y> = b f / gamma must hold exactly for this regenerative promoter."""
    f = ts.burst_frequency(*PARAMS)
    b = ts.burst_size(*PARAMS, K_Y)
    assert ts.mean_y(*PARAMS, K_Y, GAMMA) == pytest.approx(b * f / GAMMA, rel=1e-10)


def test_bound_probability_is_the_exact_stationary_value():
    """pbound = 1 - q_s q_n, independent of the MFPT route. This one is right."""
    _, _, _, _, q_s, q_n = ts.derived(*PARAMS)
    assert ts.pbound(*PARAMS) == pytest.approx(1.0 - q_s * q_n, rel=1e-12)


def test_burst_size_pmf_is_normalised():
    pmf = ts.burst_size_pmf(*PARAMS, K_Y, mmax=4000)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-8)


def test_burst_size_pgf_is_normalised():
    """B(1) = 1."""
    assert ts.burst_size_pgf(1.0, *PARAMS, K_Y) == pytest.approx(1.0, abs=1e-8)


def test_pmf_and_pgf_agree_on_the_mean_burst_size():
    """The two independent routes to <burst size> must agree with each other."""
    pmf = ts.burst_size_pmf(*PARAMS, K_Y, mmax=4000)
    m = np.arange(pmf.size)
    h = 1e-6
    dB = (ts.burst_size_pgf(1.0, *PARAMS, K_Y)
          - ts.burst_size_pgf(1.0 - h, *PARAMS, K_Y)) / h
    assert (m * pmf).sum() == pytest.approx(dB, rel=1e-4)


def test_pmf_mean_is_consistent_with_the_bound_probability():
    """<burst size>/k_y is the mean ON duration, which must satisfy
    p_bound = t_on / (t_on + t_off). The phase-type PMF does satisfy this."""
    pmf = ts.burst_size_pmf(*PARAMS, K_Y, mmax=4000)
    m = np.arange(pmf.size)
    t_on_from_pmf = (m * pmf).sum() / K_Y
    p_bound = ts.pbound(*PARAMS)
    t_off = ts.t_off(*PARAMS)
    assert t_on_from_pmf == pytest.approx(p_bound / (1 - p_bound) * t_off, rel=1e-6)


@pytest.mark.xfail(
    reason="burst_size() = k_y * tbound(), and tbound() is inconsistent with "
           "both the phase-type PMF and the stationary bound probability "
           "(36.81 vs 33.13 at these rates). The PMF, the PGF and pbound all "
           "agree with each other; tbound() is the outlier. See PR discussion.",
    strict=True,
)
def test_burst_size_matches_the_phase_type_pmf():
    pmf = ts.burst_size_pmf(*PARAMS, K_Y, mmax=4000)
    m = np.arange(pmf.size)
    assert (m * pmf).sum() == pytest.approx(ts.burst_size(*PARAMS, K_Y), rel=1e-6)


def test_burst_size_is_overdispersed_relative_to_a_single_geometric():
    """Three geometric components, so CV^2 exceeds the single-geometric value."""
    pmf = ts.burst_size_pmf(*PARAMS, K_Y, mmax=4000)
    m = np.arange(pmf.size)
    mean = (m * pmf).sum()
    var = (m**2 * pmf).sum() - mean**2
    b = ts.burst_size(*PARAMS, K_Y)
    assert var / mean**2 > (1 + b) / b

    rates = ts.burst_size_geometric_rates(*PARAMS, K_Y)
    assert len(rates) == 3
    assert np.all((np.asarray(rates) > 0) & (np.asarray(rates) < 1))


@pytest.mark.xfail(
    reason="fano() divides the excess term by mean_y instead of by the mean "
           "promoter activity sigma = 1 - q_s q_n. Since mean_y = k_y sigma / "
           "gamma, the k_y cancels and the excess comes out a factor k_y/gamma "
           "too small -- correct only when k_y == gamma. The sibling "
           "gates.analytic(...,'OR') computes it correctly and agrees with FSP "
           "to 8 decimal places. See PR discussion.",
    strict=True,
)
def test_closed_form_fano_matches_exact_fsp():
    """The Cox/moment-closure Fano factor must equal the FSP stationary one."""
    P = ts.fsp_stationary(*PARAMS, K_Y, GAMMA, ymax=600)
    y = np.arange(P.size)
    mu = (y * P).sum()
    var = (y**2 * P).sum() - mu**2
    assert P[-1] < 1e-9, "ymax too small: mass is still leaking off the grid"
    assert var / mu == pytest.approx(ts.fano(*PARAMS, K_Y, GAMMA), rel=1e-4)


def test_gates_module_reproduces_the_exact_fsp_fano():
    """Cross-check: the independent gates implementation does match FSP.

    This is what pins the discrepancy above on fano() rather than on the FSP
    solver or on the test's parameter conventions.
    """
    P = ts.fsp_stationary(*PARAMS, K_Y, GAMMA, ymax=4000)
    y = np.arange(P.size)
    mu = (y * P).sum()
    var = (y**2 * P).sum() - mu**2

    mu_gate, F_gate = G.analytic(*PARAMS, K_Y, GAMMA, "OR")
    assert mu == pytest.approx(mu_gate, rel=1e-6)
    assert var / mu == pytest.approx(F_gate, rel=1e-6)


# The effective-burst-parameter test that lived here covered burst_size_eff and
# burst_frequency_eff, which have since been removed from heterodimer.py: both
# were derived from fano(), whose excess term is a factor k_y/gamma too small
# (see test_closed_form_fano_matches_exact_fsp). stochtf.analytical.pgf.moments
# gives the corrected Fano factor.


def test_single_site_limit():
    """With site n switched off the two-site result collapses to one site."""
    f = ts.burst_frequency(A_S, B_S, 1e-12, B_N)
    assert f == pytest.approx(B_S / (A_S + B_S) * A_S, rel=1e-6)
    b = ts.burst_size(A_S, B_S, 1e-12, B_N, K_Y)
    assert b == pytest.approx(K_Y / B_S, rel=1e-6)


@pytest.mark.slow
def test_gillespie_agrees_with_the_phase_type_burst_size():
    """Direct SSA, checked against the phase-type PMF to 4 s.e.

    This is the fourth independent route to the mean burst size (PMF, PGF,
    pbound, SSA); all four agree, and all four disagree with k_y * tbound().
    """
    _, size, _ = ts.gillespie_bursts(*PARAMS, K_Y, n_bursts=60000, seed=1)
    pmf = ts.burst_size_pmf(*PARAMS, K_Y, mmax=4000)
    m = np.arange(pmf.size)

    se = size.std(ddof=1) / np.sqrt(size.size)
    assert abs(size.mean() - (m * pmf).sum()) < 4 * se


@pytest.mark.slow
def test_gillespie_off_duration_is_exponential():
    """<T_off> = 1/(alpha_s + alpha_n) exactly; this part of the module is right."""
    _, _, off = ts.gillespie_bursts(*PARAMS, K_Y, n_bursts=60000, seed=1)
    se = off.std(ddof=1) / np.sqrt(off.size)
    assert abs(off.mean() - ts.t_off(*PARAMS)) < 4 * se


@pytest.mark.slow
@pytest.mark.xfail(
    reason="t_on() returns tbound(), which the SSA contradicts: simulated mean "
           "ON duration matches the phase-type value, not tbound(). Same root "
           "cause as the burst-size and <y> = b f / gamma failures.",
    strict=True,
)
def test_gillespie_agrees_with_tbound():
    dur, _, _ = ts.gillespie_bursts(*PARAMS, K_Y, n_bursts=60000, seed=1)
    se = dur.std(ddof=1) / np.sqrt(dur.size)
    assert abs(dur.mean() - ts.t_on(*PARAMS)) < 4 * se
