"""What stationary counts can and cannot determine.

These encode findings that constrain how the inference results may be read, so
that a later change which appears to "fix" the wide posteriors gets checked
against the underlying information content rather than against intuition.
"""

import numpy as np
import pytest

from stochtf.inference import identifiability as ident

#: Bursty regime: slow switching relative to gamma, large k_y. This is the most
#: favourable case for resolving the promoter, and it still fails for two sites.
TWO_SITE = (0.3, 0.2, 0.15, 0.5, 40.0)
#: Same regime with site n switched off.
SINGLE_SITE = (0.3, 0.2, 1e-9, 1.0, 40.0)

N_OBS = 800


@pytest.mark.parametrize("gate", ["OR", "AND", "ADD"])
def test_the_two_sites_are_exchangeable(gate):
    """An exact discrete degeneracy: swapping the sites leaves P(y) unchanged.

    So (alpha_s, beta_s) and (alpha_n, beta_n) are identifiable only as an
    unordered pair, and the posterior has two equivalent modes.
    """
    assert ident.exchange_symmetry_residual(TWO_SITE, gate) < 1e-12


def test_single_site_telegraph_is_identifiable():
    """alpha, beta and k_y are all determined, as the Poisson-Beta literature says."""
    se = ident.standard_errors(SINGLE_SITE, N_OBS, "OR", free=[0, 1, 4])
    assert np.all(se < 0.2), f"expected all under 20%, got {se * 100}"


def test_two_site_switching_rates_are_not_identifiable():
    """Adding the second site destroys it, in the same bursty regime."""
    se = ident.standard_errors(TWO_SITE, N_OBS, "OR")
    alpha_s, beta_s, alpha_n, beta_n, k_y = se
    assert k_y < 0.1, "k_y should still be sharp"
    assert min(alpha_s, beta_s, alpha_n, beta_n) > 1.0, (
        "the four switching rates should be undetermined (>100% error); "
        f"got {se[:4] * 100}"
    )


def test_no_parameter_regime_rescues_the_two_site_rates():
    """Slow, fast and mixed regimes alike leave the switching rates flat."""
    regimes = [
        (100.0, 12.0, 60.0, 40.0, 20.0),   # fast switching
        (0.5, 0.05, 0.2, 0.4, 30.0),       # moderate
        (0.3, 0.1, 0.05, 0.5, 40.0),       # slow and bursty
        (2.0, 0.3, 0.1, 3.0, 25.0),        # asymmetric
    ]
    for theta in regimes:
        se = ident.standard_errors(theta, N_OBS, "OR")
        assert min(se[:4]) > 1.0, f"unexpectedly identified at {theta}: {se[:4]}"


def test_only_two_directions_are_determined():
    """Of five log-parameter directions, most of the information is in two."""
    vals, vecs = ident.identifiable_directions(TWO_SITE, N_OBS, "OR")
    assert vals[0] > 1e3          # k_y
    assert vals[1] > 1e2          # an occupancy-like combination
    assert vals[3] < 1.0          # flat
    assert vals[4] < 1.0          # flat
    assert vals[0] / vals[4] > 1e6, "expected a severely ill-conditioned problem"


def test_best_determined_direction_is_k_y():
    vals, vecs = ident.identifiable_directions(TWO_SITE, N_OBS, "OR")
    leading = np.abs(vecs[:, 0])
    assert leading.argmax() == 4, "k_y should dominate the leading direction"


def test_information_scales_with_sample_size():
    """Fisher information is additive over iid observations."""
    i1 = ident.fisher_information(SINGLE_SITE, 100, "OR", free=[0, 1, 4])
    i2 = ident.fisher_information(SINGLE_SITE, 400, "OR", free=[0, 1, 4])
    assert np.allclose(i2, 4 * i1, rtol=1e-6)


def test_describe_reports_the_verdict():
    text = ident.describe(TWO_SITE, N_OBS, "OR")
    assert "NOT identified" in text
    assert "Fano" in text
