"""Checks on the sFSP stationary solver.

The reference points are deliberately independent of the solver: the closed-form
route in ``stochtf.analytical.pgf`` (heterodimer only), a truncation-free moment
recursion, and a brute-force eigen-solve of the redirected generator assembled
by hand from equation (3.14) of the paper.
"""

import numpy as np
import pytest
from scipy.linalg import null_space

from stochtf.analytical import pgf
from stochtf.cme import stationary as st

GAMMA = 1.0

#: (label, a_s, b_s, a_n, b_n, k_y), spanning bursty to fast-switching.
CASES = [
    ("bursty", 1.0, 0.06, 0.15, 0.24, 20.0),
    ("pinned betas", 0.9, 0.04, 0.30, 0.26, 56.0),
    ("fast switching", 30.0, 20.0, 15.0, 25.0, 40.0),
    ("very bursty", 0.05, 0.05, 0.02, 0.10, 60.0),
]
IDS = [c[0] for c in CASES]


def heterodimer(a_s, b_s, a_n, b_n):
    """Two independent sites, OR gate; states ordered 00, 10, 01, 11."""
    Q = st.generator_from_jumps(4, [
        (0, 1, a_s), (0, 2, a_n), (1, 0, b_s), (1, 3, a_n),
        (2, 0, b_n), (2, 3, a_s), (3, 1, b_n), (3, 2, b_s)])
    return Q, np.array([0.0, 1.0, 1.0, 1.0])


def exclusive(a_s, b_s, a_n, b_n):
    """One contested site: empty, SOX2-bound, NANOG-bound."""
    Q = st.generator_from_jumps(3, [
        (0, 1, a_s), (0, 2, a_n), (1, 0, b_s), (2, 0, b_n)])
    return Q, np.array([0.0, 1.0, 1.0])


CHAINS = {"heterodimer": heterodimer, "exclusive": exclusive}


def redirected_generator(Q, act, k_y, gamma, y_max, designated=0):
    """Qbar = Q_n + c_n b_l, assembled explicitly as in equation (3.14)."""
    n_states, ny = Q.shape[0], y_max + 1
    size = n_states * ny
    Qn = np.zeros((size, size))
    c = np.zeros(size)

    for s in range(n_states):
        for y in range(ny):
            i = s * ny + y
            for t in range(n_states):
                if t != s and Q[s, t] != 0:
                    Qn[i, t * ny + y] += Q[s, t]
                    Qn[i, i] -= Q[s, t]
            up = k_y * act[s]
            if up > 0:
                Qn[i, i] -= up          # the diagonal keeps the escaping rate
                if y < ny - 1:
                    Qn[i, i + 1] += up
                else:
                    c[i] += up          # ... and it lands in the outflow vector
            if y > 0:
                Qn[i, i - 1] += gamma * y
                Qn[i, i] -= gamma * y

    Qbar = Qn.copy()
    Qbar[:, designated * ny] += c       # add c_n to the designated column
    return Qbar, c


# ----------------------------------------------------------------------
# the redirected generator
# ----------------------------------------------------------------------

@pytest.mark.parametrize("chain", sorted(CHAINS))
def test_redirection_gives_a_valid_generator(chain):
    """Rows of Qbar must sum to zero, or it is not a rate matrix at all.

    This is the point of redirecting rather than absorbing: the projected chain
    stays conservative, so it has a real stationary distribution.

    Args:
        chain: Promoter chain under test.
    """
    Q, act = CHAINS[chain](*CASES[0][1:5])
    Qbar, c = redirected_generator(Q, act, CASES[0][5], GAMMA, y_max=40)
    assert np.abs(Qbar.sum(axis=1)).max() < 1e-12
    assert np.all(c >= 0)
    assert c.sum() > 0  # something really does escape at this truncation


@pytest.mark.parametrize("chain", sorted(CHAINS))
def test_solver_matches_a_brute_force_eigensolve(chain):
    """The fast sweep must reproduce the stationary law of the explicit Qbar."""
    label, a_s, b_s, a_n, b_n, k_y = CASES[0]
    Q, act = CHAINS[chain](a_s, b_s, a_n, b_n)
    y_max = 40

    Qbar, c = redirected_generator(Q, act, k_y, GAMMA, y_max)
    basis = null_space(Qbar.T)
    assert basis.shape[1] == 1, "the projected chain must be irreducible"
    reference = np.abs(basis[:, 0])
    reference /= reference.sum()

    p, diagnostics = st.stationary_pmf(Q, act, k_y, GAMMA, y_max=y_max,
                                       tol=np.inf, return_diagnostics=True)
    expected = reference.reshape(Q.shape[0], y_max + 1).sum(axis=0)
    assert np.abs(p - expected).max() < 1e-12
    assert diagnostics["outflow"] == pytest.approx(float(c @ reference),
                                                   rel=1e-9)


@pytest.mark.parametrize("chain", sorted(CHAINS))
def test_answer_does_not_depend_on_the_redirection_target(chain):
    """x_l is a free choice in the method; it must not move the answer."""
    Q, act = CHAINS[chain](*CASES[0][1:5])
    solutions = [st.stationary_pmf(Q, act, CASES[0][5], GAMMA, y_max=120,
                                   tol=np.inf, designated=s)
                 for s in range(Q.shape[0])]
    for other in solutions[1:]:
        assert np.abs(solutions[0] - other).max() < 1e-10


# ----------------------------------------------------------------------
# correctness against independent routes
# ----------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_heterodimer_matches_the_closed_form_route(case):
    """pgf reaches the same distribution a different way, for this topology."""
    _, a_s, b_s, a_n, b_n, k_y = case
    Q, act = heterodimer(a_s, b_s, a_n, b_n)
    p = st.stationary_pmf(Q, act, k_y, GAMMA)
    q = pgf.stationary_pmf(a_s, b_s, a_n, b_n, k_y, GAMMA, "OR",
                           y_max=p.size - 1)
    assert np.abs(p - q / q.sum()).max() < 1e-12


@pytest.mark.parametrize("chain", sorted(CHAINS))
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_moments_of_the_projection_match_the_exact_recursion(chain, case):
    """The moment recursion truncates nothing, so it audits the projection."""
    _, a_s, b_s, a_n, b_n, k_y = case
    Q, act = CHAINS[chain](a_s, b_s, a_n, b_n)
    p = st.stationary_pmf(Q, act, k_y, GAMMA)

    grid = np.arange(p.size)
    mean = float(grid @ p)
    var = float((grid**2) @ p) - mean**2
    mean_exact, var_exact, _ = st.moments(Q, act, k_y, GAMMA)

    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert mean == pytest.approx(mean_exact, rel=1e-9)
    assert var == pytest.approx(var_exact, rel=1e-9)


# ----------------------------------------------------------------------
# the error certificate
# ----------------------------------------------------------------------

def test_convergence_factor_bounds_and_tracks_the_error():
    """Theorem 3.1: the computable gamma brackets the error you cannot compute.

    Both must fall together as the projection grows, and the true error must
    stay below gamma -- that is parts (C) and (D) with M of order one.
    """
    Q, act = heterodimer(1.0, 0.06, 0.15, 0.24)
    truth = st.stationary_pmf(Q, act, 20.0, GAMMA, y_max=600, tol=np.inf)

    factors, errors = [], []
    for y_max in (25, 30, 35, 40, 50, 60):
        p, d = st.stationary_pmf(Q, act, 20.0, GAMMA, y_max=y_max, tol=np.inf,
                                 return_diagnostics=True)
        padded = np.zeros(truth.size)
        padded[:p.size] = p
        factors.append(d["convergence_factor"])
        errors.append(float(np.abs(padded - truth).sum()))

    assert all(np.diff(factors) < 0), "gamma must fall as the grid grows"
    assert all(np.diff(errors) < 0), "so must the error it stands in for"
    for factor, error in zip(factors, errors):
        assert error <= factor          # part (C), with M <= 1 here
        assert error >= 1e-3 * factor   # part (D): not a vacuous bound


def test_expansion_stops_once_the_tolerance_is_met():
    """Algorithm 1 grows the projection until gamma drops below tol."""
    Q, act = heterodimer(1.0, 0.06, 0.15, 0.24)
    loose = st.stationary_pmf(Q, act, 20.0, GAMMA, y_max=8, tol=1e-2,
                              return_diagnostics=True)[1]
    tight = st.stationary_pmf(Q, act, 20.0, GAMMA, y_max=8, tol=1e-12,
                              return_diagnostics=True)[1]
    assert loose["convergence_factor"] <= 1e-2
    assert tight["convergence_factor"] <= 1e-12
    assert tight["y_max"] > loose["y_max"]


@pytest.mark.parametrize("chain", sorted(CHAINS))
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_drift_condition_holds_for_the_lyapunov_function(chain, case):
    """V = 1 + y must satisfy QV <= C1 - C2 V, which licenses the bound."""
    _, a_s, b_s, a_n, b_n, k_y = case
    _, act = CHAINS[chain](a_s, b_s, a_n, b_n)
    C1, C2 = st.drift_constants(act, k_y, GAMMA)
    assert C1 > 0 and C2 > 0

    y = np.arange(0, 500)
    for state in range(act.size):
        drift = k_y * act[state] - GAMMA * y      # QV(state, y)
        assert np.all(drift <= C1 - C2 * (1.0 + y) + 1e-12)


def test_rejects_unusable_rates():
    Q, act = heterodimer(1.0, 0.06, 0.15, 0.24)
    with pytest.raises(ValueError):
        st.stationary_pmf(Q, act, 20.0, 0.0)      # gamma must be positive
    with pytest.raises(ValueError):
        st.stationary_pmf(Q, act, -1.0, GAMMA)    # negative transcription
