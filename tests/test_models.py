"""Checks on the ABC-SMC layer.

Skipped unless the optional [inference] extra is installed.

The simulator is checked against exact stationary means computed analytically,
which is the property that broke in the previous ABC implementation: it sampled
a non-stationary trajectory and could not reach the observed means at all.
"""

import numpy as np
import pytest

pytest.importorskip("pymc")
pytest.importorskip("pymc_extras")

import cloudpickle  # noqa: E402
import pymc as pm  # noqa: E402

from stochtf.inference.models import (  # noqa: E402
    FIXED_BETA_N,
    FIXED_BETA_S,
    GAMMA,
    MODELS,
    PROMOTERS,
    heterodimer_chain,
    monomer_chain,
    simulate_counts,
)

#: alpha_s, beta_s, alpha_n, beta_n, k_y -- bursty, like the observed counts.
RATES = (1.0, 0.06, 0.15, 0.24, 20.0)
COUNTS = np.array([0, 1, 5, 30, 42, 17])


def exact_mean(promoter, a_s, b_s, a_n, b_n, k_y):
    """Stationary mean count: k_y * P(promoter occupied) / gamma."""
    if promoter == "heterodimer":
        # Silent only when both sites are free, and they are independent.
        occupied = 1.0 - (b_s / (a_s + b_s)) * (b_n / (a_n + b_n))
    else:
        # Exclusive star chain: pi proportional to (1, a_s/b_s, a_n/b_n).
        weights = np.array([1.0, a_s / b_s, a_n / b_n])
        occupied = weights[1:].sum() / weights.sum()
    return k_y * occupied / GAMMA


def simulate(promoter, rates=RATES, n=40_000, seed=0):
    a_s, b_s, a_n, b_n, k_y = rates
    rng = np.random.default_rng(seed)
    return simulate_counts(rng, a_s, a_n, b_s, b_n, k_y, promoter=promoter,
                           size=n)


# ----------------------------------------------------------------------
# promoter topologies
# ----------------------------------------------------------------------

@pytest.mark.parametrize("builder", [heterodimer_chain, monomer_chain])
def test_stationary_distribution_solves_the_generator(builder):
    """pi Q = 0: the law each cell starts from really is stationary."""
    starts, targets, rates, act, pi = builder(*RATES[:4])
    n_states = act.size

    Q = np.zeros((n_states, n_states))
    for state in range(n_states):
        for j in range(starts[state], starts[state + 1]):
            Q[state, targets[j]] += rates[j]
            Q[state, state] -= rates[j]

    assert pi.sum() == pytest.approx(1.0)
    assert np.all(pi > 0)
    assert np.allclose(pi @ Q, 0.0, atol=1e-12)


def test_monomer_binding_is_exclusive():
    """Three states, no doubly bound one, and both bound states transcribe."""
    starts, targets, rates, act, pi = monomer_chain(*RATES[:4])
    assert act.size == 3
    assert list(act) == [0.0, 1.0, 1.0]
    # Every jump either leaves or returns to the empty state: no S -> N.
    for state in (1, 2):
        assert list(targets[starts[state]:starts[state + 1]]) == [0]


def test_topologies_differ_in_occupancy():
    """Same rates, different mean: sharing a site costs occupancy.

    With alpha = beta = 1 on both factors the heterodimer is silent only 1/4 of
    the time, the exclusive monomer 1/3 of it, so the models are distinguishable
    from the counts rather than being a reparameterisation of each other.
    """
    rates = (1.0, 1.0, 1.0, 1.0, 20.0)
    assert exact_mean("heterodimer", *rates) == pytest.approx(15.0)
    assert exact_mean("monomer", *rates) == pytest.approx(40.0 / 3.0)


# ----------------------------------------------------------------------
# simulator
# ----------------------------------------------------------------------

@pytest.mark.parametrize("promoter", sorted(PROMOTERS))
def test_simulator_reproduces_the_stationary_mean(promoter):
    """Every draw is an independent stationary sample, so the mean must land.

    The old simulator recorded ten correlated points along one burning-in
    trajectory and missed this by a wide margin.
    """
    y = simulate(promoter)
    expected = exact_mean(promoter, *RATES)
    standard_error = y.std() / np.sqrt(y.size)
    assert abs(y.mean() - expected) < 4 * standard_error


@pytest.mark.parametrize("promoter", sorted(PROMOTERS))
def test_simulator_is_stationary_in_the_tail_too(promoter):
    """Splitting the sample in half must not shift it: there is no burn-in."""
    y = simulate(promoter, n=40_000)
    first, second = y[: y.size // 2], y[y.size // 2:]
    pooled = np.sqrt(first.var() / first.size + second.var() / second.size)
    assert abs(first.mean() - second.mean()) < 4 * pooled


@pytest.mark.parametrize("promoter", sorted(PROMOTERS))
def test_simulator_is_invariant_under_rescaling_time(promoter):
    """Only ratios to gamma are identifiable, so scaling every rate together
    must leave the counts alone. This is why GAMMA is pinned at 1.

    Regression: KERNEL_HORIZON counts mRNA lifetimes but was passed through as
    absolute time, so for gamma < 1 the kernel was truncated early -- 22% of
    the integral lost at gamma = 0.1, while gamma = 1 stayed correct and hid it.
    """
    a_s, b_s, a_n, b_n, k_y = RATES
    expected = exact_mean(promoter, a_s, b_s, a_n, b_n, k_y)
    for scale in (0.1, 1.0, 10.0):
        rng = np.random.default_rng(5)
        y = simulate_counts(rng, a_s * scale, a_n * scale, b_s * scale,
                            b_n * scale, k_y * scale, promoter=promoter,
                            size=30_000, gamma=scale)
        assert abs(y.mean() - expected) < 0.05 * expected


@pytest.mark.parametrize("promoter", sorted(PROMOTERS))
def test_simulator_honours_size_and_returns_counts(promoter):
    y = simulate(promoter, n=137)
    assert y.shape == (137,)
    assert y.dtype == np.float64
    assert np.all(y >= 0)
    assert np.all(y == np.rint(y))


@pytest.mark.parametrize("promoter", sorted(PROMOTERS))
def test_simulator_is_reproducible(promoter):
    """sample_smc reruns proposals across processes; a seed must pin them."""
    assert np.array_equal(simulate(promoter, n=500, seed=3),
                          simulate(promoter, n=500, seed=3))
    assert not np.array_equal(simulate(promoter, n=500, seed=3),
                              simulate(promoter, n=500, seed=4))


@pytest.mark.parametrize("promoter", sorted(PROMOTERS))
@pytest.mark.parametrize("bad", [
    (0.0, 0.06, 0.0, 0.24, 20.0),    # neither factor can bind
    (1.0, 0.0, 0.15, 0.24, 20.0),    # beta_s = 0: bound forever, no stationarity
    (-1.0, 0.06, 0.15, 0.24, 20.0),  # negative rate
    (1.0, 0.06, 0.15, 0.24, np.nan),
])
def test_out_of_support_parameters_score_as_a_bad_fit(promoter, bad):
    """The sampler must be able to reject them, not crash on them."""
    y = simulate(promoter, rates=bad, n=64)
    assert y.shape == (64,)
    assert np.all(y == 0.0)


@pytest.mark.parametrize("promoter", sorted(PROMOTERS))
def test_simulator_pickles(promoter):
    """sample_smc cloudpickles the model out to worker processes."""
    fn = MODELS[promoter].simulator
    restored = cloudpickle.loads(cloudpickle.dumps(fn))
    rng = np.random.default_rng(7)
    a_s, b_s, a_n, b_n, k_y = RATES
    assert restored(rng, a_s, a_n, b_s, b_n, k_y, size=32).shape == (32,)


# ----------------------------------------------------------------------
# PyMC models
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(MODELS))
def test_off_rates_are_pinned_by_default(name):
    """Only the on-rates and k_y are free; the betas are constants."""
    built = MODELS[name]().build_model(y=COUNTS)
    free = {v.name for v in built.free_RVs}
    assert free == {"alpha_s", "alpha_n", "k_y"}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_pinned_values_reach_the_simulator(name):
    """A pinned rate must be the stated number, not merely absent from the trace."""
    built = MODELS[name]().build_model(y=COUNTS)
    (observed,) = built.observed_RVs
    # Simulator params follow the order (alpha_s, alpha_n, beta_s, beta_n, k_y).
    params = observed.owner.op.dist_params(observed.owner)
    beta_s, beta_n = params[2], params[3]
    # Broadcast across the observations, so check every entry, not just one.
    assert np.allclose(np.asarray(beta_s.eval()), FIXED_BETA_S)
    assert np.allclose(np.asarray(beta_n.eval()), FIXED_BETA_N)
    assert (FIXED_BETA_S, FIXED_BETA_N) == (0.04, 0.26)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_clearing_the_pin_restores_the_prior(name):
    """The pinning has to be reversible, or the four-rate model is unreachable."""
    cfg = {**MODELS[name].get_default_model_config(),
           "beta_s_fixed": None, "beta_n_fixed": None}
    built = MODELS[name](model_config=cfg).build_model(y=COUNTS)
    free = {v.name for v in built.free_RVs}
    assert free == {"alpha_s", "alpha_n", "beta_s", "beta_n", "k_y"}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_observed_node_is_a_simulator(name):
    """No Simulator, no ABC: sample_smc would fall back to a real likelihood."""
    built = MODELS[name]().build_model(y=COUNTS)
    (observed,) = built.observed_RVs
    assert isinstance(observed.owner.op, pm.distributions.simulator.SimulatorRV)
    assert built.rvs_to_values[observed].eval().shape == COUNTS.shape


@pytest.mark.parametrize("name", sorted(MODELS))
def test_abc_logp_is_finite_and_varies_with_the_parameters(name):
    """The discrepancy has to actually discriminate between proposals."""
    model = MODELS[name]()
    built = model.build_model(y=np.repeat(COUNTS, 8))
    logp = built.compile_logp()

    point = built.initial_point()
    near = float(logp(point))
    far = float(logp({k: v + 4.0 for k, v in point.items()}))  # priors are logged
    assert np.isfinite(near)
    assert near > far


@pytest.mark.parametrize("name", sorted(MODELS))
def test_epsilon_is_resolved_from_the_data_and_recorded(name):
    """A tolerance in molecules, scaled to the gene rather than hard-coded."""
    model = MODELS[name]()
    assert model.model_config["epsilon"] is None

    counts = np.repeat(COUNTS, 4)
    model.build_model(y=counts)
    epsilon = model.model_config["epsilon"]
    assert epsilon == pytest.approx(0.25 * counts.std())

    # Once resolved it is reused, so refitting cannot silently change the target.
    model.build_model(y=counts * 10)
    assert model.model_config["epsilon"] == pytest.approx(epsilon)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_explicit_epsilon_is_respected(name):
    model = MODELS[name](model_config={**MODELS[name].get_default_model_config(),
                                       "epsilon": 3.0})
    model.build_model(y=COUNTS)
    assert model.model_config["epsilon"] == 3.0


def test_build_model_requires_observed_counts():
    with pytest.raises(ValueError, match="observed counts are required"):
        MODELS["heterodimer"]().build_model()
