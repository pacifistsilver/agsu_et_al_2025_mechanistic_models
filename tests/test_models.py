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
import pytensor  # noqa: E402
import pytensor.tensor as pt  # noqa: E402

from stochtf.cme import stationary as cme_stationary  # noqa: E402
from stochtf.inference.models import (  # noqa: E402
    FIXED_BETA_N,
    FIXED_BETA_S,
    GAMMA,
    MODELS,
    PROMOTERS,
    JointStationaryLogLike,
    StationaryLogLike,
    chain_generator,
    heterodimer_chain,
    log_likelihood,
    model_probabilities,
    monomer_chain,
    simulate_counts,
    simulate_joint,
)

#: alpha_s, beta_s, alpha_n, beta_n, k_y -- bursty, like the observed counts.
RATES = (1.0, 0.06, 0.15, 0.24, 20.0)
COUNTS = np.array([0, 1, 5, 30, 42, 17])

#: Models with one fixed topology. "joint" carries an extra
#: model_index RV and a shifted parameter order, so tests that
#: assume a single promoter use this list instead.
SINGLE_MODELS = ["heterodimer", "monomer"]


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

    Args:
        promoter: Promoter topology under test.
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
    """Checks that scaling every rate together leaves the counts alone.

    Only ratios to gamma are identifiable, which is why GAMMA is pinned at 1.

    Regression: KERNEL_HORIZON counts mRNA lifetimes but was passed through as
    absolute time, so for gamma < 1 the kernel was truncated early -- 22% of
    the integral lost at gamma = 0.1, while gamma = 1 stayed correct and hid it.

    Args:
        promoter: Promoter topology under test.
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
    (1.0, 0.0, 0.15, 0.24, 20.0),   # beta_s = 0: bound forever, not stationary
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

@pytest.mark.parametrize("name", SINGLE_MODELS)
def test_off_rates_are_pinned_by_default(name):
    """Only the on-rates and k_y are free; the betas are constants."""
    built = MODELS[name]().build_model(y=COUNTS)
    free = {v.name for v in built.free_RVs}
    assert free == {"alpha_s", "alpha_n", "k_y"}


@pytest.mark.parametrize("name", SINGLE_MODELS)
def test_pinned_values_reach_the_simulator(name):
    """A pinned rate must be the stated number, not just absent from it."""
    cfg = {**MODELS[name].get_default_model_config(), "method": "abc"}
    built = MODELS[name](model_config=cfg).build_model(y=COUNTS)
    (observed,) = built.observed_RVs
    # Simulator params follow the order (alpha_s, alpha_n, beta_s, beta_n, k_y).
    params = observed.owner.op.dist_params(observed.owner)
    beta_s, beta_n = params[2], params[3]
    # Broadcast across the observations, so check every entry, not just one.
    assert np.allclose(np.asarray(beta_s.eval()), FIXED_BETA_S)
    assert np.allclose(np.asarray(beta_n.eval()), FIXED_BETA_N)
    assert (FIXED_BETA_S, FIXED_BETA_N) == (0.04, 0.26)


@pytest.mark.parametrize("name", SINGLE_MODELS)
def test_pinned_values_reach_the_exact_likelihood(name):
    """The model's own score must equal the likelihood at the pinned betas.

    Stronger than reading the constants back out of the graph: it pins down
    what the sampler is actually being handed.

    Args:
        promoter: Promoter topology under test.
    """
    built = MODELS[name]().build_model(y=COUNTS)
    point = built.initial_point()
    alpha_s = float(np.exp(point["alpha_s_log__"]))
    alpha_n = float(np.exp(point["alpha_n_log__"]))
    k_y = float(np.exp(point["k_y_log__"]))

    got = float(built.compile_fn(built.datalogp)(point))
    expected = log_likelihood(COUNTS, alpha_s, alpha_n, FIXED_BETA_S,
                              FIXED_BETA_N, k_y, promoter=name)
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("name", SINGLE_MODELS)
def test_clearing_the_pin_restores_the_prior(name):
    """Pinning has to be reversible, or the four-rate model is unreachable."""
    cfg = {**MODELS[name].get_default_model_config(),
           "beta_s_fixed": None, "beta_n_fixed": None}
    built = MODELS[name](model_config=cfg).build_model(y=COUNTS)
    free = {v.name for v in built.free_RVs}
    assert free == {"alpha_s", "alpha_n", "beta_s", "beta_n", "k_y"}


def abc_config(name, **overrides):
    return {**MODELS[name].get_default_model_config(), "method": "abc",
            **overrides}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_exact_is_the_default_method(name):
    """Scoring the projected distribution beats an epsilon ball, so it leads."""
    assert MODELS[name].get_default_model_config()["method"] == "exact"


@pytest.mark.parametrize("name", sorted(MODELS))
def test_abc_method_puts_a_simulator_in_the_graph(name):
    """No Simulator, no ABC: sample_smc would fall back to a real likelihood."""
    built = MODELS[name](model_config=abc_config(name)).build_model(y=COUNTS)
    (observed,) = built.observed_RVs
    assert isinstance(observed.owner.op, pm.distributions.simulator.SimulatorRV)
    assert built.rvs_to_values[observed].eval().shape == COUNTS.shape


@pytest.mark.parametrize("name", sorted(MODELS))
def test_exact_method_scores_through_a_potential(name):
    """SMC tempers model.datalogp, so the term must land there."""
    built = MODELS[name]().build_model(y=COUNTS)
    assert not built.observed_RVs
    data_logp = float(built.compile_fn(built.datalogp)(built.initial_point()))
    assert np.isfinite(data_logp) and data_logp != 0.0


@pytest.mark.parametrize("method", ["exact", "abc"])
@pytest.mark.parametrize("name", sorted(MODELS))
def test_logp_is_finite_and_varies_with_the_parameters(name, method):
    """Either way the score has to actually discriminate between proposals."""
    cfg = abc_config(name) if method == "abc" else None
    built = MODELS[name](model_config=cfg).build_model(y=np.repeat(COUNTS, 8))
    logp = built.compile_logp()

    point = built.initial_point()
    near = float(logp(point))
    # Priors are logged.
    far = float(logp({k: v + 4.0 for k, v in point.items()}))
    assert np.isfinite(near)
    assert near > far


@pytest.mark.parametrize("name", sorted(MODELS))
def test_epsilon_is_resolved_from_the_data_and_recorded(name):
    """A tolerance in molecules, scaled to the gene rather than hard-coded."""
    model = MODELS[name](model_config=abc_config(name))
    assert model.model_config["epsilon"] is None

    counts = np.repeat(COUNTS, 4)
    model.build_model(y=counts)
    epsilon = model.model_config["epsilon"]
    assert epsilon == pytest.approx(0.25 * counts.std())

    # Once resolved it is reused, so refitting cannot change the target.
    model.build_model(y=counts * 10)
    assert model.model_config["epsilon"] == pytest.approx(epsilon)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_exact_method_leaves_epsilon_unset(name):
    """Recording a tolerance nothing read would misrepresent the saved fit."""
    model = MODELS[name]()
    model.build_model(y=COUNTS)
    assert model.model_config["epsilon"] is None


@pytest.mark.parametrize("name", sorted(MODELS))
def test_explicit_epsilon_is_respected(name):
    model = MODELS[name](model_config=abc_config(name, epsilon=3.0))
    model.build_model(y=COUNTS)
    assert model.model_config["epsilon"] == 3.0


# ----------------------------------------------------------------------
# exact likelihood by finite state projection
# ----------------------------------------------------------------------

@pytest.mark.parametrize("promoter", sorted(PROMOTERS))
def test_likelihood_matches_the_projected_distribution(promoter):
    """The log-likelihood is just the pmf summed over the observations."""
    a_s, b_s, a_n, b_n, k_y = RATES
    Q, act = chain_generator(promoter, a_s, b_s, a_n, b_n)
    p = cme_stationary.stationary_pmf(Q, act, k_y, GAMMA)
    expected = float(np.sum(np.log(p[COUNTS])))
    got = log_likelihood(COUNTS, a_s, a_n, b_s, b_n, k_y, promoter=promoter)
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("promoter", sorted(PROMOTERS))
def test_likelihood_peaks_near_the_generating_parameters(promoter):
    """Perturbing any rate away from the truth must lower the score."""
    a_s, b_s, a_n, b_n, k_y = RATES
    rng = np.random.default_rng(1)
    counts = simulate_counts(rng, a_s, a_n, b_s, b_n, k_y, promoter=promoter,
                             size=4000)
    base = log_likelihood(counts, a_s, a_n, b_s, b_n, k_y, promoter=promoter)
    for index, factor in ((0, 3.0), (1, 3.0), (4, 1.5), (4, 0.6)):
        rates = [a_s, a_n, b_s, b_n, k_y]
        rates[index] *= factor
        assert log_likelihood(counts, *rates, promoter=promoter) < base


@pytest.mark.parametrize("promoter", sorted(PROMOTERS))
@pytest.mark.parametrize("bad", [
    (1.0, 0.15, 0.0, 0.24, 20.0),     # beta_s = 0: never releases
    (-1.0, 0.15, 0.06, 0.24, 20.0),   # negative rate
    (1.0, 0.15, 0.06, 0.24, np.nan),
])
def test_likelihood_rejects_out_of_support_parameters(promoter, bad):
    assert log_likelihood(COUNTS, *bad, promoter=promoter) == -np.inf


@pytest.mark.parametrize("promoter", sorted(PROMOTERS))
def test_logp_op_pickles_with_its_data(promoter):
    """sample_smc cloudpickles the kernel out to worker processes."""
    op = StationaryLogLike(COUNTS, promoter)
    restored = cloudpickle.loads(cloudpickle.dumps(op))
    assert np.array_equal(restored.counts, op.counts)
    assert restored.promoter == promoter


def test_distinct_datasets_never_compare_equal():
    """No __props__, so graph merging cannot swap one dataset for another."""
    assert (StationaryLogLike(COUNTS, "heterodimer")
            != StationaryLogLike(COUNTS + 1, "heterodimer"))
    assert (StationaryLogLike(COUNTS, "heterodimer")
            != StationaryLogLike(COUNTS, "monomer"))


def test_build_model_requires_observed_counts():
    with pytest.raises(ValueError, match="observed counts are required"):
        MODELS["heterodimer"]().build_model()


# ----------------------------------------------------------------------
# fitting both topologies at once
# ----------------------------------------------------------------------

def test_joint_model_adds_a_discrete_topology_index():
    """The topology becomes a parameter, so SMC can move between the two."""
    built = MODELS["joint"]().build_model(y=COUNTS)
    free = {v.name for v in built.free_RVs}
    assert free == {"model_index", "alpha_s", "alpha_n", "k_y"}
    index = next(v for v in built.free_RVs if v.name == "model_index")
    assert index.dtype.startswith("int")


@pytest.mark.parametrize("index,promoter", [(0, "monomer"),
                                            (1, "heterodimer")])
def test_joint_likelihood_matches_the_selected_topology(index, promoter):
    """Each index value must score exactly what that single model would.

    Without this the index could be sampled perfectly and still be comparing
    the wrong pair of distributions.

    Args:
        index: Model index under test.
    """
    a_s, b_s, a_n, b_n, k_y = RATES
    op = JointStationaryLogLike(COUNTS)
    inputs = [pt.dscalar(n) for n in ("m", "a_s", "a_n", "b_s", "b_n", "k_y")]
    fn = pytensor.function(inputs, op(*inputs))

    got = float(fn(index, a_s, a_n, b_s, b_n, k_y))
    expected = log_likelihood(COUNTS, a_s, a_n, b_s, b_n, k_y,
                              promoter=promoter)
    assert got == pytest.approx(expected, rel=1e-12)


def test_joint_index_prior_is_configurable():
    """Equal prior odds by default, so the posterior reads as evidence."""
    assert MODELS["joint"].get_default_model_config()["heterodimer_prior"] == 0.5

    cfg = {**MODELS["joint"].get_default_model_config(),
           "heterodimer_prior": 0.25}
    built = MODELS["joint"](model_config=cfg).build_model(y=COUNTS)
    index = next(v for v in built.free_RVs if v.name == "model_index")
    assert float(index.owner.inputs[-1].eval()) == pytest.approx(0.25)


def test_model_probabilities_divides_out_the_prior():
    """The Bayes factor must report evidence, not the prior it started from."""
    class FakePosterior(dict):
        pass

    idata = {"posterior": {"model_index": np.array([[1, 1, 1, 0]])}}
    probs = model_probabilities(idata, heterodimer_prior=0.5)
    assert probs["heterodimer"] == pytest.approx(0.75)
    assert probs["monomer"] == pytest.approx(0.25)
    assert probs["bayes_factor_het_over_mono"] == pytest.approx(3.0)

    # Same posterior, but reached from a prior that already favoured it: the
    # data then contributed less, and the factor has to say so.
    skewed = model_probabilities(idata, heterodimer_prior=0.75)
    assert skewed["bayes_factor_het_over_mono"] == pytest.approx(1.0)


def test_joint_simulator_dispatches_on_the_index():
    """The ABC path must switch topology too, not silently use one of them."""
    a_s, b_s, a_n, b_n, k_y = RATES
    for index, promoter in ((0, "monomer"), (1, "heterodimer")):
        joint = simulate_joint(np.random.default_rng(11), index, a_s, a_n,
                               b_s, b_n, k_y, size=4000)
        direct = simulate_counts(np.random.default_rng(11), a_s, a_n, b_s,
                                 b_n, k_y, promoter=promoter, size=4000)
        assert np.array_equal(joint, direct)
