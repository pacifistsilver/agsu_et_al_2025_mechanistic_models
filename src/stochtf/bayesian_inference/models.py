"""Bayesian models for the monomer and heterodimer promoters.

Both are sampled with ``pm.sample_smc``. They differ in how a proposal is
scored, selected by ``model_config["method"]``:

``exact`` (default)
    The stationary distribution P(y | theta) is computed numerically by finite
    state projection (:mod:`stochtf.cme.stationary`) and the counts are scored
    against the whole of it. There is no tolerance to tune and no simulation
    noise, so the sampler targets the true posterior.

``abc``
    The likelihood-free route: counts are simulated and compared to the data
    through a discrepancy, using ``pm.Simulator``, which with ``sample_smc``
    is ABC-SMC. Kept because it needs only a forward simulator, so it still
    applies to variants the projection cannot reach -- and because agreement
    between the two is a good check on both.

Projection replaced simulation as the default because it is both exact and,
at these state-space sizes, no more expensive: the promoter has three or four
states and the count grid is a few hundred wide, so one solve costs a few
milliseconds -- about what simulating a dataset costs, but without the noise
that forces epsilon to stay wide.

The two promoters
-----------------
Both are small continuous-time chains driving one birth-death mRNA species, and
both are parameterised the same way -- (alpha_s, beta_s) for SOX2, (alpha_n,
beta_n) for NANOG, k_y for transcription -- so their fits are comparable. They
differ only in topology, which is the hypothesis under test:

``heterodimer``
    Two independent sites, states (sigma_s, sigma_n) in {0,1}^2 ordered 00, 10,
    01, 11 as in :mod:`stochtf.analytical.heterodimer`. Site s flips 0->1 at
    alpha_s and 1->0 at beta_s, site n likewise, and mRNA is made at k_y
    whenever *at least one* site is bound.

``monomer``
    One site the two factors compete for, S <- 0 -> N: binding is exclusive, so
    the states are empty / SOX2-bound / NANOG-bound and there is no doubly
    bound state. mRNA is made at k_y whenever the site is occupied, which is
    the ``k_y (n_b + s_b)`` emission of :mod:`stochtf.analytical.monomer` --
    the two terms cannot both be one here.

The simulator (ABC method only)
-------------------------------
Conditional on the promoter path sigma(t), transcription is an inhomogeneous
Poisson process and each mRNA survives to the observation time with probability
exp(-gamma * age), so the stationary count is exactly

    y | sigma ~ Poisson( k_y * integral_0^inf act(sigma(s)) exp(-gamma s) ds )

with s the age (time before observation). Only the promoter needs simulating;
the mRNA layer is integrated out. Both chains are reversible -- one is a pair
of two-state chains, the other a star -- so the time-reversed promoter has the
same law as the forward one, and starting each cell from the stationary
occupancy then running forward in age samples the true stationary distribution.
Between switches the activity is constant and the kernel integrates in closed
form, so one iteration per promoter switch is all it costs.

That matters for three reasons, all of which the previous ABC layer got wrong:

Stationarity
    The old simulator started every cell empty at t = 0 and recorded 10 counts
    along *one* trajectory at t_max/10 ... t_max. Those draws are correlated
    with each other and, at the settings used, not yet stationary -- the first
    was taken half an mRNA lifetime in. Here every draw is an independent
    stationary sample and there is no burn-in to choose.

Cost
    A full Gillespie run fires ~2 * mean * horizon reactions per cell; at
    esrrb's mean of 251 over 15 lifetimes that is ~7500 reactions. Simulating
    only the promoter costs a handful of iterations per cell in the bursty
    regime, which is what makes ABC over ~10^5 proposals feasible.

Transcription rate
    ``k_y`` and ``gamma_y`` used to be *default arguments* of the simulator
    that ``pm.Simulator`` never overrode, pinning k_y/gamma at 2 and capping
    the model mean at 2 against observed means of 30-250. k_y is now inferred.

Identifiability
---------------
Stationary counts determine only the rates *relative to* the mRNA degradation
rate, so gamma is fixed at 1: the time unit is one mRNA lifetime and every
other rate is inferred in those units.

They also cannot determine four switching rates at once, so the two off-rates
are pinned by default -- beta_s = 0.04 and beta_n = 0.26, the pair
single-molecule tracking measures directly -- leaving alpha_s, alpha_n and k_y
free. Clearing ``beta_s_fixed``/``beta_n_fixed`` in ``model_config`` restores
them as inferred parameters, at the cost of posteriors that are mostly prior.

Pinning the off-rates also breaks the factor-exchange symmetry that would
otherwise make the marginals ambiguous: with beta_s != beta_n the two sites are
no longer interchangeable, so alpha_s and alpha_n mean what their names say.
See :mod:`stochtf.inference.identifiability` for what is and is not determined.

Discrepancy (ABC method only)
-----------------------------
The default is the sorted count vector under a Gaussian kernel, i.e. the 1D
2-Wasserstein distance between the empirical distributions. It uses the whole
distribution rather than the two-number ``[mean, fano]`` summary the old
version accepted on, so far less information is thrown away. ``epsilon`` is a
tolerance in molecules and defaults to a quarter of the observed standard
deviation, which scales across genes; the old hard-coded ``epsilon = 2.0`` did
not. The ABC posterior remains broader than the true posterior by an amount set
by epsilon and by simulator noise -- which is the reason the exact method is
the default.

Note that :mod:`stochtf.inference.likelihood` computes the same quantity for
the *independent-site* gates via ``stochtf.analytical.pgf``. That route cannot
express the exclusive monomer, whose promoter is not a product of two
two-state sites, which is why :func:`log_likelihood` here goes through the
projection instead.
"""

import json
from typing import Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numba import njit
from pymc_extras.model_builder import ModelBuilder
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op

from stochtf import constants
from stochtf.cme import stationary as cme_stationary
from stochtf.inference.likelihood import MIN_PROB, prepare_counts

#: Time unit: rates are expressed relative to the mRNA degradation rate, which
#: stationary count data cannot identify separately.
GAMMA = 1.0

#: How far back in age the exp(-gamma s) kernel is followed, in mRNA lifetimes.
#: The neglected tail is exp(-15) ~ 3e-7 of the integral, far below the Monte
#: Carlo noise of the simulation itself.
KERNEL_HORIZON = 15.0

#: Promoter switches per cell before the rest of the kernel is closed off with
#: the occupancy-averaged activity. Reachable only when switching is more than
#: ~25x faster than degradation, and that is exactly the regime where the
#: promoter self-averages, so what the closure discards is a fluctuation the
#: data could not resolve anyway. Without it one absurd prior draw could stall
#: a whole SMC stage.
MAX_SWITCHES = 400

#: Default ABC tolerance as a fraction of the observed standard deviation.
EPSILON_SCALE = 0.25

#: Floor on epsilon, so degenerate data cannot produce a zero-width kernel.
MIN_EPSILON = 1e-3

#: Off-rates held fixed rather than inferred, in units of gamma. Stationary
#: counts cannot determine four switching rates at once (see
#: :mod:`stochtf.inference.identifiability`), so pinning the two dissociation
#: rates -- the pair that single-molecule tracking measures directly -- leaves
#: the on-rates and k_y to be estimated from a problem the data can support.
#: Set either to None in ``model_config`` to infer it with its log-normal prior
#: instead.
FIXED_BETA_S = constants.BETA_S
FIXED_BETA_N = constants.BETA_N

#: Log-normal priors, in units of gamma. Switching rates are centred on 1 (one
#: event per mRNA lifetime) and spread over four orders of magnitude either
#: side, covering both the bursty regime the observed Fano factors imply and
#: the fast-switching limit. k_y is centred on 20, the order needed to reach
#: the observed means of 30-250. The beta priors apply only when the
#: corresponding ``*_fixed`` entry is cleared.
DEFAULT_MODEL_CONFIG: Dict = {
    "alpha_s_mu": -1.0, "alpha_s_sigma": 2.0,
    "alpha_n_mu": -2.0, "alpha_n_sigma": 2.0,
    "beta_s_mu": 0.0, "beta_s_sigma": 2.0,
    "beta_n_mu": 0.0, "beta_n_sigma": 2.0,
    "k_y_mu": 3.0, "k_y_sigma": 1.5,
    # Pinned rates. A value here replaces the prior above with a constant.
    "beta_s_fixed": FIXED_BETA_S,
    "beta_n_fixed": FIXED_BETA_N,
    # "exact" scores the whole distribution from the finite state projection;
    # "abc" falls back to simulating counts and comparing them through the
    # epsilon kernel below. Both are sampled with pm.sample_smc.
    "method": "exact",
    # ABC kernel. epsilon=None resolves to epsilon_scale * std(observed) at
    # build time and is written back here, so the value used is recorded.
    # sum_stat and distance go straight to pm.Simulator; keep them strings so
    # model_config stays JSON-serialisable for save().
    "epsilon": None,
    "epsilon_scale": EPSILON_SCALE,
    "sum_stat": "sort",
    "distance": "gaussian",
}

DEFAULT_SAMPLER_CONFIG: Dict = {
    "draws": 1000,
    "chains": 4,
}


# ----------------------------------------------------------------------
# promoter topologies
# ----------------------------------------------------------------------
#
# Each builder returns the jump chain in flat form, so one kernel serves both:
#   starts   -- offsets into targets/rates, one per state plus a final bound
#   targets  -- destination state of each jump
#   rates    -- rate of each jump
#   act      -- transcription activity per state, in units of k_y
#   pi       -- stationary distribution over states

def heterodimer_chain(a_s, b_s, a_n, b_n):
    """Two independent sites; mRNA whenever either is bound (states 10/01/11).

    State order 00, 10, 01, 11 matches :func:`stochtf.analytical.pgf.promoter_generator`,
    so the two can be compared directly.
    """
    starts = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    targets = np.array([1, 2,   0, 3,   0, 3,   1, 2], dtype=np.int64)
    rates = np.array([a_s, a_n,  b_s, a_n,  b_n, a_s,  b_n, b_s], dtype=np.float64)
    act = np.array([0.0, 1.0, 1.0, 1.0])

    # The sites are independent, so the stationary law is a product.
    p_s, p_n = a_s / (a_s + b_s), a_n / (a_n + b_n)
    pi = np.array([(1 - p_s) * (1 - p_n), p_s * (1 - p_n),
                   (1 - p_s) * p_n, p_s * p_n])
    return starts, targets, rates, act, pi


def monomer_chain(a_s, b_s, a_n, b_n):
    """One site, exclusive occupancy: S <- 0 -> N, no doubly bound state.

    States are empty, SOX2-bound, NANOG-bound; mRNA is made whenever the site
    is occupied. Detailed balance on this star gives pi proportional to
    (1, a_s/b_s, a_n/b_n).
    """
    starts = np.array([0, 2, 3, 4], dtype=np.int64)
    targets = np.array([1, 2,   0,   0], dtype=np.int64)
    rates = np.array([a_s, a_n,  b_s,  b_n], dtype=np.float64)
    act = np.array([0.0, 1.0, 1.0])

    weights = np.array([1.0, a_s / b_s, a_n / b_n])
    return starts, targets, rates, act, weights / weights.sum()


#: Promoter name -> jump-chain builder. The keys are the model names used
#: throughout the inference layer.
PROMOTERS = {"monomer": monomer_chain, "heterodimer": heterodimer_chain}


# ----------------------------------------------------------------------
# simulator
# ----------------------------------------------------------------------

@njit(cache=True)
def _seed_numba(seed):
    """Seed numba's own generator, which is separate from NumPy's."""
    np.random.seed(seed)


@njit(cache=True)
def _simulate_counts(starts, targets, rates, act, pi, k_y, gamma, n_cells,
                     horizon, max_switches):
    """``n_cells`` independent stationary counts from a promoter jump chain.

    Each cell draws its promoter state from ``pi``, walks the chain forward in
    age, accumulates the kernel-weighted activity in closed form between
    switches, and draws one Poisson count from the total.
    """
    n_states = act.size

    exit_rates = np.zeros(n_states)
    for s in range(n_states):
        for j in range(starts[s], starts[s + 1]):
            exit_rates[s] += rates[j]

    cumulative_pi = np.cumsum(pi)
    mean_act = 0.0
    for s in range(n_states):
        mean_act += pi[s] * act[s]

    out = np.empty(n_cells, dtype=np.float64)

    for i in range(n_cells):
        # Start from stationarity, so the walk in age is stationary throughout.
        u = np.random.random()
        state = n_states - 1
        for s in range(n_states):
            if u < cumulative_pi[s]:
                state = s
                break

        integral = 0.0
        age = 0.0
        decay = 1.0  # exp(-gamma * age)
        switches = 0

        while age < horizon:
            rate = exit_rates[state]
            if rate <= 0.0:
                next_age = horizon
            else:
                next_age = age - np.log(np.random.random()) / rate
                if not (next_age < horizon):  # also catches the log(0) = -inf draw
                    next_age = horizon

            next_decay = np.exp(-gamma * next_age)
            # Activity is constant between switches, so the kernel integrates
            # exactly: int exp(-gamma s) ds = (decay - next_decay) / gamma.
            integral += act[state] * (decay - next_decay) / gamma
            age = next_age
            decay = next_decay

            switches += 1
            if switches >= max_switches:
                # Fast switching: close off the rest self-averaged.
                integral += mean_act * decay / gamma
                break

            if rate > 0.0 and age < horizon:
                threshold = np.random.random() * rate
                acc = 0.0
                for j in range(starts[state], starts[state + 1]):
                    acc += rates[j]
                    if threshold < acc:
                        state = targets[j]
                        break

        lam = k_y * integral
        if lam > 0.0 and np.isfinite(lam):
            out[i] = np.random.poisson(lam)
        else:
            out[i] = 0.0

    return out


def _n_from_size(size):
    """Number of cells ``pm.Simulator`` is asking for."""
    if size is None:
        return 1
    return int(np.prod(np.atleast_1d(size)))


def _seed_from(rng):
    """Take a seed from PyMC's generator so simulations are reproducible.

    The kernel is numba-compiled and uses numba's own global generator, which
    a NumPy ``Generator`` cannot drive directly; reseeding it per call is what
    ties the two together.
    """
    if rng is None:
        return int(np.random.randint(0, 2**31 - 1))
    if hasattr(rng, "integers"):
        return int(rng.integers(0, 2**31 - 1))
    return int(rng.randint(0, 2**31 - 1))  # legacy RandomState


def simulate_counts(rng, alpha_s, alpha_n, beta_s, beta_n, k_y,
                    promoter="heterodimer", size=None, gamma=GAMMA):
    """Stationary counts under one of the promoters, one draw per cell.

    Argument order follows the priors declared below (alpha_s, alpha_n, beta_s,
    beta_n), which is *not* the order the analytical routines take.

    Parameters outside the model's support give an all-zero sample rather than
    an exception, so the sampler simply scores them as a bad fit.
    """
    a_s, a_n = float(np.asarray(alpha_s).item()), float(np.asarray(alpha_n).item())
    b_s, b_n = float(np.asarray(beta_s).item()), float(np.asarray(beta_n).item())
    k = float(np.asarray(k_y).item())
    n_cells = _n_from_size(size)

    values = (a_s, a_n, b_s, b_n, k, gamma)
    if (not all(np.isfinite(values)) or min(values) < 0.0 or gamma <= 0.0
            or b_s <= 0.0 or b_n <= 0.0 or a_s + a_n <= 0.0):
        return np.zeros(n_cells, dtype=np.float64)

    starts, targets, rates, act, pi = PROMOTERS[promoter](a_s, b_s, a_n, b_n)
    _seed_numba(_seed_from(rng))
    # KERNEL_HORIZON counts mRNA lifetimes, so the absolute time it corresponds
    # to depends on gamma. Passing it through unscaled would silently truncate
    # the kernel for gamma < 1 -- at gamma = 0.1 it drops 22% of the integral.
    return _simulate_counts(starts, targets, rates, act, pi, k, gamma, n_cells,
                            KERNEL_HORIZON / gamma, MAX_SWITCHES)


# pm.Simulator calls fn(rng, *params, size), so each model needs its promoter
# bound into a module-level function -- module level because sample_smc ships
# the simulator to worker processes.

def simulate_monomer(rng, alpha_s, alpha_n, beta_s, beta_n, k_y, size=None):
    """Exclusive single site, S <- 0 -> N."""
    return simulate_counts(rng, alpha_s, alpha_n, beta_s, beta_n, k_y,
                           promoter="monomer", size=size)


def simulate_heterodimer(rng, alpha_s, alpha_n, beta_s, beta_n, k_y, size=None):
    """Two independent sites; any bound site gives the full rate k_y."""
    return simulate_counts(rng, alpha_s, alpha_n, beta_s, beta_n, k_y,
                           promoter="heterodimer", size=size)


# ----------------------------------------------------------------------
# exact likelihood, by finite state projection
# ----------------------------------------------------------------------

def chain_generator(promoter, a_s, b_s, a_n, b_n):
    """Dense promoter generator and activity vector for one topology."""
    starts, targets, rates, act, _ = PROMOTERS[promoter](a_s, b_s, a_n, b_n)
    jumps = [(state, int(targets[j]), float(rates[j]))
             for state in range(act.size)
             for j in range(starts[state], starts[state + 1])]
    return cme_stationary.generator_from_jumps(act.size, jumps), act


def log_likelihood(counts, alpha_s, alpha_n, beta_s, beta_n, k_y,
                   promoter="heterodimer", gamma=GAMMA):
    """Exact log-likelihood of iid stationary counts, by finite state projection.

    Once :mod:`stochtf.cme.stationary` can produce P(y | theta) numerically for
    either topology, the whole distribution can be scored directly and there is
    no tolerance to tune and no simulation noise -- the sampler targets the true
    posterior rather than an ABC approximation to it.

    Returns ``-inf`` for parameters outside the model's support rather than
    raising, so a sampler can simply reject them.
    """
    y = prepare_counts(counts)
    rates = (alpha_s, alpha_n, beta_s, beta_n, k_y, gamma)
    if not all(np.isfinite(rates)):
        return -np.inf
    if min(alpha_s, alpha_n) < 0 or min(beta_s, beta_n) <= 0 or k_y < 0:
        return -np.inf
    if gamma <= 0 or alpha_s + alpha_n <= 0:
        return -np.inf

    Q, act = chain_generator(promoter, alpha_s, beta_s, alpha_n, beta_n)
    try:
        p = cme_stationary.stationary_pmf(Q, act, k_y, gamma)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return -np.inf
    if not np.all(np.isfinite(p)):
        return -np.inf

    # The projection is sized from the model's own moments, so an observation
    # can still fall beyond it; those get the floor rather than -inf.
    inside = y[y < p.size]
    total = float(np.sum(np.log(np.maximum(p[inside], MIN_PROB))))
    total += float(y.size - inside.size) * np.log(MIN_PROB)
    return total


def simulate_joint(rng, model_index, alpha_s, alpha_n, beta_s, beta_n, k_y,
                   size=None):
    """Simulate under whichever topology the model index currently selects."""
    promoter = "heterodimer" if int(np.asarray(model_index).item()) else "monomer"
    return simulate_counts(rng, alpha_s, alpha_n, beta_s, beta_n, k_y,
                           promoter=promoter, size=size)


class StationaryLogLike(Op):
    """Gradient-free PyTensor Op wrapping :func:`log_likelihood`.

    The likelihood needs a linear solve per proposal, so there is no tractable
    gradient. That is fine: ``pm.sample_smc`` uses gradient-free Metropolis
    kernels within each stage.

    This is a hand-written Op rather than ``pytensor.wrap_py`` because the Op
    has to survive pickling. ``wrap_py`` reduces by *name lookup*, so an Op
    built inside a factory closing over the observed counts cannot be
    unpickled. A normal Op subclass pickles by state instead: the class
    resolves by reference and ``counts``/``promoter`` travel by value, which
    matters because ``sample_smc`` runs chains in separate processes.

    No ``__props__`` is declared, so equality falls back to object identity.
    That is deliberate -- two instances holding different datasets must never
    compare equal, or PyTensor's graph merging could substitute one for the
    other.
    """

    def __init__(self, counts, promoter):
        self.counts = prepare_counts(counts)
        self.promoter = promoter

    def make_node(self, alpha_s, alpha_n, beta_s, beta_n, k_y):
        inputs = [pt.as_tensor_variable(v)
                  for v in (alpha_s, alpha_n, beta_s, beta_n, k_y)]
        return Apply(self, inputs, [pt.scalar(dtype="float64")])

    def perform(self, node, inputs, output_storage):
        alpha_s, alpha_n, beta_s, beta_n, k_y = (float(v) for v in inputs)
        value = log_likelihood(self.counts, alpha_s, alpha_n, beta_s, beta_n,
                               k_y, promoter=self.promoter, gamma=GAMMA)
        output_storage[0][0] = np.asarray(value, dtype="float64")


class JointStationaryLogLike(Op):
    """:class:`StationaryLogLike` with the topology chosen by an extra input.

    Takes the model index first, so one Potential can serve both promoters and
    SMC can move between them within a single run.
    """

    #: Index value -> promoter. 0 is the monomer, 1 the heterodimer.
    PROMOTER_BY_INDEX = ("monomer", "heterodimer")

    def __init__(self, counts):
        self.counts = prepare_counts(counts)

    def make_node(self, model_index, alpha_s, alpha_n, beta_s, beta_n, k_y):
        inputs = [pt.as_tensor_variable(v) for v in
                  (model_index, alpha_s, alpha_n, beta_s, beta_n, k_y)]
        return Apply(self, inputs, [pt.scalar(dtype="float64")])

    def perform(self, node, inputs, output_storage):
        index = int(round(float(inputs[0])))
        index = min(max(index, 0), len(self.PROMOTER_BY_INDEX) - 1)
        alpha_s, alpha_n, beta_s, beta_n, k_y = (float(v) for v in inputs[1:])
        value = log_likelihood(self.counts, alpha_s, alpha_n, beta_s, beta_n,
                               k_y, promoter=self.PROMOTER_BY_INDEX[index],
                               gamma=GAMMA)
        output_storage[0][0] = np.asarray(value, dtype="float64")


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------

class _PromoterModel(ModelBuilder):
    """Shared plumbing. Subclasses set :attr:`promoter` and :attr:`simulator`.

    Both fitting methods are sampled by ``pm.sample_smc``; they differ only in
    how a proposal is scored. ``method="exact"`` evaluates the whole
    distribution by finite state projection, ``method="abc"`` simulates counts
    and compares them through the epsilon kernel.
    """

    #: Key into :data:`PROMOTERS`.
    promoter = None
    #: Module-level simulator taking (rng, alpha_s, alpha_n, beta_s, beta_n,
    #: k_y, size). Only used by the ABC method.
    simulator = None
    version = "3.0"

    def _data_setter(self, X, y=None):
        pass  # ModelBuilder requires this method; X/y are not standard inputs

    def _generate_and_preprocess_model_data(self, X, y=None):
        pass  # ModelBuilder requires this method

    @staticmethod
    def get_default_model_config() -> Dict:
        return dict(DEFAULT_MODEL_CONFIG)

    @staticmethod
    def get_default_sampler_config() -> Dict:
        return dict(DEFAULT_SAMPLER_CONFIG)

    def _prepare(self, y):
        """Flatten the counts and, for ABC only, resolve epsilon against them."""
        if y is None:
            raise ValueError("observed counts are required to build the model")
        counts = np.asarray(y, dtype="float64").ravel()
        if counts.size == 0:
            raise ValueError("observed counts are empty")

        cfg = self.model_config
        if cfg.get("method", "exact") != "abc":
            # Leave epsilon unresolved: recording a tolerance that nothing read
            # would misrepresent the fit in the saved trace.
            return counts, None

        epsilon = cfg.get("epsilon")
        if epsilon is None:
            scale = cfg.get("epsilon_scale", EPSILON_SCALE)
            epsilon = max(float(scale * counts.std()), MIN_EPSILON)
            cfg["epsilon"] = epsilon  # record what was actually used
        return counts, float(epsilon)

    def _rate(self, name):
        """A rate parameter: pinned to a constant, or given its log-normal prior.

        A pinned rate is a constant in the graph rather than a zero-variance
        prior, so it costs the sampler nothing and never appears in the trace.
        """
        cfg = self.model_config
        fixed = cfg.get(f"{name}_fixed")
        if fixed is not None:
            return pt.constant(float(fixed), name=name)
        return pm.LogNormal(name, mu=cfg[f"{name}_mu"],
                            sigma=cfg[f"{name}_sigma"])

    def build_model(self, X=None, y=None, **kwargs):
        counts, epsilon = self._prepare(y)
        cfg = self.model_config

        with pm.Model() as self.model:
            alpha_s = self._rate("alpha_s")
            alpha_n = self._rate("alpha_n")
            beta_s = self._rate("beta_s")
            beta_n = self._rate("beta_n")
            k_y = self._rate("k_y")

            if cfg.get("method", "exact") == "exact":
                logp = StationaryLogLike(counts, self.promoter)
                # Grouped with the data log-probability, so SMC tempers it.
                pm.Potential("likelihood",
                             logp(alpha_s, alpha_n, beta_s, beta_n, k_y))
            else:
                pm.Simulator(
                    "counts",
                    self.simulator,
                    params=(alpha_s, alpha_n, beta_s, beta_n, k_y),
                    sum_stat=cfg.get("sum_stat", "sort"),
                    distance=cfg.get("distance", "gaussian"),
                    epsilon=epsilon,
                    observed=counts,
                )

        return self.model

    def fit(self, data, sampler_config: dict = None, sample_prior: bool = True,
            **kwargs):
        """Fit by ABC-SMC.

        ``pm.sample_smc`` on a model whose only observed node is a
        ``pm.Simulator`` anneals from the prior to the ABC posterior, moving
        particles with an independent Metropolis kernel. There is no gradient
        and no likelihood -- each proposal costs one simulated dataset.
        """
        if sampler_config is None:
            sampler_config = self.sampler_config
        else:
            # Keep what was actually used, so save() records it rather than
            # the defaults the run overrode.
            self.sampler_config = dict(sampler_config)

        self.build_model(y=data)

        with self.model:
            self.idata = pm.sample_smc(
                draws=sampler_config["draws"],
                chains=sampler_config.get("chains", 8),
                **kwargs,
            )
            # Worth keeping for a single fit: the prior predictive shows
            # whether the priors can generate data resembling the observations
            # at all, which is the first thing to check when a fit will not
            # converge. Worth skipping when fitting hundreds of datasets in a
            # sweep, where it is a large fraction of the runtime and nothing
            # reads it.
            if sample_prior:
                prior = pm.sample_prior_predictive(draws=500, random_seed=500)
                if "prior" in prior:
                    self.idata["prior"] = prior["prior"]
                if "prior_predictive" in prior:
                    self.idata["prior_predictive"] = prior["prior_predictive"]

        return self.idata

    def save(self, fname: str):
        """Sanitise the DataTree before delegating to ModelBuilder's saver.

        NetCDF cannot represent object-dtype arrays, and ModelBuilder.load()
        raises KeyError if its metadata attrs are absent, so both are fixed up
        here first.
        """
        if getattr(self, "idata", None) is not None:
            attrs = self.idata.attrs
            if "model_config" not in attrs:
                attrs["model_config"] = json.dumps(self.model_config)
            if "sampler_config" not in attrs:
                attrs["sampler_config"] = json.dumps(
                    self.sampler_config
                    if hasattr(self, "sampler_config")
                    else self.get_default_sampler_config()
                )
            if "version" not in attrs:
                attrs["version"] = self.version
            if "model_type" not in attrs:
                attrs["model_type"] = self.model_type

            for group_name in self.idata.groups:
                group = self.idata[group_name]
                for var_name in list(group.data_vars.keys()):
                    var = group[var_name]
                    if var.dtype == object:
                        try:
                            # Force integer/float mixes into pure floats
                            group[var_name] = var.astype(float)
                        except Exception:
                            # Sequence arrays cannot be coerced; drop them
                            del group[var_name]

        super().save(fname)


class MonomerModel(_PromoterModel):
    """SOX2 and NANOG competing for one site, S <- 0 -> N.

    Binding is exclusive, so occupancy saturates at one factor and the two
    compete: raising alpha_s displaces NANOG. That competition is the whole
    difference from :class:`HeterodimerModel`, and it is what the counts have
    to distinguish.
    """

    model_type = "MonomerModel"
    promoter = "monomer"
    simulator = staticmethod(simulate_monomer)


class HeterodimerModel(_PromoterModel):
    """Two independent sites; any bound site gives the full transcription rate.

    Both sites can be occupied at once, so the promoter is silent only when
    both are free and the ON fraction is 1 - q_s q_n -- higher than the
    monomer's for the same rates.
    """

    model_type = "HeterodimerModel"
    promoter = "heterodimer"
    simulator = staticmethod(simulate_heterodimer)


#: Prior probability that the heterodimer is the right topology. 0.5 gives the
#: two equal prior odds, so the posterior on ``model_index`` reads directly as
#: the weight of evidence.
DEFAULT_HETERODIMER_PRIOR = 0.5


class JointModel(_PromoterModel):
    """Both topologies fitted at once, with a model index SMC samples over.

    Rather than fitting the two promoters separately and comparing afterwards,
    the topology becomes a parameter: ``model_index`` is 0 for the monomer and
    1 for the heterodimer, and every proposal is scored under whichever it
    currently selects. The posterior mean of that index is P(heterodimer |
    data), and the posterior odds against the prior odds is the Bayes factor.
    This is the ABC-SMC model-selection scheme of Toni & Stumpf, and it works
    here without any trans-dimensional machinery for one specific reason: the
    two promoters take the *same* five rates, so the parameter space does not
    change when the index flips and a single set of priors serves both.

    Two things to keep in mind when reading the result.

    The comparison is only as meaningful as its shared footing. Both topologies
    must see the same data, the same priors and the same scoring method, which
    is automatic here but would not survive, say, giving one of them a
    different epsilon.

    And the data may simply not decide. At matched burst frequency and burst
    size the two put nearly the same law on the counts -- their Fano factors
    agree to better than 2% everywhere both are reachable (see
    ``figures/fig_8_fb_sfsp``) -- so a posterior near the prior is the honest
    answer rather than a failure of the sampler. Check
    :func:`model_probabilities` against the prior before reading anything into
    it.
    """

    model_type = "JointModel"
    simulator = staticmethod(simulate_joint)

    @staticmethod
    def get_default_model_config() -> Dict:
        return {**DEFAULT_MODEL_CONFIG,
                "heterodimer_prior": DEFAULT_HETERODIMER_PRIOR}

    def build_model(self, X=None, y=None, **kwargs):
        counts, epsilon = self._prepare(y)
        cfg = self.model_config

        with pm.Model() as self.model:
            # 0 = monomer, 1 = heterodimer. Discrete, and sample_smc moves it.
            model_index = pm.Bernoulli(
                "model_index",
                p=cfg.get("heterodimer_prior", DEFAULT_HETERODIMER_PRIOR))

            alpha_s = self._rate("alpha_s")
            alpha_n = self._rate("alpha_n")
            beta_s = self._rate("beta_s")
            beta_n = self._rate("beta_n")
            k_y = self._rate("k_y")

            if cfg.get("method", "exact") == "exact":
                logp = JointStationaryLogLike(counts)
                pm.Potential("likelihood",
                             logp(model_index, alpha_s, alpha_n, beta_s,
                                  beta_n, k_y))
            else:
                pm.Simulator(
                    "counts",
                    self.simulator,
                    params=(model_index, alpha_s, alpha_n, beta_s, beta_n,
                            k_y),
                    sum_stat=cfg.get("sum_stat", "sort"),
                    distance=cfg.get("distance", "gaussian"),
                    epsilon=epsilon,
                    observed=counts,
                )

        return self.model


def model_probabilities(idata, heterodimer_prior=DEFAULT_HETERODIMER_PRIOR):
    """Posterior topology probabilities and the Bayes factor, from a joint fit.

    The Bayes factor divides out the prior odds, so it measures what the data
    contributed rather than what was assumed. A value near 1 means the counts
    did not separate the two topologies.
    """
    index = np.asarray(idata["posterior"]["model_index"]).ravel()
    p_het = float(index.mean())
    p_mono = 1.0 - p_het

    prior_odds = heterodimer_prior / (1.0 - heterodimer_prior)
    if p_mono == 0.0:
        bayes_factor = np.inf
    elif p_het == 0.0:
        bayes_factor = 0.0
    else:
        bayes_factor = (p_het / p_mono) / prior_odds

    return {"monomer": p_mono, "heterodimer": p_het,
            "bayes_factor_het_over_mono": bayes_factor,
            "n_draws": index.size}


MODELS = {"monomer": MonomerModel, "heterodimer": HeterodimerModel,
          "joint": JointModel}
