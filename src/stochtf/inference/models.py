"""Bayesian models for the monomer and heterodimer 
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

#: Promoter switches per cell before the kernel is closed off with the
#: occupancy-averaged activity. Only reachable when switching is ~25x faster
#: than degradation, where the promoter self-averages and the discarded
#: fluctuation is below what the data resolve. Caps the cost of a wild prior
#: draw, which could otherwise stall a whole SMC stage.
MAX_SWITCHES = 400

#: Default ABC tolerance as a fraction of the observed standard deviation.
EPSILON_SCALE = 0.25

#: Floor on epsilon, so degenerate data cannot produce a zero-width kernel.
MIN_EPSILON = 1e-3

#: Off-rates pinned rather than inferred, in units of gamma. Stationary counts
#: cannot determine four switching rates at once (see
#: :mod:`stochtf.inference.identifiability`), so pinning the two dissociation
#: rates -- the pair single-molecule tracking measures directly -- leaves the
#: on-rates and k_y identifiable. Set either to None in ``model_config`` to
#: infer it instead.
FIXED_BETA_S = constants.BETA_S
FIXED_BETA_N = constants.BETA_N

#: Log-normal priors, in units of gamma. Switching rates centre on 1 (one
#: event per mRNA lifetime), spread four orders of magnitude either side to
#: cover both the bursty and fast-switching regimes. k_y centres on 20, the
#: order needed for the observed means of 30-250. The beta priors apply only
#: when the matching ``*_fixed`` entry is cleared.
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
    # "abc" simulates counts and compares them through the epsilon kernel.
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
    """Builds the jump chain for two independent binding sites.

    mRNA is made whenever either site is bound (states 10, 01, 11). State order
    00, 10, 01, 11 matches
    :func:`stochtf.analytical.pgf.promoter_generator`, so the two are directly
    comparable.

    Args:
        a_s: SOX2 binding rate.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate.
        b_n: NANOG unbinding rate.

    Returns:
        A tuple (starts, targets, rates, act, pi) in flat jump-chain form.
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
    """Builds the jump chain for one contested binding site.

    Occupancy is exclusive (S <- 0 -> N), so the states are empty, SOX2-bound
    and NANOG-bound, with no doubly bound state. mRNA is made whenever the site
    is occupied. Detailed balance on this star gives pi proportional to
    (1, a_s/b_s, a_n/b_n).

    Args:
        a_s: SOX2 binding rate.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate.
        b_n: NANOG unbinding rate.

    Returns:
        A tuple (starts, targets, rates, act, pi) in flat jump-chain form.
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
    """Seeds numba's generator, which is separate from NumPy's."""
    np.random.seed(seed)


@njit(cache=True)
def _simulate_counts(starts, targets, rates, act, pi, k_y, gamma, n_cells,
                     horizon, max_switches):
    """Draws independent stationary counts from a promoter jump chain.

    Each cell draws its promoter state from ``pi``, walks the chain forward in
    age, accumulates the kernel-weighted activity in closed form between
    switches, and draws one Poisson count from the total.

    Args:
        starts: Offsets into ``targets``/``rates``, one per state plus a bound.
        targets: Destination state of each jump.
        rates: Rate of each jump.
        act: Per-state transcription activity.
        pi: Stationary promoter occupancy.
        k_y: Transcription rate in the active states.
        gamma: mRNA degradation rate.
        n_cells: Number of independent counts to draw.
        horizon: Age horizon, in mRNA lifetimes.
        max_switches: Cap on promoter switches per cell.

    Returns:
        An array of ``n_cells`` stationary counts.
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
                # Also catches the log(0) = -inf draw.
                if not (next_age < horizon):
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
    """Returns the number of cells ``pm.Simulator`` is asking for."""
    if size is None:
        return 1
    return int(np.prod(np.atleast_1d(size)))


def _seed_from(rng):
    """Takes a seed from PyMC's generator so simulations are reproducible.

    The kernel is numba-compiled and uses numba's own global generator, which a
    NumPy ``Generator`` cannot drive directly. Reseeding it per call ties the
    two together.

    Args:
        rng: A NumPy ``Generator``, or None to draw from NumPy's global
          generator instead.

    Returns:
        A non-negative int seed for the numba kernel.
    """
    if rng is None:
        return int(np.random.randint(0, 2**31 - 1))
    if hasattr(rng, "integers"):
        return int(rng.integers(0, 2**31 - 1))
    return int(rng.randint(0, 2**31 - 1))  # legacy RandomState


def simulate_counts(rng, alpha_s, alpha_n, beta_s, beta_n, k_y,
                    promoter="heterodimer", size=None, gamma=GAMMA):
    """Draws stationary counts under one promoter, one draw per cell.

    Argument order follows the priors declared below (alpha_s, alpha_n, beta_s,
    beta_n), which is not the order the analytical routines take.

    Args:
        rng: NumPy ``Generator`` used to seed the numba kernel.
        promoter: Key into :data:`PROMOTERS`.
        alpha_s: SOX2 binding rate.
        alpha_n: NANOG binding rate.
        beta_s: SOX2 unbinding rate.
        beta_n: NANOG unbinding rate.
        k_y: Transcription rate in the active states.
        size: Number of cells requested by ``pm.Simulator``.

    Returns:
        Stationary counts, one per cell. Parameters outside the model's support
        give an all-zero sample rather than an exception, so the sampler scores
        them as a bad fit.
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
    """Computes the exact log-likelihood by finite state projection.

    :mod:`stochtf.cme.stationary` gives P(y | theta) numerically for either
    topology, so the whole distribution is scored directly. There is no
    tolerance to tune and no simulation noise.

    Args:
        counts: Molecule numbers, one per cell.
        promoter: Key into :data:`PROMOTERS`.
        alpha_s: SOX2 binding rate.
        alpha_n: NANOG binding rate.
        beta_s: SOX2 unbinding rate.
        beta_n: NANOG unbinding rate.
        k_y: Transcription rate in the active states.

    Returns:
        The log-likelihood, or -inf for parameters outside the model's support.
        Out-of-support parameters return rather than raise so that a sampler can
        simply reject them.
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
    """A :class:`StationaryLogLike` whose topology is chosen by an input.

    Takes the model index first, so one Potential serves both promoters and SMC
    can move between them within a single run.
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
    """Shared plumbing for the promoter models.

    Both fitting methods are sampled by ``pm.sample_smc`` and differ only in
    how a proposal is scored: ``method="exact"`` evaluates the whole
    distribution by finite state projection, ``method="abc"`` simulates counts
    and compares them through the epsilon kernel.

    Attributes:
        promoter: Key into :data:`PROMOTERS`, set by each subclass.
        simulator: Forward simulator used under ``method="abc"``.
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
        """Flattens counts and, for ABC only, resolves epsilon against them."""
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
        """Declares a rate as a pinned constant or a log-normal prior.

        A pinned rate enters the graph as a constant rather than a
        zero-variance prior, so it costs the sampler nothing and never appears
        in the trace.

        Args:
            name: Parameter name, also the key checked for a pinned value.
            cfg: Model config holding the priors and any ``*_fixed`` entries.

        Returns:
            A PyMC random variable, or a constant if the rate is pinned.
        """
        cfg = self.model_config
        fixed = cfg.get(f"{name}_fixed")
        if fixed is not None:
            return pt.constant(float(fixed), name=name)
        return pm.LogNormal(name, mu=cfg[f"{name}_mu"],
                            sigma=cfg[f"{name}_sigma"])

    def build_model(self, X=None, y=None, **kwargs):
        """Builds the PyMC model for one promoter topology.

        Args:
            X: Unused. Present for the ModelBuilder interface.
            y: Observed counts, one per cell.
            **kwargs: Unused.

        Returns:
            The built ``pm.Model``.
        """
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
        """Fits the model by sequential Monte Carlo.

        ``pm.sample_smc`` anneals from the prior to the posterior, moving
        particles with a gradient-free independent Metropolis kernel. Under
        ``method="abc"`` the only observed node is a ``pm.Simulator``, so each
        proposal costs one simulated dataset and the target is the ABC
        posterior rather than the true one.

        Args:
            data: Observed counts, one per cell.
            sampler_config: Overrides for draws and chains. Defaults if None.
            sample_prior: Whether to draw the prior predictive too. Worth
              skipping when fitting hundreds of datasets in a sweep.
            **kwargs: Forwarded to ``pm.sample_smc``.

        Returns:
            The fitted ``InferenceData``.
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
        """Sanitises the trace, then delegates to ModelBuilder's saver.

        NetCDF cannot represent object-dtype arrays, and ``ModelBuilder.load``
        raises KeyError if its metadata attrs are absent, so both are fixed up
        first.

        Args:
            fname: Destination ``.nc`` path.
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
    This is the model-selection scheme of Toni & Stumpf. It needs no
    trans-dimensional machinery because the two promoters take the same five
    rates, so the parameter space does not change when the index flips and one
    set of priors serves both.

    Two caveats when reading the result. The comparison is only as meaningful
    as its shared footing: both topologies must see the same data, priors and
    scoring method. And the data may simply not decide -- at matched burst
    frequency and size the two put nearly the same law on the counts, so a
    posterior near the prior is the honest answer rather than a failed sampler.
    Check :func:`model_probabilities` against the prior before reading anything
    into it.
    """

    model_type = "JointModel"
    simulator = staticmethod(simulate_joint)

    @staticmethod
    def get_default_model_config() -> Dict:
        return {**DEFAULT_MODEL_CONFIG,
                "heterodimer_prior": DEFAULT_HETERODIMER_PRIOR}

    def build_model(self, X=None, y=None, **kwargs):
        """Builds the joint model, with the topology as a parameter.

        Args:
            X: Unused. Present for the ModelBuilder interface.
            y: Observed counts, one per cell.
            **kwargs: Unused.

        Returns:
            The built ``pm.Model``.
        """
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
    """Reads topology probabilities and the Bayes factor off a joint fit.

    The Bayes factor divides out the prior odds, so it measures what the data
    contributed rather than what was assumed. A value near 1 means the counts
    did not separate the two topologies.

    Args:
        idata: ``InferenceData`` from a :class:`JointModel` fit.
        heterodimer_prior: Prior probability assigned to the heterodimer.

    Returns:
        A dict with keys ``monomer``, ``heterodimer`` and
        ``bayes_factor_het_over_mono``.
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


def log_evidence(idata):
    """Returns the mean SMC log marginal likelihood of one fit.

    ``pm.sample_smc`` records log Z once per annealing stage, writing NaN for
    every stage before the last, and casts the stat to object dtype whenever
    there is more than one chain. Chains choose their own beta schedules, so
    the rows can differ in length; flattening and taking a nan-mean handles
    that as well as the square case.

    Args:
        idata: ``InferenceData`` from an SMC fit.

    Returns:
        The mean log marginal likelihood over chains.
    """
    stat = np.asarray(idata.sample_stats["log_marginal_likelihood"])
    values = np.concatenate([np.asarray(row, dtype="float64").ravel()
                             for row in stat])
    return float(np.nanmean(values))


def bayes_factor(idatas):
    """Returns the Bayes factor of ``idatas[1]`` over ``idatas[0]``.

    Args:
        idatas: Two SMC fits, monomer first and heterodimer second.

    Returns:
        The ratio of their marginal likelihoods.
    """
    return float(np.exp(log_evidence(idatas[1]) - log_evidence(idatas[0])))


def fit_both(counts, sampler_config=None, model_config=None,
             progressbar=False, **kwargs):
    """Fits the monomer and the heterodimer independently on the same counts.

    Both fits get the same priors and scoring method, which is what makes the
    ratio of their evidences a Bayes factor rather than an artefact of the
    setup.

    Args:
        counts: Molecule numbers, one per cell.
        sampler_config: Overrides for draws and chains, applied to both fits.
        model_config: Model config applied to both fits. Defaults if None.
        progressbar: Whether to show the sampler progress bar.
        **kwargs: Forwarded to each fit.

    Returns:
        A tuple (monomer, heterodimer) of fitted models. Pass their ``idata``
        to :func:`bayes_factor` in that order.
    """
    fits = []
    for name in ("monomer", "heterodimer"):
        model = MODELS[name](model_config=dict(model_config)
                             if model_config else None)
        config = dict(model.get_default_sampler_config())
        if sampler_config:
            config.update(sampler_config)
        model.fit(counts, sampler_config=config, progressbar=progressbar,
                  sample_prior=False, **kwargs)
        fits.append(model)
    return tuple(fits)


MODELS = {"monomer": MonomerModel, "heterodimer": HeterodimerModel,
          "joint": JointModel}
