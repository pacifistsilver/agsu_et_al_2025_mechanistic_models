"""Bayesian models for the monomer and heterodimer promoters.
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

GAMMA = 1.0
KERNEL_HORIZON = 15.0
MAX_SWITCHES = 400
EPSILON_SCALE = 0.25
MIN_EPSILON = 1e-3
FIXED_BETA_S = constants.BETA_S
FIXED_BETA_N = constants.BETA_N

DEFAULT_MODEL_CONFIG: Dict = {
    "alpha_s_mu": -1.0, "alpha_s_sigma": 2.0,
    "alpha_n_mu": -2.0, "alpha_n_sigma": 2.0,
    "beta_s_mu": 0.0, "beta_s_sigma": 2.0,
    "beta_n_mu": 0.0, "beta_n_sigma": 2.0,
    "k_y_mu": 3.0, "k_y_sigma": 1.5,
    "beta_s_fixed": FIXED_BETA_S,
    "beta_n_fixed": FIXED_BETA_N,
    "method": "exact",
    "epsilon": None,
    "epsilon_scale": EPSILON_SCALE,
    "sum_stat": "sort",
    "distance": "gaussian",
}

DEFAULT_SAMPLER_CONFIG: Dict = {
    "draws": 1000,
    "chains": 4,
}

def heterodimer_chain(a_s, b_s, a_n, b_n):
    """Two independent sites; mRNA whenever either is bound (states 10/01/11).    """
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
    """
    starts = np.array([0, 2, 3, 4], dtype=np.int64)
    targets = np.array([1, 2,   0,   0], dtype=np.int64)
    rates = np.array([a_s, a_n,  b_s,  b_n], dtype=np.float64)
    act = np.array([0.0, 1.0, 1.0])

    weights = np.array([1.0, a_s / b_s, a_n / b_n])
    return starts, targets, rates, act, weights / weights.sum()


PROMOTERS = {"monomer": monomer_chain, "heterodimer": heterodimer_chain}

@njit(cache=True)
def _seed_numba(seed):
    """Seed numba's own generator, which is separate from NumPy's."""
    np.random.seed(seed)


@njit(cache=True)
def _simulate_counts(starts, targets, rates, act, pi, k_y, gamma, n_cells,
                     horizon, max_switches):
    """``n_cells`` independent stationary counts from a promoter jump chain.
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
    """
    if rng is None:
        return int(np.random.randint(0, 2**31 - 1))
    if hasattr(rng, "integers"):
        return int(rng.integers(0, 2**31 - 1))
    return int(rng.randint(0, 2**31 - 1))  # legacy RandomState


def simulate_counts(rng, alpha_s, alpha_n, beta_s, beta_n, k_y,
                    promoter="heterodimer", size=None, gamma=GAMMA):
    """Stationary counts under one of the promoters, one draw per cell.
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
    return _simulate_counts(starts, targets, rates, act, pi, k, gamma, n_cells,
                            KERNEL_HORIZON / gamma, MAX_SWITCHES)


def simulate_monomer(rng, alpha_s, alpha_n, beta_s, beta_n, k_y, size=None):
    """Exclusive single site, S <- 0 -> N."""
    return simulate_counts(rng, alpha_s, alpha_n, beta_s, beta_n, k_y,
                           promoter="monomer", size=size)


def simulate_heterodimer(rng, alpha_s, alpha_n, beta_s, beta_n, k_y, size=None):
    """Two independent sites; any bound site gives the full rate k_y."""
    return simulate_counts(rng, alpha_s, alpha_n, beta_s, beta_n, k_y,
                           promoter="heterodimer", size=size)


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


# models

class _PromoterModel(ModelBuilder):
    promoter = None
    simulator = None
    version = "3.0"

    def _data_setter(self, X, y=None):
        pass  

    def _generate_and_preprocess_model_data(self, X, y=None):
        pass  

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
            return counts, None

        epsilon = cfg.get("epsilon")
        if epsilon is None:
            scale = cfg.get("epsilon_scale", EPSILON_SCALE)
            epsilon = max(float(scale * counts.std()), MIN_EPSILON)
            cfg["epsilon"] = epsilon  # record what was actually used
        return counts, float(epsilon)

    def _rate(self, name):
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
        """
        if sampler_config is None:
            sampler_config = self.sampler_config
        else:
            self.sampler_config = dict(sampler_config)

        self.build_model(y=data)

        with self.model:
            self.idata = pm.sample_smc(
                draws=sampler_config["draws"],
                chains=sampler_config.get("chains", 8),
                **kwargs,
            )
            if sample_prior:
                prior = pm.sample_prior_predictive(draws=500, random_seed=500)
                if "prior" in prior:
                    self.idata["prior"] = prior["prior"]
                if "prior_predictive" in prior:
                    self.idata["prior_predictive"] = prior["prior_predictive"]

        return self.idata

    def save(self, fname: str):
        """Sanitise the DataTree before delegating to ModelBuilder's saver.
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
    """

    model_type = "MonomerModel"
    promoter = "monomer"
    simulator = staticmethod(simulate_monomer)


class HeterodimerModel(_PromoterModel):
    """Two independent sites; any bound site gives the full transcription rate.
    """

    model_type = "HeterodimerModel"
    promoter = "heterodimer"
    simulator = staticmethod(simulate_heterodimer)


#: Prior probability that the heterodimer is the right topology. 
DEFAULT_HETERODIMER_PRIOR = 0.5


class JointModel(_PromoterModel):
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
