"""ABC-SMC model definitions for the monomer and heterodimer promoters.

Both models previously duplicated ``_data_setter``,
``_generate_and_preprocess_model_data``, the two config staticmethods, ``fit``
and a 37-line ``save`` verbatim; the shared behaviour now lives on
:class:`_SMCModel` and each subclass supplies only its priors and simulator.
"""

import json
from typing import Dict

import polars as pl
import pymc as pm
from pymc_extras.model_builder import ModelBuilder

from stochtf.inference.abc_smc import (
    pymc_fast_simulator_het,
    pymc_fast_simulator_monomer,
    summary_stat,
)

#: Prior scales shared by both models.
DEFAULT_MODEL_CONFIG: Dict = {
    "alpha_s_sigma": 1.0,
    "alpha_n_sigma": 1.0,
    "beta_s_sigma": 0.1,
    "beta_n_sigma": 0.3,
    "gamma_y_sigma": 0.01,
    "k_y_sigma": 0.5,
}

DEFAULT_SAMPLER_CONFIG: Dict = {
    "draws": 500,
    "tune": 1000,
    "chains": 5,
    "target_accept": 0.95,
}

#: Tolerance for the ABC acceptance kernel.
EPSILON = 2.0


class _SMCModel(ModelBuilder):
    """Shared ABC-SMC plumbing. Subclasses define ``_prior`` and ``_simulator``."""

    #: pm distribution used for all four rate priors.
    _prior = None
    #: numba-compiled simulator passed to pm.Simulator.
    _simulator = None

    def _data_setter(self, X, y=None):
        pass  # ModelBuilder requires this method; X/y are not used as standard inputs

    def _generate_and_preprocess_model_data(self, X, y=None):
        pass  # ModelBuilder requires this method

    @staticmethod
    def get_default_model_config() -> Dict:
        """Prior scales used when no model_config is supplied at construction."""
        return dict(DEFAULT_MODEL_CONFIG)

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """Sampler settings used when no sampler_config is supplied."""
        return dict(DEFAULT_SAMPLER_CONFIG)

    def build_model(self, X=None, y=None, **kwargs):
        with pm.Model() as self.model:
            cfg = self.model_config
            alpha_s = self._prior("alpha_s", sigma=cfg["alpha_s_sigma"])
            alpha_n = self._prior("alpha_n", sigma=cfg["alpha_n_sigma"])
            beta_s = self._prior("beta_s", sigma=cfg["beta_s_sigma"])
            beta_n = self._prior("beta_n", sigma=cfg["beta_n_sigma"])

            pm.Simulator(
                "sim",
                self._simulator,
                params=(alpha_s, alpha_n, beta_s, beta_n),
                sum_stat=summary_stat,
                epsilon=EPSILON,
                observed=y,
            )

    def fit(self, data: pl.DataFrame, sampler_config: dict = None, **kwargs):
        """Fit by Sequential Monte Carlo instead of ModelBuilder's default NUTS."""
        if sampler_config is None:
            sampler_config = self.sampler_config

        self.build_model(self.model_config, y=data)

        with self.model:
            self.idata = pm.sample_smc(draws=sampler_config["draws"], **kwargs)

            prior_distribution = pm.sample_prior_predictive(draws=500, random_seed=500)
            log_likelihood = pm.compute_log_likelihood(self.idata)
            log_prior = pm.stats.compute_log_prior(self.idata)

            if "prior" in prior_distribution:
                self.idata["prior"] = prior_distribution["prior"]
            if "prior_predictive" in prior_distribution:
                self.idata["prior_predictive"] = prior_distribution["prior_predictive"]
            self.idata["log_likelihood"] = log_likelihood["log_likelihood"]
            self.idata["log_prior"] = log_prior["log_prior"]
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


class MonomerModel(_SMCModel):
    """Independent SOX2 and NANOG binding, half-normal priors."""

    model_type = "MonomerModel"
    version = "0.1"
    _prior = staticmethod(pm.HalfNormal)
    _simulator = staticmethod(pymc_fast_simulator_monomer)


class DimerModel(_SMCModel):
    """Heterodimer binding, log-normal priors."""

    model_type = "DimerModel"
    version = "0.1"
    _prior = staticmethod(pm.LogNormal)
    _simulator = staticmethod(pymc_fast_simulator_het)


MODELS = {"monomer": MonomerModel, "dimer": DimerModel}
