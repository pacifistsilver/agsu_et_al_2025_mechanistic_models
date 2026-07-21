import pymc as pm
import numpy as np
import polars as pl
from abc_smc import pymc_fast_simulator_monomer, pymc_fast_simulator_het, summary_stat
from src.ssa_params import monomer_params
from pymc_extras.model_builder import ModelBuilder
from typing import Dict

initial_params, _, _ = monomer_params
synthetic_data = np.load(file="./data/synthetic_data/synthetic_heterodimer_data.npy")
nanog = np.load(file="./data/nanog.npy")
sox2=np.load(file="./data/sox2.npy")
rex1=np.load(file="./data/rex1.npy")
esrrb=np.load(file="./data/esrrb.npy")

array = [sox2, nanog, rex1]

class MonomerModel(ModelBuilder):
    model_type = "MonomerModel"
    version = "0.1"
    
    def _data_setter(self, X, y=None):
        pass # ModelBuilder requires this method, even if we don't use X/y standard inputs
        
    def _generate_and_preprocess_model_data(self, X, y=None):
        pass # ModelBuilder requires this method

    @staticmethod
    def get_default_model_config() -> Dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization.
        The model config dict is generally used to specify the prior values we want to build the model with.
        It supports more complex data structures like lists, dictionaries, etc.
        It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.
        """
        model_config: Dict = {
            "alpha_s_sigma": 1,
            "alpha_n_sigma": 1,
            "beta_s_sigma": 0.05,
            "beta_n_sigma": 0.1,
            "gamma_y_sigma": 0.1,
            "k_y_sigma": 0.1
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
        sampler_config: Dict = {
            "draws": 1000,
            "tune": 1000,
            "chains": 5,
            "target_accept": 0.95,
        }
        return sampler_config

    def build_model(self, X=None, y=None, **kwargs):       
        with pm.Model() as self.model:
            cfg = self.model_config
            alpha_s = pm.Uniform("alpha_s", lower = 0, upper=cfg["alpha_s_sigma"])    
            alpha_n = pm.Uniform("alpha_n", lower = 0, upper=cfg["alpha_n_sigma"])    
            k_y = pm.Uniform("k_y", lower = 0, upper=cfg["k_y_sigma"])
            gamma_y = pm.Uniform("gamma_y", lower = 0, upper=cfg["gamma_y_sigma"])

            sim = pm.Simulator(
                "sim", 
                pymc_fast_simulator_monomer, 
                params=(alpha_s, alpha_n, k_y, gamma_y), 
                sum_stat=summary_stat,
                epsilon = 5,
                observed=synthetic_data
            )
                
    def fit(self, data: pl.DataFrame, sampler_config: dict = None, **kwargs):
        """Override fit to use SMC instead of the default NUTS sampler"""
        if sampler_config is None:
            sampler_config = self.sampler_config

        self.build_model(self.model_config, data)

        with self.model:
            # Use pm.sample_smc for Sequential Monte Carlo
            self.idata = pm.sample_smc(
                draws=sampler_config["draws"],
                **kwargs
            )

            prior_distribution = pm.sample_prior_predictive(draws = 500, random_seed=500)
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
        """
        Custom save method that sanitizes the DataTree object to prevent 
        NetCDF mixed-type object array crashes before delegating to the parent saver.
        """
        if hasattr(self, "idata") and self.idata is not None:
            import json
            
            # Inject ModelBuilder metadata into idata.attrs so load() doesn't crash with KeyError
            if "model_config" not in self.idata.attrs:
                self.idata.attrs["model_config"] = json.dumps(self.model_config)
            if "sampler_config" not in self.idata.attrs:
                self.idata.attrs["sampler_config"] = json.dumps(self.sampler_config if hasattr(self, "sampler_config") else self.get_default_sampler_config())
            if "version" not in self.idata.attrs:
                self.idata.attrs["version"] = self.version
            if "model_type" not in self.idata.attrs:
                self.idata.attrs["model_type"] = self.model_type
            
            # DataTree uses .groups instead of ._groups
            for group_name in self.idata.groups:
                # In DataTree, we access groups like a dictionary
                group = self.idata[group_name]
                
                # Copy the keys to list to avoid mutating while iterating
                var_names = list(group.data_vars.keys())
                for var_name in var_names:
                    var = group[var_name]
                    if var.dtype == object:
                        try:
                            # Force integer/float mixes into pure floats
                            group[var_name] = var.astype(float)
                        except Exception:
                            # If completely broken (e.g. sequence arrays), delete it safely
                            del group[var_name]
                            
        # Call the default ModelBuilder save method
        super().save(fname)
        
        
class DimerModel(ModelBuilder):
    model_type = "DimerModel"
    version = "0.1"
    
    def _data_setter(self, X, y=None):
        pass # ModelBuilder requires this method, even if we don't use X/y standard inputs
        
    def _generate_and_preprocess_model_data(self, X, y=None):
        pass # ModelBuilder requires this method

    @staticmethod
    def get_default_model_config() -> Dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization.
        The model config dict is generally used to specify the prior values we want to build the model with.
        It supports more complex data structures like lists, dictionaries, etc.
        It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.
        """
        model_config: Dict = {
            "alpha_s_sigma": 1.0,
            "alpha_n_sigma": 1.0,
            "beta_s_sigma": 0.1,
            "beta_n_sigma": 0.3,
            "gamma_y_sigma": 0.01,
            "k_y_sigma": 0.5,
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
        sampler_config: Dict = {
            "draws": 500,
            "tune": 1000,
            "chains": 5,
            "target_accept": 0.95,
        }
        return sampler_config

    def build_model(self, X=None, y=None, **kwargs):       
        with pm.Model() as self.model:
            cfg = self.model_config
            alpha_s = pm.HalfNormal("alpha_s", sigma=cfg["alpha_s_sigma"]) 
            alpha_n = pm.HalfNormal("alpha_n", sigma=cfg["alpha_n_sigma"]) 
            beta_s = pm.HalfNormal("beta_s", sigma=cfg["beta_s_sigma"])   
            beta_n = pm.HalfNormal("beta_n", sigma=cfg["beta_n_sigma"])   

            sim = pm.Simulator(
                "sim", 
                pymc_fast_simulator_het, 
                params=(alpha_s, alpha_n, beta_s, beta_n), 
                sum_stat=summary_stat,
                epsilon = 1,
                observed=synthetic_data
            )            
                
    def fit(self, data: pl.DataFrame, sampler_config: dict = None, **kwargs):
        """Override fit to use SMC instead of the default NUTS sampler"""
        if sampler_config is None:
            sampler_config = self.sampler_config

        self.build_model(self.model_config, y= data)

        with self.model:
            # Use pm.sample_smc for Sequential Monte Carlo
            self.idata = pm.sample_smc(
                draws=sampler_config["draws"],
                **kwargs
            )
            prior_distribution = pm.sample_prior_predictive(draws = 500, random_seed=500)
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
        """
        Custom save method that sanitizes the DataTree object to prevent 
        NetCDF mixed-type object array crashes before delegating to the parent saver.
        """
        if hasattr(self, "idata") and self.idata is not None:
            import json
            
            # Inject ModelBuilder metadata into idata.attrs so load() doesn't crash with KeyError
            if "model_config" not in self.idata.attrs:
                self.idata.attrs["model_config"] = json.dumps(self.model_config)
            if "sampler_config" not in self.idata.attrs:
                self.idata.attrs["sampler_config"] = json.dumps(self.sampler_config if hasattr(self, "sampler_config") else self.get_default_sampler_config())
            if "version" not in self.idata.attrs:
                self.idata.attrs["version"] = self.version
            if "model_type" not in self.idata.attrs:
                self.idata.attrs["model_type"] = self.model_type
            
            # DataTree uses .groups instead of ._groups
            for group_name in self.idata.groups:
                # In DataTree, we access groups like a dictionary
                group = self.idata[group_name]
                
                # Copy the keys to list to avoid mutating while iterating
                var_names = list(group.data_vars.keys())
                for var_name in var_names:
                    var = group[var_name]
                    if var.dtype == object:
                        try:
                            # Force integer/float mixes into pure floats
                            group[var_name] = var.astype(float)
                        except Exception:
                            # If completely broken (e.g. sequence arrays), delete it safely
                            del group[var_name]
                            
        # Call the default ModelBuilder save method
        super().save(fname)


if __name__ == '__main__':  
    # Instantiate the model
    model = DimerModel()    
    idata = model.fit(nanog)
    cfg = model.model_config
    fname = f"nanog.nc"
    model.save(fname)

    
