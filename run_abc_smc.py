import pymc as pm
import numpy as np
import polars as pl
from abc_smc import pymc_fast_simulator, pymc_fast_simulator_het
from src.ssa_params import monomer_params
from pymc_extras.model_builder import ModelBuilder
from typing import Dict

initial_params, _, _ = monomer_params
observed_data = np.load(file="sim_data.npy")
nanog_data = np.load(file="./data/nanog.npy")
sox2_data=np.load(file="./data/sox2.npy")

def my_summary_stats(data):
    mean = np.mean(data)
    var = np.var(data)
    fano = var/mean if mean>0 else 0.0
    cv = np.sqrt(var)/mean if mean>0 else 0.0
    return np.log1p(np.array([mean, var, fano, cv]))

observed_stats = my_summary_stats(nanog_data)
observed_stats_sox2 = my_summary_stats(sox2_data)
observed_stats_nanog = my_summary_stats(nanog_data)


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
            "alpha_s": 0.5,
            "alpha_n": 0.3,
            "beta_s": 0.06,
            "beta_n": 0.2,
            "gamma_y": 1.0,
            "k_y": 10.0
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
            "tune": 100,
            "chains": 5,
            "target_accept": 0.95,
        }
        return sampler_config

    def build_model(self, X=None, y=None, **kwargs):       
        with pm.Model() as self.model:
            alpha_s = pm.HalfNormal("alpha_s", sigma=1.0)
            alpha_n = pm.HalfNormal("alpha_n", sigma=1.0)    
            beta_s = pm.HalfNormal("beta_s", sigma=0.5)    
            beta_n = pm.HalfNormal("beta_n", sigma=0.8)    
            k_y = pm.HalfNormal("k_y", sigma=0.01)
            gamma_y = pm.HalfNormal("gamma_y", sigma=0.01)

            sim = pm.Simulator(
                "sim", 
                pymc_fast_simulator, 
                params=(alpha_s, beta_s, alpha_n, beta_n, k_y, gamma_y), 
                distance="laplace",
                epsilon=np.array([1.0, 4.0, 0.25, 0.04]),
                observed=observed_stats
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
                chains=sampler_config["chains"],
                **kwargs
            )

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
            "alpha_s": 0.5,
            "alpha_n": 0.3,
            "beta_s": 0.06,
            "beta_n": 0.2,
            "gamma_y": 1.0,
            "k_y": 10.0
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
            "tune": 100,
            "chains": 5,
            "target_accept": 0.95,
        }
        return sampler_config

    def build_model(self, X=None, y=None, **kwargs):
        with pm.Model() as self.model:
            alpha_s = pm.HalfNormal("alpha_s", sigma=1.0)
            alpha_n = pm.HalfNormal("alpha_n", sigma=1.0)    
            beta_s = pm.HalfNormal("beta_s", sigma=0.5)    
            beta_n = pm.HalfNormal("beta_n", sigma=0.8)    
            k_y = pm.HalfNormal("k_y", sigma=0.01)
            gamma_y = pm.HalfNormal("gamma_y", sigma=0.01)

            sim = pm.Simulator(
                "sim", 
                pymc_fast_simulator_het, 
                params=(alpha_s, beta_s, alpha_n, beta_n, k_y, gamma_y), 
                distance="laplace",
                epsilon=np.array([1.0, 4.0, 0.25, 0.04]),
                observed=y
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
                chains=sampler_config["chains"],
                **kwargs
            )
            prior_distribution = pm.sample_prior_predictive(draws = 500, random_seed=500)
            self.idata.append(prior_distribution)

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
    model = MonomerModel()
    idata = model.fit(observed_data)
    fname = "monomer_synthetic_fit1.nc"
    model.save(fname)
    
