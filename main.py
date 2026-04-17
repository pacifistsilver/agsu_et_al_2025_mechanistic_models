import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import polars as pl
import pandas as pd
import os
import itertools
import concurrent.futures
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
"""Partition function model and Gillespie model of gene expression given clustering.

This module obtains residence time, mRNA expression from a realisation of a single trajectory of the chemical master equation.
Additionally, from the trajectory parameters, we also determine the expected transcription rate (rho).

Typical usage:
model_call = ModelCall.run_trajectory(params)

If you face any issues, email: dwl25@ic.ac.uk or danielluo1143@gmail.com 
"""

"""
TODO: 
    1. random starting scenarios according to some binomial distribution (i.e. random binding position for tf/none bound/ etc...)
    2. reassociation following hop should follow some gaussian distribution.
    3. add correct exceptions for all functions. 
"""

class PartitionFunction: 
    """Calculate Rho (expected transcription rate) from the partition function.
    
    The full partition function is defined as: 
    \begin{equation} \mathcal Z_2 = \sum_{\ell=0}^{\min(n,N-n)} {N-n \choose \ell}{n\choose \ell}\ell!\, c^\ell
    \end{equation}
    where N is the number of identical SOX2 binding sites, n the number of individually bound TF,
    K1 the intrinsic affinity for these sites, S the amount of free bulk TF.
    In addition, there is a dimensionless binding factor c, and the number of paired contacts l. 
    
    Rho can be obtained in three ways:
        (1) alpha * (1 - (zed**-1))
        (2) alpha * n
        (3) (alpha * K_alpha * n) / (K_alpha * n + 1)
    where alpha is the rate of transcription, n is the number of bound sites, K_alpha is the
    constant which sets the rate at which alpha saturates for large n (where K_alpha^-1 << n),
    and zed the sum of the partition function.
    
    Attributes:
        None 
    """   
    def calculate_Zn(self, n: int, N: int, K1: float, S: float, c: float) -> float:
        """Calculate partition function for a single n.
        
        Args:
            n: Number of bound sites.
            N: Total sites
            K1: Intrinsic affinity for sites.
            S: Bulk TF.
            c: Dimensionless binding factor.
        
        Returns:
            A float sum of the first and second 
            summation of the partition function
            
        Raises:
            None
        """
        binding_term = math.comb(N, n) * ((K1 * S)**n) # first summation
        
        interaction_sum = sum(
            math.comb(N - n, l) * math.comb(n, l) * math.factorial(l) * (c**l)
            for l in range(min(n, N - n) + 1)
        ) # second summation
        return binding_term * interaction_sum
    
    def calculate_total_Z(self, N: int, K1: float, S: float, c: float) -> float:
        return sum(self.calculate_Zn(n, N, K1, S, c) for n in range(N + 1)) # summation over all n -> N
    
    def return_maximal_rho(self, N: int, K1: float, S: float, c: float, alpha: float) -> float:
        """Calculate Rho in case of any binding achieve maximal rate.

        First method of calculating rho which is the case where transcription rate alpha
        is achieved by any binding of a TF to DNA.
        
        Args:
            n: Number of bound sites.
            N: Total sites
            K1: Intrinsic affinity for sites.
            S: Bulk TF.
            c: Dimensionless binding factor.
            alpha: Rate of transcription
                    
        Returns:
            Float sum of the transcription rate rho.
        
        Raises:
            None
        """
        zed = self.calculate_total_Z(N, K1, S, c)
        return alpha * (1 - (zed**-1)) if zed else 0.0
    
    def return_nonmaximal_rho(self, N: int, K1: float, S: float, c: float, alpha: float, K_alpha: float = None, mode: str = "constant") -> float:
        """Calculate Rho in case where binding leads to a multiplicative effect on transcription rate.

        Second and third method of calculating rho where transcription 
        is proportional to occupancy time on a per Sox2 basis
        
        Args:
            n: Number of bound sites.
            N: Total sites
            K1: Intrinsic affinity for sites.
            S: Bulk TF.
            c: Dimensionless binding factor.
            alpha: Rate of transcription
            K_alpha: Constant which determines rate at which saturation occurs.
            mode: Chooes
                    
        Returns:
            Float sum of the transcription rate rho.
        
        Raises:
            None
        """
        
        numerator_sum = 0.0 # sum of individual zed 
        Z_total = self.calculate_total_Z(N, K1, S, c)
        
        for n in range(1, N + 1):
            Zn = self.calculate_Zn(n, N, K1, S, c)
            
            if mode == "constant":
                alpha_n = alpha
            elif mode == "linear": 
                alpha_n = alpha * n
            elif mode == "saturating" and K_alpha is not None:
                alpha_n = (alpha * K_alpha * n) / (K_alpha * n + 1)
            else:
                alpha_n = 0.0
                
            numerator_sum += alpha_n * Zn
            
        return numerator_sum / Z_total if Z_total else 0.0    

class ModelCall:
    """Stochastic simulation of a single trajectory. 
    
    Attributes:
        model_params:
            Trajectory parameters.
        model_var:
            Initial variable states (free TF, bound TF, mRNA)
        model_binding_sites:
            Number of binding sites in chromatin lattice.
        sim_max_time:
            Max time to run simulation for.
        record_interval:
            Simulation time interval to record reactions for.
        track_history: 
            If True save reaction history.
    """
    def __init__(self, model_param: dict, model_var: dict, model_binding_sites: int, sim_max_time: int, record_interval: float = 1.0, track_history: bool = True):
        """Initialise class instance.
        
        Args:
            model_params:
                Trajectory parameters.
            model_var:
                Initial variable states (free TF, bound TF, mRNA)
            model_binding_sites:
                Number of binding sites in chromatin lattice.
            sim_max_time:
                Max time to run simulation for.
            record_interval:
                Simulation time interval to record reactions for.
            track_history: 
                If True save reaction history.
        """
        self.t_max: int = sim_max_time
        self.record_interval = record_interval
        self.sox2_model_parameters: dict = model_param
        self.sox2_model_variables: dict = model_var
        self.n_binding_sites: int = model_binding_sites
        self.track_history = track_history

    def _initialize_state(self):
        """Initialise simulation variables, parameters, chromatin states.
        
        Initialises the stoichiometry, hopping kernel, reaction constants, etc...
        
        Args:
            None
        Raises:
            None 
        """
        self.sox2_simulation_parameters = [self.sox2_model_parameters[key] for key in self.sox2_model_parameters]
        self.sox2_simulation_variables = [self.sox2_model_variables[key] for key in self.sox2_model_variables]

        self.t = 0.0
        self.next_record_time = 0.0
        
        self.is_free = np.ones(self.n_binding_sites, dtype=bool)
        self.is_unpaired_bound = np.zeros(self.n_binding_sites, dtype=bool)
        self.chromatin_lattice = np.zeros(self.n_binding_sites, dtype=np.int8)
        self.parameter_states = np.array(self.sox2_simulation_parameters, dtype=np.int32)
        self.bridged_to = np.full(self.n_binding_sites, -1, dtype=np.int32)
        
        # Kernel & Hopping Weights
        self.kernel_matrix = np.ones((self.n_binding_sites, self.n_binding_sites), dtype=float)
        np.fill_diagonal(self.kernel_matrix, 0.0)
        self.hop_weights = np.sum(self.kernel_matrix, axis=1)

        # Reaction Constants
        self.stoichiometry_matrix = np.array([
            [1, -1, -1, 1, 0, 0, 0, 0],
            [0, 1, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 0, 0],
        ], dtype=np.int32)

        self.reaction_names = {
            0: "prod_s", 1: "bind", 2: "deg_s", 3: "unbind",
            4: "prod_m", 5: "deg_m", 6: "pair"
        }

        self.times, self.bulk_states, self.spatial_states, self.reaction_history, self.residence_time_states = [], [], [], [], []
        self.site_bind_times = np.full(self.n_binding_sites, -1.0)

    def _calculate_propensities(self):
        """Update propensities and return new propensities.
        
        Args:
            None
        Returns:
            Propensities array containing all chemical reactions and
            sum of propensities of all chemical reactions.
        Raises:
            None
        """
        k_prod_s, k_deg_s, k_prod_m, k_deg_m, k_bind, k_unbind, k_hop, _ = self.sox2_simulation_variables
        sox2_free, sox2_bound, mrna_count = self.parameter_states
        
        unbound_sites = np.sum(self.is_free)
        total_hop_weight = np.sum(self.hop_weights[self.is_unpaired_bound])

        propensities = np.array([
            k_prod_s,  
            k_bind * sox2_free * unbound_sites,  
            k_deg_s * sox2_free,  
            k_unbind * sox2_bound,  
            k_prod_m * sox2_bound,  
            k_deg_m * mrna_count,  
            k_hop * total_hop_weight,  
        ])
        return propensities, np.sum(propensities)

    def _execute_spatial_reaction(self, reaction_index):
        """Track updates to chromatin lattice array in case of binding/sliding.
        
        Handles three cases for the lattice: (1) bind (2) unbind (3) pair.
        Updates chromatin_lattice index position based on reaction choice.
        
        Args:
            reaction_index:
                Index reference to propensity array. i.e. which reaction was selected.
        Raises:
            None
        """
        site_target, site_paired_with = -1, -1

        if reaction_index == 1:  # bind reaction
            free_indices = np.where(self.is_free)[0]
            site_target = np.random.choice(free_indices)
            
            self.is_free[site_target] = False
            self.is_unpaired_bound[site_target] = True
            self.chromatin_lattice[site_target] = 1
            self.hop_weights -= self.kernel_matrix[:, site_target]
            
            self.site_bind_times[site_target] = self.t
            site_paired_with = 0 

        elif reaction_index == 3:  # unbind reaction
            bound_indices = np.where(~self.is_free)[0]
            if len(bound_indices) > 0:
                chosen_site = np.random.choice(bound_indices)
                duration = self.t - self.site_bind_times[chosen_site]
                self.residence_time_states.append([chosen_site, duration])
                self.site_bind_times[chosen_site] = -1.0 
                
                self.is_free[chosen_site] = True
                self.chromatin_lattice[chosen_site] = 0
                self.hop_weights += self.kernel_matrix[:, chosen_site]
                
                if self.bridged_to[chosen_site] != -1:
                    paired_site = self.bridged_to[chosen_site]
                    self.bridged_to[chosen_site] = -1
                    self.bridged_to[paired_site] = -1
                    self.is_unpaired_bound[paired_site] = True 
                    
                    self.parameter_states[0] -= 1  
                    self.parameter_states[1] += 1  
                else:
                    self.is_unpaired_bound[chosen_site] = False
                    
                site_target = chosen_site
                site_paired_with = 0 

        elif reaction_index == 6:  # pair reaction
            unpaired_indices = np.where(self.is_unpaired_bound)[0]
            unpaired_weights = self.hop_weights[unpaired_indices]
            
            # check for free sites and select next site.
            dwell_site = np.random.choice(unpaired_indices, p=(unpaired_weights / np.sum(unpaired_weights)))
            dest_weights = self.kernel_matrix[dwell_site, :] * self.is_free
            next_site = np.random.choice(self.n_binding_sites, p=(dest_weights / np.sum(dest_weights)))

            # update chromatin_lattice accordingly
            site_target, site_paired_with = dwell_site, next_site
            self.is_free[next_site] = False
            self.chromatin_lattice[next_site] = 1
            self.hop_weights -= self.kernel_matrix[:, next_site]
            self.is_unpaired_bound[dwell_site] = False 
            
            self.site_bind_times[next_site] = self.t
            # in the bridged_to array, add the new bridged sites.
            self.bridged_to[dwell_site] = next_site
            self.bridged_to[next_site] = dwell_site

        if self.track_history:
            self.reaction_history.append((self.t, self.reaction_names[reaction_index], site_target, site_paired_with))

    def _record_snapshot(self, t_next_reaction):
        while self.next_record_time <= self.t + t_next_reaction and self.next_record_time <= self.t_max:
            self.times.append(self.next_record_time)
            self.bulk_states.append(self.parameter_states.copy())
            self.spatial_states.append(self.chromatin_lattice.copy())
            self.next_record_time += self.record_interval

    def _generate_dataframes(self):
        """Setup model output dataframes.
        
        Args:
            None
        Returns:
            sim_variable_states_df: 
                Polars df showing time evolution of system variables
                with columns "time", "sox2_free", "sox2_bound", "mRNA".
            sim_site_dwell_times_df:
                Polars df storing residence times at any bound state during course of sim.
                With columns "dwell_site" and "dwell_time. 
            sim_reaction_history_df:
                Polars df containing all reactions that occur within bound of simulation.
                Columns "time", "reaction_type", "site_target", "site_paired_with"
        Raises:
            None
        """
        for i in range(self.n_binding_sites):
            if not self.is_free[i] and self.site_bind_times[i] != -1.0:
                self.residence_time_states.append([i, self.t - self.site_bind_times[i]])

        self.sim_variable_states_df = pl.DataFrame({
            "time": self.times,
            "sox2_free": [s[0] for s in self.bulk_states],
            "sox2_bound": [s[1] for s in self.bulk_states],
            "mrna": [s[2] for s in self.bulk_states],
        })
        
        self.sim_site_dwell_times_df = pl.DataFrame(
            {"dwell_site": [s[0] for s in self.residence_time_states], "dwell_time": [s[1] for s in self.residence_time_states]},
            schema={"dwell_site": pl.Int64, "dwell_time": pl.Float64},
        )

        self.sim_reaction_history_df = pl.DataFrame(
            self.reaction_history, 
            schema=["time", "reaction_type", "site_target", "site_paired_with"]
        )            
        return self.sim_variable_states_df, self.sim_site_dwell_times_df, self.sim_reaction_history_df

    def run_trajectory(self):
        """Runs the Gillespie simulation.
        
        Contains all the Gillespie logic.
        
        Args:
            None
        Returns:
            self._generate_dataframes():
                Method call to generate outputs dataframes. 
                See _generate_dataframes docstring for more information.
        Raises:
            None
        """
        self._initialize_state()

        while self.t < self.t_max:
            propensities, total_prop = self._calculate_propensities()
            if total_prop == 0:
                break
            
            r1, r2 = np.random.random(2)
            t_next_reaction = (1.0 / total_prop) * np.log(1.0 / r1)
            
            self._record_snapshot(t_next_reaction)
            self.t += t_next_reaction

            cumulative_props = np.cumsum(propensities)
            reaction_index = np.searchsorted(cumulative_props, r2 * total_prop)
            self.parameter_states += self.stoichiometry_matrix[:, reaction_index]

            self._execute_spatial_reaction(reaction_index)

        return self._generate_dataframes()