"""Partition function model and Gillespie model of gene expression given clustering.

This module obtains residence time and mRNA expression from a realisation of a 
single trajectory of the chemical master equation. Additionally, from the trajectory 
parameters, we also determine the expected transcription rate (rho).

Typical usage:
    model_call = ModelCall.run_trajectory(params)

If you face any issues, email: dwl25@ic.ac.uk or danielluo1143@gmail.com 
"""

import math
import os
import itertools
import concurrent.futures

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

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
    where alpha is the rate of transcription, n is the number of bound sites, 
    K_alpha is the constant which sets the rate at which alpha saturates for large n (where K_alpha^-1 << n),
    and zed the sum of the partition function.
    """   
    
    def calculate_Zn(n: int, N: int, K1: float, S: float, c: float) -> float:
        """Calculate partition function for a single n.
        
        Args:
            n: Number of bound sites.
            N: Total sites.
            K1: Intrinsic affinity for sites.
            S: Bulk TF.
            c: Dimensionless binding factor.
        
        Returns:
            A float sum of the first and second summation of the partition function.
        """
        binding_term = math.comb(N, n) * ((K1 * S)**n) # first summation
        
        pairing_sum = sum(
            math.comb(N - n, l) * math.comb(n, l) * math.factorial(l) * (c**l)
            for l in range(min(n, N - n) + 1)
        ) # second summation
        return binding_term * pairing_sum

    @staticmethod
    def calculate_total_Z(N: int, K1: float, S: float, c: float) -> float:
        """Calculate the partition function over all possible bound sites (i.e. n = 0 to N)
        
        Args:
            N: Total sites.
            K1: Intrinsic affinity for sites.
            S: Bulk TF.
            c: Dimensionless binding factor.
            
        Returns:
            The float sum of the partition function over all n from 0 to N.
        """
        max_n = min(N, int(S))
        return sum(PartitionFunction.calculate_Zn(n, N, K1, S, c) for n in range(max_n + 1)) # summation over all n -> N
    
    @staticmethod
    def return_maximal_rho(N: int, K1: float, S: float, c: float, alpha: float) -> float:
        """Calculate Rho in the case where any binding achieves maximal rate.

        First method of calculating the transcription rate rho which is the case where the maximal transcription rate is achieved by the binding of a TF to DNA.
        
        Args:
            N: Total sites.
            K1: Intrinsic affinity for sites.
            S: Bulk TF.
            c: Dimensionless binding factor.
            alpha: Rate of transcription.
                    
        Returns:
            Float sum of the transcription rate rho.
        """
        zed = PartitionFunction.calculate_total_Z(N, K1, S, c)
        return alpha * (1 - (zed**-1)) if zed else 0.0

    @staticmethod
    def return_nonmaximal_rho(N: int, K1: float, S: float, c: float, alpha: float, K_alpha: float = None, mode: str = "constant") -> float:
        """Calculate Rho in cases where binding leads to a multiplicative effect on transcription rate.

        Second and third method of calculating rho where transcription is proportional to the number of bound TF. 
        Switch between the second and third methods by specifying mode as either "linear" or "saturating". 
        See the method return_maximal_rho for the other method of calculating transcription rate. 
        
        Args:
            N: Total sites.
            N_bound: sites that are bound by a TF.
            K1: Intrinsic affinity for sites.
            S: Bulk TF.
            c: Dimensionless binding factor.
            alpha: Rate of transcription.
            K_alpha: Constant which determines the rate at which saturation occurs.
            mode: Chooses the mode of calculation ("constant", "linear", or "saturating").
                    
        Returns:
            Float sum of the transcription rate rho.
        """
        numerator_sum = 0.0 # sum of individual zed 
        max_n = min(N, int(S))
        Z_total = PartitionFunction.calculate_total_Z(N, K1, S, c)
        
        for n in range(1, max_n + 1):
            Zn = PartitionFunction.calculate_Zn(n, N, K1, S, c)
            print(Zn)
            
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
        sox2_model_parameters (dict): Trajectory parameters (e.g., initial states for sox2_free, nanog_free, sox2_bound, nanog_bound, mrna_count).
        sox2_model_variables (dict): Reaction kinetic parameters (k_prod_s, k_prod_n, k_deg_s, k_deg_n, k_bind_s, k_bind_n, etc.).
        n_binding_sites (int): Number of binding sites in chromatin lattice.
        t_max (int): Max time to run simulation for.
        record_interval (float): Simulation time interval to record reactions for.
        track_history (bool): If True, save reaction history.
    """
    def __init__(self, model_param: dict, model_var: dict, model_binding_sites: int, sim_max_time: int, record_interval: float = 1.0, track_history: bool = True):
        self.t_max: int = sim_max_time
        self.record_interval = record_interval
        self.sox2_model_parameters: dict = model_param
        self.sox2_model_variables: dict = model_var
        self.n_binding_sites: int = model_binding_sites
        self.track_history = track_history
        self.chromatin_lattice = np.zeros(self.n_binding_sites, dtype=np.int8)

    def _initialise_state(self):
        """Initialise simulation variables, parameters, and chromatin states."""
        self.state_names = [
            "sox2_monomer_free", "nanog_monomer_free", "sox2_monomer_bound", "nanog_monomer_bound",
            "nanog_sox2_dimer_bound", "nanog_nanog_dimer_bound",
            "nanog_sox2_dimer_free", "nanog_nanog_dimer_free", "mrna"
        ]
        self.state_map = {name: i for i, name in enumerate(self.state_names)}
        self.parameter_states = np.zeros(len(self.state_names), dtype=np.int32)
        for name, idx in self.state_map.items():
            self.parameter_states[idx] = self.sox2_model_parameters.get(name, 0)
        self.t = 0.0
        self.next_record_time = 0.0
        
        self.is_free = np.ones(self.n_binding_sites, dtype=bool)
        self.is_unpaired_bound = np.zeros(self.n_binding_sites, dtype=bool)
        self.bridged_to = np.full(self.n_binding_sites, -1, dtype=np.int32)
        self.promoter_site = int((len(self.chromatin_lattice) - 1)/2)
        
        # Kernel Matrix for pairing distances
        chromatin_indices = np.arange(self.n_binding_sites)
        dist_matrix = np.abs(chromatin_indices[:, None] - chromatin_indices[None, :]) 
        self.kernel_matrix = np.exp(-dist_matrix / 1)
        np.fill_diagonal(self.kernel_matrix, 0.0)
        self.current_pair_weights = np.zeros_like(self.kernel_matrix)

        self.stoichiometry_matrix = np.array([
                [ 1,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0], # 0: s_monomer_free
                [ 0,  1,  0,  0,  0, -1,  0,  0,  0,  0,  0], # 1: n_monomer_free
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 2: s_monomer_bound
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 3: n_monomer_bound
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 4: ns_dimer_bound 
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 5: nn_dimer_bound 
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 6: ns_dimer_free
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 7: nn_dimer_free
                [ 0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  0], # 8: mrna            
        ], dtype=np.int32)
        
        
        self.reaction_names = {
            0: "prod_s", 1: "prod_n", 2: "bind_s", 3: "bind_n", 
            4: "deg_s", 5: "deg_n", 6: "unbind_s", 7: "unbind_n",
            8: "prod_m", 9: "deg_m", 10: "k_dimerise"
        }

        self.times, self.bulk_states, self.spatial_states, self.reaction_history, self.residence_time_states = [], [], [], [], []
        self.site_bind_times = np.full(self.n_binding_sites, -1.0)

    def _calculate_propensities(self):
        """Update and return new propensities."""
        v = self.sox2_model_variables
        m = self.state_map
        
        sox2_monomer_free = self.parameter_states[m['sox2_monomer_free']]
        nanog_monomer_free = self.parameter_states[m['nanog_monomer_free']]
        sox2_monomer_bound = self.parameter_states[m['sox2_monomer_bound']]
        nanog_monomer_bound = self.parameter_states[m['nanog_monomer_bound']]
        nanog_sox2_dimer_free = self.parameter_states[m['nanog_sox2_dimer_free']]
        nanog_nanog_dimer_free = self.parameter_states[m['nanog_nanog_dimer_free']]
        nanog_nanog_dimer_bound = self.parameter_states[m['nanog_nanog_dimer_bound']]
        nanog_sox2_dimer_bound = self.parameter_states[m['nanog_sox2_dimer_bound']]

        mrna_count = self.parameter_states[m['mrna']]        
        unbound_sites = np.sum(self.is_free)
        
        u_s_mask = (self.chromatin_lattice == 1) & self.is_unpaired_bound
        u_n_mask = (self.chromatin_lattice == 2) & self.is_unpaired_bound
        
        # check for valid pairs
        u_s_mask_2d = u_s_mask[:, None] # sox2 matrix
        u_n_mask_2d = u_n_mask[:, None] # nanog matrix
        valid_pair_matrix = (u_s_mask_2d & u_n_mask) | (u_n_mask_2d & u_s_mask) | (u_n_mask_2d & u_n_mask) # all valid nanog-nanog, sox2-nanog pairs
        
        # take upper triangle to prevent double-counting symmetric pairs like (i,j) and (j,i)
        valid_pair_matrix = np.triu(valid_pair_matrix, k=1)
        
        self.current_pair_weights = self.kernel_matrix * valid_pair_matrix
        total_pair_weight = np.sum(self.current_pair_weights)

        
        promoter_has_sox2 = (
            self.chromatin_lattice[self.promoter_site] == 1 or 
            (self.chromatin_lattice[self.promoter_site] == 2 and 
            self.bridged_to[self.promoter_site] != -1 and 
            self.chromatin_lattice[self.bridged_to[self.promoter_site]] == 1)
        )
        
        # probably when selecting bind reaction, randomly choose which species sox2, sox2-nanog to bind.
        
        propensities = np.array([
                v.get('k_prod_s', 0.0),                               # 0
                v.get('k_prod_n', 0.0),                               # 1
                v.get('k_bind_s', 0.0) * (sox2_monomer_free + nanog_sox2_dimer_free) * unbound_sites,  # 2
                v.get('k_bind_n', 0.0) * (nanog_monomer_free + nanog_nanog_dimer_free) * unbound_sites,  # 3
                v.get('k_deg_s', 0.0) * sox2_monomer_free,                       # 4
                v.get('k_deg_n', 0.0) * nanog_monomer_free,                       # 5
                v.get('k_unbind_s', 0.0) * (sox2_monomer_bound + nanog_sox2_dimer_bound),               # 6
                v.get('k_unbind_n', 0.0) * (nanog_monomer_bound + nanog_nanog_dimer_bound),               # 7
                v.get('k_prod_m', 0.0) if promoter_has_sox2 else 0.0, # 8
                v.get('k_deg_m', 0.0) * mrna_count,                   # 9
                v.get('k_dimerise', 0.0) * total_pair_weight          # 10            ])
        ])
        return propensities, np.sum(propensities)

    def _execute_spatial_reaction(self, reaction_index):
        """Track updates to the chromatin lattice array in case of binding/sliding."""
        site_target, site_paired_with = -1, -1
        m = self.state_map
        
        if reaction_index in [2, 3]:  # bind reaction            
            free_indices = np.where(self.is_free)[0]
            if len(free_indices) == 0: return
            
            tf_type = 1 if reaction_index == 2 else 2 # 1 = Sox2, 2 = Nanog
            
            if tf_type == 1:
                m_free = self.parameter_states[m['sox2_monomer_free']]
                d_free = self.parameter_states[m['nanog_sox2_dimer_free']]
            else:
                m_free = self.parameter_states[m['nanog_monomer_free']]
                d_free = self.parameter_states[m['nanog_nanog_dimer_free']]
            total_free = m_free + d_free
            if total_free <= 0: return
            
            is_dimer = (np.random.rand() < d_free / total_free) if total_free > 0 else False
            
            # bind dimer
            if is_dimer and len(free_indices) >= 2:
                sites = np.random.choice(free_indices, 2, replace=False)
                s1, s2 = sites[0], sites[1]
                
                self.is_free[s1] = self.is_free[s2] = False
                self.is_unpaired_bound[s1] = self.is_unpaired_bound[s2] = False
                self.site_bind_times[s1] = self.site_bind_times[s2] = self.t
                self.bridged_to[s1], self.bridged_to[s2] = s2, s1
                
                if tf_type == 1: # NS dimer
                    self.chromatin_lattice[s1], self.chromatin_lattice[s2] = 1, 2
                    self.parameter_states[m['nanog_sox2_dimer_free']] -= 1
                    self.parameter_states[m['nanog_sox2_dimer_bound']] += 1
                else: # NN dimer
                    self.chromatin_lattice[s1], self.chromatin_lattice[s2] = 2, 2
                    self.parameter_states[m['nanog_nanog_dimer_free']] -= 1
                    self.parameter_states[m['nanog_nanog_dimer_bound']] += 1
            
            # bind monomer            
            elif len(free_indices) >= 1:
                site_target = np.random.choice(free_indices)
                self.is_free[site_target] = False
                self.is_unpaired_bound[site_target] = True
                self.site_bind_times[site_target] = self.t
                self.chromatin_lattice[site_target] = tf_type
                
                if tf_type == 1:
                    self.parameter_states[m['sox2_monomer_free']] -= 1
                    self.parameter_states[m['sox2_monomer_bound']] += 1
                else:
                    self.parameter_states[m['nanog_monomer_free']] -= 1
                    self.parameter_states[m['nanog_monomer_bound']] += 1
            
        elif reaction_index in [6, 7]:  # unbind reaction
            tf_type = 1 if reaction_index == 6 else 2
            bound_indices = np.where(self.chromatin_lattice == tf_type)[0]
            
            if len(bound_indices) > 0:
                chosen_site = np.random.choice(bound_indices)
                duration = self.t - self.site_bind_times[chosen_site]
                self.residence_time_states.append([chosen_site, duration])
                self.site_bind_times[chosen_site] = -1.0 
                
                self.is_free[chosen_site] = True
                self.chromatin_lattice[chosen_site] = 0
                
                # site was a dimer
                if self.bridged_to[chosen_site] != -1:
                    paired_site = self.bridged_to[chosen_site]
                    paired_site_type = self.chromatin_lattice[paired_site]
                    
                    self.bridged_to[chosen_site] = -1
                    self.bridged_to[paired_site] = -1
                    
                    self.is_free[paired_site] = True
                    self.chromatin_lattice[paired_site] = 0
                    self.site_bind_times[paired_site] = -1.0
                    
                    # decrement bound and increment free
                    if (tf_type == 1 and paired_site_type == 2) or (tf_type == 2 and paired_site_type == 1):
                        self.parameter_states[m['nanog_sox2_dimer_bound']] -= 1
                        self.parameter_states[m['nanog_sox2_dimer_free']] += 1
                    elif tf_type == 2 and paired_site_type == 2:
                        self.parameter_states[m['nanog_nanog_dimer_bound']] -= 1
                        self.parameter_states[m['nanog_nanog_dimer_free']] += 1
                # site found was a monomer   
                else:
                    self.is_unpaired_bound[chosen_site] = False
                    if tf_type == 1:
                        self.parameter_states[m['sox2_monomer_bound']] -= 1
                        self.parameter_states[m['sox2_monomer_free']] += 1
                    else:
                        self.parameter_states[m['nanog_monomer_bound']] -= 1
                        self.parameter_states[m['nanog_monomer_free']] += 1
                        
                site_target = chosen_site
                site_paired_with = 0
        elif reaction_index == 10:  # dimerise two bound tfs
            total_w = np.sum(self.current_pair_weights)
            if total_w > 0:
                # randomly choose pair
                flat_weights = self.current_pair_weights.flatten()
                chosen_idx = np.random.choice(len(flat_weights), p=(flat_weights / total_w))
                dwell_site, next_site = divmod(chosen_idx, self.n_binding_sites)
                site_target, site_paired_with = dwell_site, next_site
                
                chromatin_dwell_site = self.chromatin_lattice[dwell_site]
                chromatin_next_site = self.chromatin_lattice[next_site]
                
                if (chromatin_dwell_site == 1 and chromatin_next_site == 2) or (chromatin_dwell_site == 2 and chromatin_next_site == 1):
                    self.parameter_states[m['nanog_sox2_dimer_bound']] += 1
                    self.parameter_states[m['sox2_monomer_bound']] -= 1
                    self.parameter_states[m['nanog_monomer_bound']] -= 1
                elif chromatin_dwell_site == 2 and chromatin_next_site == 2:
                    self.parameter_states[m['nanog_nanog_dimer_bound']] += 1
                    self.parameter_states[m['nanog_monomer_bound']] -= 2                
                
                self.is_unpaired_bound[dwell_site] = self.is_unpaired_bound[next_site] = False 
                
                self.bridged_to[dwell_site], self.bridged_to[next_site] = next_site, dwell_site
                site_target, site_paired_with = dwell_site, next_site
                
        if self.track_history and site_target != -1:
            self.reaction_history.append((self.t, self.reaction_names[reaction_index], site_target, site_paired_with))

    def _record_snapshot(self):
        """Record the current states of the simulation."""
        self.times.append(self.t)
        self.bulk_states.append(self.parameter_states.copy())
        self.spatial_states.append(self.chromatin_lattice.copy())

    def _generate_dataframes(self):
        """Setup model output dataframes."""
        for i in range(self.n_binding_sites):
                if not self.is_free[i] and self.site_bind_times[i] != -1.0:
                    self.residence_time_states.append([i, self.t - self.site_bind_times[i]])
        state_data = {
                name: [s[idx] for s in self.bulk_states]
            for name, idx in self.state_map.items()
        }
        state_data["time"] = self.times
        
        self.sim_variable_states_df = pl.DataFrame(state_data)
        
        self.sim_site_dwell_times_df = pl.DataFrame(
            {"dwell_site": [s[0] for s in self.residence_time_states], 
            "dwell_time": [s[1] for s in self.residence_time_states]},
            schema={"dwell_site": pl.Int64, "dwell_time": pl.Float64},
        )
        self.sim_reaction_history_df = pl.DataFrame(
            self.reaction_history, 
            schema=["time", "reaction_type", "site_target", "site_paired_with"]
        )      
        return self.sim_variable_states_df, self.sim_site_dwell_times_df, self.sim_reaction_history_df

    def run_trajectory(self):
        """Run the Gillespie simulation."""
        self._initialise_state()

        while self.t < self.t_max:
            propensities, total_prop = self._calculate_propensities()
            if total_prop == 0:
                break
            
            r1, r2 = np.random.random(2)
            t_next_reaction = (1.0 / total_prop) * np.log(1.0 / r1)
            
            self._record_snapshot()
            self.t += t_next_reaction

            cumulative_props = np.cumsum(propensities)
            reaction_index = np.searchsorted(cumulative_props, r2 * total_prop)
            self.parameter_states += self.stoichiometry_matrix[:, reaction_index]

            self._execute_spatial_reaction(reaction_index)

        return self._generate_dataframes()