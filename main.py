"""Partition function model and Gillespie model of gene expression given clustering.

This module obtains residence time and mRNA expression from a realisation of a 
single trajectory of the chemical master equation. Additionally, from the trajectory 
parameters, we also determine the expected transcription rate (rho).

Typical usage:
    model_call = ModelCall(
        rate_constants=MODEL_RATE_CONSTANTS,
        initial_species_states=INITIAL_MODEL_STATE,
        total_binding_sites=10,
        max_sim_time=100
    )
    df_states, df_dwell, df_rxns = model_call.run_trajectory()
    
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

class ModelBuilder:
    """Handles a single model simulation for a given set of parameters.
    
    Attributes:
        rate_constants (dict): Reaction kinetic parameters (k_prod_s, k_bind_s, etc.).
        initial_species_states (dict): Initial molecule counts for the simulation.
        total_binding_sites (int): Total number of binding sites in the chromatin lattice.
        max_sim_time (int): Maximum time to run the simulation for.
        record_interval (float): Simulation time interval to record reactions.
        track_history (bool): If True, saves a detailed reaction history.
        chromatin_promoter_site (int): The index of the primary promoter site.
    """
    def __init__(self, model_param: dict, model_var: dict, model_binding_sites: int, sim_max_time: int, record_interval: float = 1.0, track_history: bool = True, chromatin_promoter_site: int = None):
        self.model_max_runtime: int = sim_max_time
        self.model_record_interval = record_interval
        self.model_initial_species_states: dict = model_var
        self.model_rate_constants: dict = model_param
        self.model_binding_site_total: int = model_binding_sites
        self.model_track_history = track_history
        self.chromatin_lattice = np.zeros(self.model_binding_site_total, dtype=np.int8)
        
        if chromatin_promoter_site is None:
            self.chromatin_promoter_site = int((self.model_binding_site_total - 1) / 2)
        else:
            self.chromatin_promoter_site = chromatin_promoter_site

    def _initialise_state(self):
        """Initialises simulation variables, parameters, and the chromatin array.
        
        This method is responsible for building tracking arrays, kernel matrices, 
        and the dynamically sized stoichiometry matrix.
        """
        # build count tracking array species_counts from species_names
        self.species_names = [
            "sox2_monomer_free", "nanog_monomer_free", "sox2_monomer_bound", "nanog_monomer_bound",
            "nanog_sox2_dimer_bound", "nanog_nanog_dimer_bound",
            "nanog_sox2_dimer_free", "nanog_nanog_dimer_free", 
            "nanog_sox2_dimer_single_bound", "nanog_nanog_dimer_single_bound", "mRNA"
        ]
        
        self.reaction_names = {
            0: "prod_s", 1: "prod_n", 2: "bind_s", 3: "bind_n", 
            4: "deg_s", 5: "deg_n", 6: "unbind_s", 7: "unbind_n",
            8: "prod_m", 9: "deg_m", 10: "site_dimerise", 11: "bulk_dimerise",
            12: "site_dedimerise", 13: "bulk_dedimerise", 14: "site_bulk_dimerise"
        }

        self.species_state_map = {name: i for i, name in enumerate(self.species_names)}
        self.species_counts = np.zeros(len(self.species_names), dtype=np.int32)
        for name, idx in self.species_state_map.items():
            self.species_counts[idx] = self.model_initial_species_states.get(name, 0)
        self.current_t = 0.0
        self.next_record_time = 0.0
        
        # track chromatin bound status, dimer status, tethered status.
        self.chromatin_site_is_vacant = np.ones(self.model_binding_site_total, dtype=bool)
        self.chromatin_total_unbound_sites = np.sum(self.chromatin_site_is_vacant)
        self.chromatin_all_unpaired_monomers = np.zeros(self.model_binding_site_total, dtype=bool)
        self.chromatin_all_dimers = np.full(self.model_binding_site_total, -1, dtype=np.int32) # index value holds index it is paired with otherwise - 1
        self.chromatin_promoter_site = int((len(self.chromatin_lattice) - 1)/2)
        self.chromatin_all_dangling_tf = np.full(self.model_binding_site_total, -1, dtype=np.int32) # index value holds tf type of tether (e.g. 1, 2) otherwise - 1
        
        self.vacant_sites = set(range(self.model_binding_site_total))
        self.bound_sox2_sites = set()
        self.bound_nanog_sites = set()
        self.paired_dimer_sites = set()
        self.unpaired_sox2_sites = set()
        self.unpaired_nanog_sites = set()
        self.dangling_sox2_count = 0
        self.dangling_nanog_count = 0
        
        # define kernel matrix to weight dimerisation distances.
        chromatin_lattice_indicies = np.arange(self.model_binding_site_total)
        chromatin_dist_matrix = np.abs(chromatin_lattice_indicies[:, None] - chromatin_lattice_indicies[None, :]) 
        self.chromatin_dist_weighted_matrix = np.exp(-chromatin_dist_matrix / 1)
        np.fill_diagonal(self.chromatin_dist_weighted_matrix, 0.0)
        self.chromatin_dimerisation_weights = np.zeros_like(self.chromatin_dist_weighted_matrix)
        
        self.pair_weight_matrix = np.zeros_like(self.chromatin_dist_weighted_matrix)
        self.tether_weight_matrix = np.zeros_like(self.chromatin_dist_weighted_matrix)
        self.total_pair_weight_symmetric = 0.0
        self.total_tether_weight = 0.0

        # dynamically build stoichiometry matrix dependent on species_names, reaction_names variables.
        # down is self.species_names, across is self.reaction_names
        self.stoichiometry_matrix = np.zeros((len(self.species_names), len(self.reaction_names)), dtype=np.int32)
        bulk_species_counts = self.species_state_map
        self.stoichiometry_matrix[bulk_species_counts["sox2_monomer_free"], 0] = 1   # prod_s
        self.stoichiometry_matrix[bulk_species_counts["nanog_monomer_free"], 1] = 1  # prod_n
        self.stoichiometry_matrix[bulk_species_counts["sox2_monomer_free"], 4] = -1  # deg_s
        self.stoichiometry_matrix[bulk_species_counts["nanog_monomer_free"], 5] = -1 # deg_n
        self.stoichiometry_matrix[bulk_species_counts["mRNA"], 8] = 1                # prod_m
        self.stoichiometry_matrix[bulk_species_counts["mRNA"], 9] = -1               # deg_m
        self.times, self.bulk_states, self.spatial_states, self.reaction_history, self.residence_time_states, self.dimer_partner_states, self.tethered_states = [], [], [], [], [], [], []
        self.site_bind_times = np.full(self.model_binding_site_total, -1.0)

    def _update_site_weights(self, site):
        """Optimisation 1: Incrementally update spatial weight matrices for a single site."""
        
        # --- Update Pair Weights (Symmetric Matrix) ---
        old_pair_row_sum = np.sum(self.pair_weight_matrix[site, :])
        new_pair_row = np.zeros(self.model_binding_site_total)
        
        if self.chromatin_all_unpaired_monomers[site]:
            tf_type = self.chromatin_lattice[site]
            if tf_type == 1:
                valid_mask = (self.chromatin_lattice == 2) & self.chromatin_all_unpaired_monomers
            elif tf_type == 2:
                valid_mask = ((self.chromatin_lattice == 1) | (self.chromatin_lattice == 2)) & self.chromatin_all_unpaired_monomers
            else:
                valid_mask = np.zeros(self.model_binding_site_total, dtype=bool)
            
            valid_mask[site] = False
            new_pair_row[valid_mask] = self.chromatin_dist_weighted_matrix[site, valid_mask]
        
        self.pair_weight_matrix[site, :] = new_pair_row
        self.pair_weight_matrix[:, site] = new_pair_row
        self.total_pair_weight_symmetric += (np.sum(new_pair_row) - old_pair_row_sum) * 2

        # --- Update Tether Weights (Asymmetric Matrix) ---
        old_tether_row_sum = np.sum(self.tether_weight_matrix[site, :])
        new_tether_row = np.zeros(self.model_binding_site_total)
        if self.chromatin_all_dangling_tf[site] != -1:
            vacant_mask = self.chromatin_site_is_vacant
            new_tether_row[vacant_mask] = self.chromatin_dist_weighted_matrix[site, vacant_mask]
            new_tether_row[site] = 0.0
        
        self.tether_weight_matrix[site, :] = new_tether_row
        self.total_tether_weight += (np.sum(new_tether_row) - old_tether_row_sum)
        
        old_tether_col_sum = np.sum(self.tether_weight_matrix[:, site])
        new_tether_col = np.zeros(self.model_binding_site_total)
        if self.chromatin_site_is_vacant[site]:
            tether_mask = (self.chromatin_all_dangling_tf != -1)
            new_tether_col[tether_mask] = self.chromatin_dist_weighted_matrix[tether_mask, site]
            new_tether_col[site] = 0.0
            
        self.tether_weight_matrix[:, site] = new_tether_col
        self.total_tether_weight += (np.sum(new_tether_col) - old_tether_col_sum)

    def _set_site_state(self, site, is_vacant=None, tf_type=None, is_unpaired=None, dangling_tf_type=None, dimer_partner=None):
        """Centralized state updater to keep arrays, sets, and spatial matrices perfectly in sync."""
        # Update TF Type
        if tf_type is not None:
            old_tf = self.chromatin_lattice[site]
            if old_tf == 1: self.bound_sox2_sites.discard(site)
            elif old_tf == 2: self.bound_nanog_sites.discard(site)
            self.chromatin_lattice[site] = tf_type
            if tf_type == 1: self.bound_sox2_sites.add(site)
            elif tf_type == 2: self.bound_nanog_sites.add(site)

        # Update Vacancy
        if is_vacant is not None:
            self.chromatin_site_is_vacant[site] = is_vacant
            if is_vacant: self.vacant_sites.add(site)
            else: self.vacant_sites.discard(site)
            self.chromatin_total_unbound_sites = len(self.vacant_sites)

        # Update Unpaired Monomer Status
        if is_unpaired is not None:
            self.chromatin_all_unpaired_monomers[site] = is_unpaired
            current_tf = self.chromatin_lattice[site]
            if is_unpaired:
                if current_tf == 1:
                    self.unpaired_sox2_sites.add(site)
                    self.unpaired_nanog_sites.discard(site)
                elif current_tf == 2:
                    self.unpaired_nanog_sites.add(site)
                    self.unpaired_sox2_sites.discard(site)
            else:
                self.unpaired_sox2_sites.discard(site)
                self.unpaired_nanog_sites.discard(site)

        # Update Dangling Tethers
        if dangling_tf_type is not None:
            old_dangling = self.chromatin_all_dangling_tf[site]
            if old_dangling == 1: self.dangling_sox2_count -= 1
            elif old_dangling == 2: self.dangling_nanog_count -= 1
            self.chromatin_all_dangling_tf[site] = dangling_tf_type
            if dangling_tf_type == 1: self.dangling_sox2_count += 1
            elif dangling_tf_type == 2: self.dangling_nanog_count += 1

        # Update Dimer Partners
        if dimer_partner is not None:
            self.chromatin_all_dimers[site] = dimer_partner
            if dimer_partner != -1: self.paired_dimer_sites.add(site)
            else: self.paired_dimer_sites.discard(site)
                
        # Fire O(1) Matrix Adjustments
        self._update_site_weights(site)

    def _calculate_propensities(self):
        """Calculates current reaction propensities based on the system state.
        
        Returns:
            tuple: A tuple containing:
                - propensities (np.ndarray): 1D array of reaction propensities.
                - total_propensity (float): The sum of all propensities.
        """
        bulk_species_counts = self.species_state_map
        rate_constants = self.model_rate_constants      
        
        # update bulk states
        (
            sox2_monomer_free,
            nanog_monomer_free,
            sox2_monomer_bound,
            nanog_monomer_bound,
            nanog_sox2_dimer_bound,
            nanog_nanog_dimer_bound,
            nanog_sox2_dimer_free,
            nanog_nanog_dimer_free,
            nanog_sox2_dimer_single_bound,
            nanog_nanog_dimer_single_bound,
            mrna_count
        ) = self.species_counts
        
        total_pair_weight = self.total_pair_weight_symmetric / 2.0
        total_tether_weight = self.total_tether_weight
        
        # pairs for bulk monomer-bulk monomer
        bulk_nn_pairs = nanog_monomer_free * (nanog_monomer_free - 1) / 2.0
        bulk_sn_pairs = sox2_monomer_free * nanog_monomer_free
        total_bulk_pairs = bulk_nn_pairs + bulk_sn_pairs
        
        # pairs for bound monomer-free monomer dimerisation
        bulk_sn_bound_pairs = sox2_monomer_bound * nanog_monomer_free
        bulk_ns_bound_pairs = nanog_monomer_bound * sox2_monomer_free
        bulk_nn_bound_pairs = nanog_monomer_bound * nanog_monomer_free
        total_site_bulk_pairs = bulk_sn_bound_pairs + bulk_ns_bound_pairs + bulk_nn_bound_pairs
        
        promoter_has_sox2 = (
            self.chromatin_lattice[self.chromatin_promoter_site] == 1 or 
            (self.chromatin_lattice[self.chromatin_promoter_site] == 2 and 
            self.chromatin_all_dimers[self.chromatin_promoter_site] != -1 and 
            self.chromatin_lattice[self.chromatin_all_dimers[self.chromatin_promoter_site]] == 1)
        )
        
        # probably when selecting bind reaction, randomly choose which species sox2, sox2-nanog to bind.
        
        propensities = np.array([
                rate_constants.get('k_s_in', 0.0),                               # 0
                rate_constants.get('k_n_in', 0.0),                               # 1
                rate_constants.get('k_bind_s', 0.0) * (sox2_monomer_free + nanog_sox2_dimer_free + self.dangling_sox2_count) * self.chromatin_total_unbound_sites,    # 2
                rate_constants.get('k_bind_n', 0.0) * (nanog_monomer_free + nanog_nanog_dimer_free + self.dangling_nanog_count) * self.chromatin_total_unbound_sites,  # 3
                rate_constants.get('k_s_out', 0.0) * sox2_monomer_free,                       # 4
                rate_constants.get('k_n_out', 0.0) * nanog_monomer_free,                       # 5
                rate_constants.get('k_unbind_s', 0.0) * (sox2_monomer_bound + nanog_sox2_dimer_bound),               # 6
                rate_constants.get('k_unbind_n', 0.0) * (nanog_monomer_bound + nanog_nanog_dimer_bound),               # 7
                rate_constants.get('k_prod_m', 0.0) if promoter_has_sox2 else 0.0, # 8
                rate_constants.get('k_deg_m', 0.0) * mrna_count,                   # 9
                rate_constants.get('k_dimerise', 0.0) * total_pair_weight,         # site dimerisation     
                rate_constants.get('k_dimerise', 0.0) * total_bulk_pairs, # bulk dimerisation 
                rate_constants.get('k_dissociate', 0.0) * (nanog_sox2_dimer_bound + nanog_nanog_dimer_bound),                 # 12: site_dedimerise
                rate_constants.get('k_dissociate', 0.0) * (nanog_sox2_dimer_free + nanog_nanog_dimer_free),                   # 13: bulk_dedimerise
                rate_constants.get('k_dimerise', 0.0) * total_site_bulk_pairs
        ])
        return propensities, np.sum(propensities)
    
    ## helper functions for reactions
    
    def _update_matrices(self):
        pass
        
    

class ModelBinding():
    """Mixin class to handle transcription factor binding and unbinding."""
    def _handle_tf_binding(self, reaction_index):
        """Handles binding of free monomers or dimers to the chromatin lattice.
        
        Args:
            reaction_index (int): The index of the binding reaction (2 or 3).
        """        
        primary_site, secondary_site = -1, -1
        bulk_species_counts = self.species_state_map
        
        free_indices = np.where(self.chromatin_site_is_vacant)[0]
        if len(free_indices) == 0: return
        
        if reaction_index == 2: 
            tf_type = 1 
            monomer = "sox2_monomer_free"
            monomer_bound = "sox2_monomer_bound"
            dimer = "nanog_sox2_dimer_free"
            dimer_single = "nanog_sox2_dimer_single_bound"
        else: 
            tf_type = 2
            monomer = "nanog_monomer_free"
            monomer_bound = "nanog_monomer_bound"
            dimer = "nanog_nanog_dimer_free"
            dimer_single = "nanog_nanog_dimer_single_bound"
        
        
        count_monomer = self.species_counts[bulk_species_counts[monomer]]
        count_dimer = self.species_counts[bulk_species_counts[dimer]]
        count_tether = np.sum(self.chromatin_all_dangling_tf == tf_type)
        
        count_total_free_tfs = count_monomer + count_dimer + count_tether
        if count_total_free_tfs <= 0: return
        rand_val = np.random.rand() * count_total_free_tfs

        # bind the head of a dimer
        if rand_val >= (count_monomer + count_dimer) and count_tether > 0:
            # We selected a dangling tether
            total_w = self.total_tether_weight
            
            if total_w > 0:
                tethered_site, target_vacant_site = self._sample_spatial_pair(self.tether_weight_matrix, is_tether=True)
                bound_tf_type = self.chromatin_lattice[tethered_site]
                
                self._set_site_state(target_vacant_site, is_vacant=False, tf_type=tf_type)
                self.site_bind_times[target_vacant_site] = self.current_t
                self._link_dimer_sites(tethered_site, target_vacant_site)
                self._set_site_state(tethered_site, dangling_tf_type=-1)                

                self._link_dimer_sites(tethered_site, target_vacant_site)
                self.chromatin_all_dangling_tf[tethered_site] = -1
                
                if self._is_heterodimer(bound_tf_type, tf_type):
                    self.species_counts[bulk_species_counts['nanog_sox2_dimer_single_bound']] -= 1
                    self.species_counts[bulk_species_counts['nanog_sox2_dimer_bound']] += 1
                else:
                    self.species_counts[bulk_species_counts['nanog_nanog_dimer_single_bound']] -= 1
                    self.species_counts[bulk_species_counts['nanog_nanog_dimer_bound']] += 1

                self.reaction_history.append((self.current_t, self.reaction_names[reaction_index], tethered_site, target_vacant_site))
            return
        
        is_dimer = (rand_val >= count_monomer)
        primary_site = np.random.choice(free_indices)

        # bind a dimer
        if is_dimer:
            self._set_site_state(primary_site, is_vacant=False, tf_type=tf_type, dangling_tf_type=2)
            self.species_counts[bulk_species_counts[dimer]] -= 1
            self.species_counts[bulk_species_counts[dimer_single]] += 1
        # bind a monomer
        else:
            self._set_site_state(primary_site, is_vacant=False, tf_type=tf_type, is_unpaired=True)            
            self.species_counts[bulk_species_counts[monomer]] -= 1
            self.species_counts[bulk_species_counts[monomer_bound]] += 1

        self.reaction_history.append((self.current_t, self.reaction_names[reaction_index], primary_site, secondary_site))
    
    def _handle_tf_unbinding(self, reaction_index):
        """Handles the dissociation of bound elements from the lattice.
        
        Args:
            reaction_index (int): The index of the unbinding reaction (6 or 7).
        """
        primary_site, secondary_site = -1, -1
        bulk_species_counts = self.species_state_map

        tf_type = 1 if reaction_index == 6 else 2 # TF unbinding is SOX2 if reaction 6 otherwise NANOG
        bound_indices = np.where(self.chromatin_lattice == tf_type)[0]
        
        if len(bound_indices) > 0:
            dissociating_site = np.random.choice(bound_indices)
            duration = self.current_t - self.site_bind_times[dissociating_site]
            self.site_bind_times[dissociating_site] = -1.0 
            
            old_dimer_partner = self.chromatin_all_dimers[dissociating_site]
            old_dangling = self.chromatin_all_dangling_tf[dissociating_site]
               
            # calculate residence time
            if old_dimer_partner != -1:
                paired_site_type = self.chromatin_lattice[old_dimer_partner]
                species_label = "SOX2:NANOG" if self._is_heterodimer(tf_type, paired_site_type) else "NANOG:NANOG"
            else:
                species_label = "SOX2" if tf_type == 1 else "NANOG"
            self.residence_time_states.append([dissociating_site, duration, species_label])                
            
            self.site_bind_times[dissociating_site] = -1.0 
            self._set_site_state(dissociating_site, is_vacant=True, tf_type=0, is_unpaired=False, dangling_tf_type=-1, dimer_partner=-1)
            # site was a dimer bound to two sites
            if old_dimer_partner != -1:
                paired_site_tf_type = self.chromatin_lattice[old_dimer_partner]
                self._set_site_state(old_dimer_partner, dimer_partner=-1, dangling_tf_type=tf_type)

                if self._is_heterodimer(tf_type, paired_site_tf_type):
                    self.species_counts[bulk_species_counts['nanog_sox2_dimer_bound']] -= 1
                    self.species_counts[bulk_species_counts['nanog_sox2_dimer_single_bound']] += 1
                else:
                    self.species_counts[bulk_species_counts['nanog_nanog_dimer_bound']] -= 1
                    self.species_counts[bulk_species_counts['nanog_nanog_dimer_single_bound']] += 1
                    
                    
            # site was a single bound dimer
            elif old_dangling != -1:
                if self._is_heterodimer(tf_type, old_dangling):
                    self.species_counts[bulk_species_counts['nanog_sox2_dimer_single_bound']] -= 1
                    self.species_counts[bulk_species_counts['nanog_sox2_dimer_free']] += 1
                else:
                    self.species_counts[bulk_species_counts['nanog_nanog_dimer_single_bound']] -= 1
                    self.species_counts[bulk_species_counts['nanog_nanog_dimer_free']] += 1

            # site found was a monomer  / individually bound dimer
            else:
                if tf_type == 1:
                    self.species_counts[bulk_species_counts['sox2_monomer_bound']] -= 1
                    self.species_counts[bulk_species_counts['sox2_monomer_free']] += 1
                else:
                    self.species_counts[bulk_species_counts['nanog_monomer_bound']] -= 1
                    self.species_counts[bulk_species_counts['nanog_monomer_free']] += 1
                    
            self.reaction_history.append((self.current_t, self.reaction_names[reaction_index], primary_site, secondary_site))      
  
class ModelDimers():
    """Mixin class to handle dimerisation mechanics and updates."""
    def _sample_spatial_pair(self, weight_matrix, total_weight):
        """Randomly selects a 2D spatial pair based on flattened weights.
        
        Args:
            weight_matrix (np.ndarray): Probability matrix of spatial pairs.
            total_weight (float): Sum of the probability matrix.
            
        Returns:
            tuple: The (site_1, site_2) indices chosen.
        """
        flat_weights = weight_matrix.flatten()
        chosen_idx = np.random.choice(len(flat_weights), p=(flat_weights / np.sum(flat_weights)))
        return divmod(chosen_idx, self.model_binding_site_total)

    def _link_dimer_sites(self, site1, site2):
        """Forms a spatial bridge between two sites on the lattice.
        
        Args:
            site1 (int): The first site index.
            site2 (int): The second site index.
        """
        self._set_site_state(site1, is_unpaired=False, dimer_partner=site2)
        self._set_site_state(site2, is_unpaired=False, dimer_partner=site1)

    def _unlink_dimer_sites(self, site1, site2):
        """Breaks a spatial bridge between two sites on the lattice.
        
        Args:
            site1 (int): The first site index.
            site2 (int): The second site index.
        """
        self._set_site_state(site1, is_unpaired=True, dimer_partner=-1)
        self._set_site_state(site2, is_unpaired=True, dimer_partner=-1)
    
    def _is_heterodimer(self, tf1_type, tf2_type):
        """Returns True if the pair is SOX2:NANOG, False if NANOG:NANOG.
        
        Args:
            tf1_type (int): The TF type at site 1.
            tf2_type (int): The TF type at site 2.
            
        Returns:
            bool: True if heterodimer, False otherwise.
        """
        return (tf1_type == 1 and tf2_type == 2) or (tf1_type == 2 and tf2_type == 1)

    def _execute_site_dimerise(self, reaction_index):
        """Handles site-site dimerisation on the chromatin lattice.
        
        Args:
            reaction_index (int): Index of the site dimerisation reaction (10).
        """
        if self.total_pair_weight_symmetric > 0:
            dwell_site, next_site = self._sample_spatial_pair(self.pair_weight_matrix)
            tf1_type = self.chromatin_lattice[dwell_site]
            tf2_type = self.chromatin_lattice[next_site]
            
            bulk = self.species_state_map
            if self._is_heterodimer(tf1_type, tf2_type):
                self.species_counts[bulk['nanog_sox2_dimer_bound']] += 1
                self.species_counts[bulk['sox2_monomer_bound']] -= 1
                self.species_counts[bulk['nanog_monomer_bound']] -= 1
            else:
                self.species_counts[bulk['nanog_nanog_dimer_bound']] += 1
                self.species_counts[bulk['nanog_monomer_bound']] -= 2                
            
            self._link_dimer_sites(dwell_site, next_site)
            self.reaction_history.append((self.current_t, self.reaction_names[reaction_index], dwell_site, next_site))

    def _execute_bulk_dimerise(self, reaction_index):
        """Handles dimerisation of free monomers in the bulk state.
        
        Args:
            reaction_index (int): Index of the bulk dimerisation reaction (11).
        """
        bulk = self.species_state_map
        s_free = self.species_counts[bulk['sox2_monomer_free']]
        n_free = self.species_counts[bulk['nanog_monomer_free']]
        nn_pairs = n_free * (n_free - 1) / 2.0
        sn_pairs = s_free * n_free
        total_pairs = nn_pairs + sn_pairs
        
        if total_pairs > 0:
            is_nanog = (np.random.rand() < nn_pairs / total_pairs)
            if is_nanog and n_free >= 2:
                self.species_counts[bulk['nanog_nanog_dimer_free']] += 1
                self.species_counts[bulk['nanog_monomer_free']] -= 2
            elif not is_nanog and s_free >= 1 and n_free >= 1:
                self.species_counts[bulk['nanog_sox2_dimer_free']] += 1
                self.species_counts[bulk['sox2_monomer_free']] -= 1
                self.species_counts[bulk['nanog_monomer_free']] -= 1
        self.reaction_history.append((self.current_t, self.reaction_names[reaction_index], -1, -1))

    def _execute_tether_bind(self, reaction_index):
        """Handles a single-bound dimer binding its free head to a vacant site.
        
        Args:
            reaction_index (int): Index of the tether binding reaction (12).
        """        
        bulk = self.species_state_map
        s_bound = self.species_counts[bulk['sox2_monomer_bound']]
        n_bound = self.species_counts[bulk['nanog_monomer_bound']]
        s_free = self.species_counts[bulk['sox2_monomer_free']]
        n_free = self.species_counts[bulk['nanog_monomer_free']]

        sn_pairs = s_bound * n_free
        ns_pairs = n_bound * s_free
        nn_pairs = n_bound * n_free
        total_pairs = sn_pairs + ns_pairs + nn_pairs

        if total_pairs > 0:
            rand_val = np.random.rand() * total_pairs
            site = -1
            
            if rand_val < sn_pairs:
                valid_sites = list(self.unpaired_sox2_sites)
                if len(valid_sites) > 0:
                    site = np.random.choice(valid_sites)
                    self._set_site_state(site, is_unpaired=False, dangling_tf_type=2)
                    self.species_counts[bulk['sox2_monomer_bound']] -= 1
                    self.species_counts[bulk['nanog_monomer_free']] -= 1
                    self.species_counts[bulk['nanog_sox2_dimer_single_bound']] += 1
            elif rand_val < sn_pairs + ns_pairs:
                valid_sites = list(self.unpaired_nanog_sites)
                if len(valid_sites) > 0:
                    site = np.random.choice(valid_sites)
                    self._set_site_state(site, is_unpaired=False, dangling_tf_type=1)
                    self.species_counts[bulk['nanog_monomer_bound']] -= 1
                    self.species_counts[bulk['sox2_monomer_free']] -= 1
                    self.species_counts[bulk['nanog_sox2_dimer_single_bound']] += 1
            else:
                valid_sites = list(self.unpaired_nanog_sites)
                if len(valid_sites) > 0:
                    site = np.random.choice(valid_sites)
                    self._set_site_state(site, is_unpaired=False, dangling_tf_type=2)
                    self.species_counts[bulk['nanog_monomer_bound']] -= 1
                    self.species_counts[bulk['nanog_monomer_free']] -= 1
                    self.species_counts[bulk['nanog_nanog_dimer_single_bound']] += 1

            self.reaction_history.append((self.current_t, self.reaction_names[reaction_index], site, -1))

    def _execute_site_dedimerise(self, reaction_index):
        """Breaks a bound dimer back into individual bound monomers.
        
        Args:
            reaction_index (int): Index of the site dedimerisation reaction (13).
        """        
        paired_indices = list(self.paired_dimer_sites)
        if len(paired_indices) > 0:
            dissociating_site = np.random.choice(paired_indices)
            partner_site = self.chromatin_all_dimers[dissociating_site]
            
            tf1_type = self.chromatin_lattice[dissociating_site]
            tf2_type = self.chromatin_lattice[partner_site]
            
            self._unlink_dimer_sites(dissociating_site, partner_site)
            
            bulk = self.species_state_map
            if self._is_heterodimer(tf1_type, tf2_type):
                self.species_counts[bulk['nanog_sox2_dimer_bound']] -= 1
                self.species_counts[bulk['sox2_monomer_bound']] += 1
                self.species_counts[bulk['nanog_monomer_bound']] += 1
            else:
                self.species_counts[bulk['nanog_nanog_dimer_bound']] -= 1
                self.species_counts[bulk['nanog_monomer_bound']] += 2
                
            self.reaction_history.append((self.current_t, self.reaction_names[reaction_index], dissociating_site, partner_site))
    
    def _execute_bulk_dedimerise(self, reaction_index):
        """Breaks a free bulk dimer into two free bulk monomers.
        
        Args:
            reaction_index (int): Index of the bulk dedimerisation reaction (14).
        """
        bulk = self.species_state_map
        ns_free = self.species_counts[bulk['nanog_sox2_dimer_free']]
        nn_free = self.species_counts[bulk['nanog_nanog_dimer_free']]
        total_dimers = ns_free + nn_free
        
        if total_dimers > 0:
            is_ns = (np.random.rand() < ns_free / total_dimers)
            if is_ns and ns_free >= 1:
                self.species_counts[bulk['nanog_sox2_dimer_free']] -= 1
                self.species_counts[bulk['sox2_monomer_free']] += 1
                self.species_counts[bulk['nanog_monomer_free']] += 1
            elif not is_ns and nn_free >= 1:
                self.species_counts[bulk['nanog_nanog_dimer_free']] -= 1
                self.species_counts[bulk['nanog_monomer_free']] += 2
        
        self.reaction_history.append((self.current_t, self.reaction_names[reaction_index], -1, -1))
        
        
    def _execute_site_bulk_dimerise(self, reaction_index):
        """Handles dimerisation between a bound monomer and a free bulk monomer, forming a single-bound dimer."""
        bulk = self.species_state_map
        s_bound = self.species_counts[bulk['sox2_monomer_bound']]
        n_bound = self.species_counts[bulk['nanog_monomer_bound']]
        s_free = self.species_counts[bulk['sox2_monomer_free']]
        n_free = self.species_counts[bulk['nanog_monomer_free']]

        sn_pairs = s_bound * n_free
        ns_pairs = n_bound * s_free
        nn_pairs = n_bound * n_free
        total_pairs = sn_pairs + ns_pairs + nn_pairs

        if total_pairs > 0:
            rand_val = np.random.rand() * total_pairs
            
            if rand_val < sn_pairs:
                # SOX2 bound + NANOG free -> NANOG:SOX2 single bound
                valid_sites = np.where((self.chromatin_lattice == 1) & self.chromatin_all_unpaired_monomers)[0]
                if len(valid_sites) > 0:
                    site = np.random.choice(valid_sites)
                    self.chromatin_all_unpaired_monomers[site] = False
                    self.chromatin_all_dangling_tf[site] = 2 # dangling NANOG
                    
                    self.species_counts[bulk['sox2_monomer_bound']] -= 1
                    self.species_counts[bulk['nanog_monomer_free']] -= 1
                    self.species_counts[bulk['nanog_sox2_dimer_single_bound']] += 1
                
            elif rand_val < sn_pairs + ns_pairs:
                # NANOG bound + SOX2 free -> NANOG:SOX2 single bound
                valid_sites = np.where((self.chromatin_lattice == 2) & self.chromatin_all_unpaired_monomers)[0]
                if len(valid_sites) > 0:
                    site = np.random.choice(valid_sites)
                    self.chromatin_all_unpaired_monomers[site] = False
                    self.chromatin_all_dangling_tf[site] = 1 # dangling SOX2
                    
                    self.species_counts[bulk['nanog_monomer_bound']] -= 1
                    self.species_counts[bulk['sox2_monomer_free']] -= 1
                    self.species_counts[bulk['nanog_sox2_dimer_single_bound']] += 1
                
            else:
                # NANOG bound + NANOG free -> NANOG:NANOG single bound
                valid_sites = np.where((self.chromatin_lattice == 2) & self.chromatin_all_unpaired_monomers)[0]
                if len(valid_sites) > 0:
                    site = np.random.choice(valid_sites)
                    self.chromatin_all_unpaired_monomers[site] = False
                    self.chromatin_all_dangling_tf[site] = 2 # dangling NANOG
                    
                    self.species_counts[bulk['nanog_monomer_bound']] -= 1
                    self.species_counts[bulk['nanog_monomer_free']] -= 1
                    self.species_counts[bulk['nanog_nanog_dimer_single_bound']] += 1

            # Use site variable to log history. If no valid sites were found (edge case), default to -1.
            record_site = site if 'site' in locals() else -1
            self.reaction_history.append((self.current_t, self.reaction_names[reaction_index], record_site, -1))



class ModelCall(ModelBuilder, ModelBinding, ModelDimers):
    """Execution wrapper connecting logic mixins to the builder and main loop."""
    def __init__(self, model_param, model_var, model_binding_sites, sim_max_time, record_interval = 1, track_history = True, chromatin_promoter_site = None):
        ModelBuilder.__init__(self, model_param, model_var, model_binding_sites, sim_max_time, record_interval, track_history, chromatin_promoter_site)

        self._reaction_function_map = {
            0: self._record_bulk_history,
            1: self._record_bulk_history,
            2: self._handle_tf_binding,
            3: self._handle_tf_binding,
            4: self._record_bulk_history,
            5: self._record_bulk_history,
            6: self._handle_tf_unbinding,
            7: self._handle_tf_unbinding,
            8: self._record_bulk_history,
            9: self._record_bulk_history,
            10: self._execute_site_dimerise,
            11: self._execute_bulk_dimerise,
            12: self._execute_site_dedimerise,
            13: self._execute_bulk_dedimerise,
            14: self._execute_site_bulk_dimerise
        }
    
    def _record_bulk_history(self, reaction_index):
        """Logs history for non-spatial bulk reactions.
        
        Args:
            reaction_index (int): Index of the bulk reaction being processed.
        """
        self.reaction_history.append((self.current_t, self.reaction_names[reaction_index], -1, -1))
    
    def _execute_spatial_reaction(self, reaction_index):
        """Dispatches the appropriate reaction logic via function mapping.
        
        Args:
            reaction_index (int): The index referring to the chosen reaction.
        """        
        self._reaction_function_map[reaction_index](reaction_index)   
        print(reaction_index)     

    def _record_snapshot(self):
        """Record the current states of the simulation."""
        self.times.append(self.current_t)
        self.bulk_states.append(self.species_counts.copy())
        self.spatial_states.append(self.chromatin_lattice.copy())
        self.dimer_partner_states.append(self.chromatin_all_dimers.copy())
        self.tethered_states.append(self.chromatin_all_dangling_tf.copy())   
    def _generate_dataframes(self):
        """Packages raw simulation lists into Polars DataFrames.
        
        Returns:
            tuple: A tuple containing:
                - sim_variable_states_df (pl.DataFrame): Species counts over time.
                - sim_site_dwell_times_df (pl.DataFrame): Binding duration data.
                - sim_reaction_history_df (pl.DataFrame): Detailed reaction log.
        """        
        # edge case: calculate residence time at the end of the simulation for existing bound
        for i in range(self.model_binding_site_total):
            if not self.chromatin_site_is_vacant[i] and self.site_bind_times[i] != -1.0:
                tf_type = self.chromatin_lattice[i]
                if self.chromatin_all_dimers[i] != -1:
                    partner = self.chromatin_all_dimers[i]
                    partner_type = self.chromatin_lattice[partner]
                    if i > partner: continue 
                    species_label = "SOX2:NANOG" if self._is_heterodimer(tf_type, partner_type) else "NANOG:NANOG"
                elif self.chromatin_all_dangling_tf[i] != -1:
                    tether_type = self.chromatin_all_dangling_tf[i]
                    species_label = "SOX2:NANOG" if self._is_heterodimer(tf_type, tether_type) else "NANOG:NANOG"
                else:
                    species_label = "SOX2" if tf_type == 1 else "NANOG"
                self.residence_time_states.append([i, self.current_t - self.site_bind_times[i], species_label])
        state_data = {
                name: [s[idx] for s in self.bulk_states]
            for name, idx in self.species_state_map.items()
        }
        state_data["time"] = self.times
        
        self.sim_variable_states_df = pl.DataFrame(state_data)
        self.sim_site_dwell_times_df = pl.DataFrame(
            {"dwell_site": [s[0] for s in self.residence_time_states], 
             "dwell_time": [s[1] for s in self.residence_time_states],
             "species": [s[2] for s in self.residence_time_states]},
            schema={"dwell_site": pl.Int64, "dwell_time": pl.Float64, "species": str},
        )
        self.sim_reaction_history_df = pl.DataFrame(
            self.reaction_history, 
            schema={
                "time": pl.Float64, 
                "reaction_type": pl.String, 
                "primary_site": pl.Int64, 
                "secondary_site": pl.Int64
            },
            orient="row"
        )        
        return self.sim_variable_states_df, self.sim_site_dwell_times_df, self.sim_reaction_history_df

    def run_trajectory(self):
        """Run the Gillespie simulation.
        
        Quick overview of Gillespie algorithm:
        1. Initialise the system.
        2. Monte Carlo -> randomly simulate time to next event given an event has occurred randomly select which event has occured.
        3. Update - based on step 2, move model time forward to the event time and update state of the system.
        4. Repeat steps 2 to 3 till some stopping criteria is achieved.
        
        Returns:
            tuple: the generated dataframes with simulation data: self.sim_variable_states_df, self.sim_site_dwell_times_df, self.sim_reaction_history_df
        """
        self._initialise_state()

        while self.current_t < self.model_max_runtime:
            propensities, total_prop = self._calculate_propensities()
            if total_prop == 0:
                break
            
            r1, r2 = np.random.random(2)
            t_next_reaction = (1.0 / total_prop) * np.log(1.0 / r1)
            
            self._record_snapshot()
            self.current_t += t_next_reaction

            cumulative_props = np.cumsum(propensities)
            reaction_index = np.searchsorted(cumulative_props, r2 * total_prop)
            self.species_counts += self.stoichiometry_matrix[:, reaction_index]

            self._execute_spatial_reaction(reaction_index)

        return self._generate_dataframes()

    
INITIAL_MODEL_STATE = {
    "sox2_monomer_free": 10, 
    "nanog_monomer_free": 1, 
    "sox2_monomer_bound": 0, 
    "nanog_monomer_bound": 0, 
    "nanog_sox2_dimer_bound": 0, 
    "nanog_nanog_dimer_bound": 0,
    "nanog_sox2_dimer_free": 0,
    "nanog_nanog_dimer_free": 0,
    "nanog_sox2_dimer_single_bound": 0,
    "nanog_nanog_dimer_single_bound": 0,
    "mRNA": 0
}

MODEL_RATE_CONSTANTS = {
    "k_s_in": 0, "k_s_out": 0,
    "k_n_in": 0, "k_in_out": 0, 
    "k_bind_s": 1.0, "k_unbind_s": 0.06,
    "k_bind_n": 1.0, "k_unbind_n": 0.24,
    "k_dimerise": 0.0,  
    "k_prod_m": 1.0,    
    "k_deg_m": 0.53, 
    "k_dissociate": 0
}
    
    
if __name__ == '__main__': 
    model = ModelCall(
        model_param=MODEL_RATE_CONSTANTS, 
        model_var=INITIAL_MODEL_STATE, 
        model_binding_sites=10, 
        sim_max_time=100, 
        track_history=False
    )
    df_states, df_dwell, df_rxns = model.run_trajectory()
    print(df_rxns)
