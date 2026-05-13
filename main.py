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

"""
TODO: 
    can probably simplify the dimerisation functions
    so to the unbinding function
    add in nanog-nanog/nanog-sox2 in the nanog binding case?
    check tethering logic

"""

class ModelContext:
    ""



class ModelCall:
    """Handle a single model simulation for a given set of parameters/variables. 
    
    Returns dataframes detailing reaction history and dwell times for a single simulation.
    
    Attributes:
        initial_model_state (dict): Trajectory parameters (e.g., initial states for sox2_free, nanog_free, sox2_bound, nanog_bound, mrna_count).
        model_rate_constants (dict): Reaction kinetic parameters (k_prod_s, k_prod_n, k_deg_s, k_deg_n, k_bind_s, k_bind_n, etc.).
        n_binding_sites (int): Number of binding sites in chromatin lattice.
        t_max (int): Max time to run simulation for.
        record_interval (float): Simulation time interval to record reactions for.
        track_history (bool): If True, save reaction history.
    """
    def __init__(self, model_param: dict, model_var: dict, model_binding_sites: int, sim_max_time: int, record_interval: float = 1.0, track_history: bool = True, promoter_site: int = None):
        self.t_max: int = sim_max_time
        self.record_interval = record_interval
        self.initial_model_state: dict = model_param
        self.model_rate_constants: dict = model_var
        self.n_binding_sites: int = model_binding_sites
        self.track_history = track_history
        self.chromatin_lattice = np.zeros(self.n_binding_sites, dtype=np.int8)
        
        if promoter_site is None:
            self.promoter_site = int((self.n_binding_sites - 1) / 2)
        else:
            self.promoter_site = promoter_site

    def _initialise_state(self):
        """Initialise simulation variables, parameters, and chromatin array.
        
        Function responsible for building:
            molecule_counts array which tracks species count status through simulation.
            self.t which stores curent simulation time.
            a kernel matrix dist_matrix to weight dimerisation distances. 
            a stoichiometry matrix dynamically based on state_names and reaction_names.
        Also tracks chromatin array status all with with length N of self.n_binding_sites via:
            site_is_vacant array that tracks the bound/unbound state of an array.
            unpaired_monomer_mask that tracks the dimerised/monomer state of all bound molecules.
            dimer_partner_map that builds a matrix containing all potential dimerisable site.s
            dangling_tf_map which builds a matrix sotring all potential tetherable sites.
            
        Also initiates arrays:
            self.times, self.bulk_states, self.spatial_states, self.reaction_history, self.residence_time_states
            (sim time), (molecule counts), (chromatin states), (history of all reactions), (dwell time for a molecule at a site)
        """
        # build count tracking array molecule_counts from state_names
        self.state_names = [
            "sox2_monomer_free", "nanog_monomer_free", "sox2_monomer_bound", "nanog_monomer_bound",
            "nanog_sox2_dimer_bound", "nanog_nanog_dimer_bound",
            "nanog_sox2_dimer_free", "nanog_nanog_dimer_free", 
            "nanog_sox2_dimer_single_bound", "nanog_nanog_dimer_single_bound", "mRNA"
        ]
        
        self.reaction_names = {
            0: "prod_s", 1: "prod_n", 2: "bind_s", 3: "bind_n", 
            4: "deg_s", 5: "deg_n", 6: "unbind_s", 7: "unbind_n",
            8: "prod_m", 9: "deg_m", 10: "site_dimerise", 11: "bulk_dimerise",
            12: "tether_bind", 13: "site_dedimerise", 14: "bulk_dedimerise"
        }

        self.state_map = {name: i for i, name in enumerate(self.state_names)}
        self.molecule_counts = np.zeros(len(self.state_names), dtype=np.int32)
        for name, idx in self.state_map.items():
            self.molecule_counts[idx] = self.initial_model_state.get(name, 0)
        self.t = 0.0
        self.next_record_time = 0.0
        
        # track chromatin bound status, dimer status, tethered status.
        self.site_is_vacant = np.ones(self.n_binding_sites, dtype=bool)
        self.total_unbound_sites = np.sum(self.site_is_vacant)
        self.unpaired_monomer_mask = np.zeros(self.n_binding_sites, dtype=bool)
        self.dimer_partner_map = np.full(self.n_binding_sites, -1, dtype=np.int32) # index value holds index it is paired with otherwise - 1
        self.promoter_site = int((len(self.chromatin_lattice) - 1)/2)
        self.dangling_tf_map = np.full(self.n_binding_sites, -1, dtype=np.int32) # index value holds tf type of tether (e.g. 1, 2) otherwise - 1
        
        # define kernel matrix to weight dimerisation distances.
        chromatin_indices = np.arange(self.n_binding_sites)
        dist_matrix = np.abs(chromatin_indices[:, None] - chromatin_indices[None, :]) 
        self.kernel_matrix = np.exp(-dist_matrix / 1)
        np.fill_diagonal(self.kernel_matrix, 0.0)
        self.current_pair_weights = np.zeros_like(self.kernel_matrix)

        # dynamically build stoichiometry matrix dependent on state_names, reaction_names variables.
        # down is self.state_names, across is self.reaction_names
        self.stoichiometry_matrix = np.zeros((len(self.state_names), len(self.reaction_names)), dtype=np.int32)
        bulk_species_counts = self.state_map
        self.stoichiometry_matrix[bulk_species_counts["sox2_monomer_free"], 0] = 1   # prod_s
        self.stoichiometry_matrix[bulk_species_counts["nanog_monomer_free"], 1] = 1  # prod_n
        self.stoichiometry_matrix[bulk_species_counts["sox2_monomer_free"], 4] = -1  # deg_s
        self.stoichiometry_matrix[bulk_species_counts["nanog_monomer_free"], 5] = -1 # deg_n
        self.stoichiometry_matrix[bulk_species_counts["mRNA"], 8] = 1                # prod_m
        self.stoichiometry_matrix[bulk_species_counts["mRNA"], 9] = -1               # deg_m
        self.times, self.bulk_states, self.spatial_states, self.reaction_history, self.residence_time_states, self.dimer_partner_states, self.tethered_states = [], [], [], [], [], [], []
        self.site_bind_times = np.full(self.n_binding_sites, -1.0)

    def _calculate_propensities(self):
        """Update and return new propensities and molecule counts
        
        Function 
        
        Returns:
            propensities: np array describing propensity of all reactions which can occur during the simulation. 
            sum of the propensities. 
        
        """
        bulk_species_counts = self.state_map
        rate_constants = self.model_rate_constants      
        
        # update bulk states
        sox2_monomer_free = self.molecule_counts[bulk_species_counts['sox2_monomer_free']]
        nanog_monomer_free = self.molecule_counts[bulk_species_counts['nanog_monomer_free']]
        sox2_monomer_bound = self.molecule_counts[bulk_species_counts['sox2_monomer_bound']]
        nanog_monomer_bound = self.molecule_counts[bulk_species_counts['nanog_monomer_bound']]
        nanog_sox2_dimer_free = self.molecule_counts[bulk_species_counts['nanog_sox2_dimer_free']]
        nanog_nanog_dimer_free = self.molecule_counts[bulk_species_counts['nanog_nanog_dimer_free']]
        nanog_nanog_dimer_bound = self.molecule_counts[bulk_species_counts['nanog_nanog_dimer_bound']]
        nanog_sox2_dimer_bound = self.molecule_counts[bulk_species_counts['nanog_sox2_dimer_bound']]
        nanog_sox2_dimer_single_bound = self.molecule_counts[bulk_species_counts['nanog_sox2_dimer_single_bound']]
        nanog_nanog_dimer_single_bound = self.molecule_counts[bulk_species_counts['nanog_nanog_dimer_single_bound']]
        mrna_count = self.molecule_counts[bulk_species_counts['mRNA']]        
        
        unpaired_sox2_mask = (self.chromatin_lattice == 1) & self.unpaired_monomer_mask
        unpaired_nanog_mask = (self.chromatin_lattice == 2) & self.unpaired_monomer_mask
        
        # check for valid pairs
        u_s_mask_2d = unpaired_sox2_mask[:, None] # sox2 matrix
        u_n_mask_2d = unpaired_nanog_mask[:, None] # nanog matrix
        valid_pair_matrix = (u_s_mask_2d & unpaired_nanog_mask) | (u_n_mask_2d & unpaired_sox2_mask) | (u_n_mask_2d & unpaired_nanog_mask) # all valid nanog-nanog, sox2-nanog pairs
        
        # take upper triangle to prevent double-counting symmetric pairs like (i,j) and (j,i)
        valid_pair_matrix = np.triu(valid_pair_matrix, k=1)
        
        self.current_pair_weights = self.kernel_matrix * valid_pair_matrix
        total_pair_weight = np.sum(self.current_pair_weights)
        
        tether_mask = (self.dangling_tf_map != -1)
        tether_valid_matrix = tether_mask[:, None] & self.site_is_vacant[None, :]
        self.current_tether_weights = self.kernel_matrix * tether_valid_matrix

        total_tether_weight = np.sum(self.current_tether_weights)
        
        bulk_nn_pairs = nanog_monomer_free * (nanog_monomer_free - 1) / 2.0
        bulk_sn_pairs = sox2_monomer_free * nanog_monomer_free
        total_bulk_pairs = bulk_nn_pairs + bulk_sn_pairs
        
        promoter_has_sox2 = (
            self.chromatin_lattice[self.promoter_site] == 1 or 
            (self.chromatin_lattice[self.promoter_site] == 2 and 
            self.dimer_partner_map[self.promoter_site] != -1 and 
            self.chromatin_lattice[self.dimer_partner_map[self.promoter_site]] == 1)
        )
        
        # probably when selecting bind reaction, randomly choose which species sox2, sox2-nanog to bind.
        
        propensities = np.array([
                rate_constants.get('k_s_in', 0.0),                               # 0
                rate_constants.get('k_n_in', 0.0),                               # 1
                rate_constants.get('k_bind_s', 0.0) * (sox2_monomer_free + nanog_sox2_dimer_free) * self.total_unbound_sites,  # 2
                rate_constants.get('k_bind_n', 0.0) * (nanog_monomer_free + nanog_nanog_dimer_free) * self.total_unbound_sites,  # 3
                rate_constants.get('k_s_out', 0.0) * sox2_monomer_free,                       # 4
                rate_constants.get('k_n_out', 0.0) * nanog_monomer_free,                       # 5
                rate_constants.get('k_unbind_s', 0.0) * (sox2_monomer_bound + nanog_sox2_dimer_bound),               # 6
                rate_constants.get('k_unbind_n', 0.0) * (nanog_monomer_bound + nanog_nanog_dimer_bound),               # 7
                rate_constants.get('k_prod_m', 0.0) if promoter_has_sox2 else 0.0, # 8
                rate_constants.get('k_deg_m', 0.0) * mrna_count,                   # 9
                rate_constants.get('k_dimerise', 0.0) * total_pair_weight,         # site dimerisation     
                rate_constants.get('k_dimerise', 0.0) * total_bulk_pairs, # bulk dimerisation 
                rate_constants.get('k_tether_bind', rate_constants.get('k_dimerise', 0.0)) * total_tether_weight,                  # 12: tether_bind
                rate_constants.get('k_dissociate', 0.0) * (nanog_sox2_dimer_bound + nanog_nanog_dimer_bound),                 # 13: site_dedimerise
                rate_constants.get('k_dissociate', 0.0) * (nanog_sox2_dimer_free + nanog_nanog_dimer_free),                   # 14: bulk_dedimerise
        ])
        return propensities, np.sum(propensities)
    
    ## helper functions for reactions
    
    def _update_matrices(self):
        pass
    
    def _sample_spatial_pair(self, weight_matrix, total_weight):
        """Helper: Randomly selects a 2D spatial pair based on flattened weights."""
        flat_weights = weight_matrix.flatten()
        chosen_idx = np.random.choice(len(flat_weights), p=(flat_weights / total_weight))
        return divmod(chosen_idx, self.n_binding_sites)

    def _link_dimer_sites(self, site1, site2):
        """Helper: Forms a full bridge between two sites on the lattice."""
        self.unpaired_monomer_mask[site1] = False
        self.unpaired_monomer_mask[site2] = False 
        self.dimer_partner_map[site1] = site2
        self.dimer_partner_map[site2] = site1

    def _unlink_dimer_sites(self, site1, site2):
        """Helper: Breaks a bridge between two sites on the lattice."""
        self.unpaired_monomer_mask[site1] = True
        self.unpaired_monomer_mask[site2] = True
        self.dimer_partner_map[site1] = -1
        self.dimer_partner_map[site2] = -1

    def _is_heterodimer(self, tf1_type, tf2_type):
        """Helper: Returns True if the pair is SOX2:NANOG, False if NANOG:NANOG."""
        return (tf1_type == 1 and tf2_type == 2) or (tf1_type == 2 and tf2_type == 1)
    
    def _handle_tf_binding(self, reaction_index):
        """
        Handle all binding reactions. 
        
        Randomly choose site of binding via primary_site. Flip site vacancy to False. Update chromatin_lattice with new TF present at index primary_site.
        """        
        primary_site, secondary_site = -1, -1
        bulk_species_counts = self.state_map
        
        free_indices = np.where(self.site_is_vacant)[0]
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

        
        total_tf_free = self.molecule_counts[bulk_species_counts[monomer]] + self.molecule_counts[bulk_species_counts[dimer]]
        if total_tf_free <= 0: return
        is_dimer = (np.random.rand() < self.molecule_counts[bulk_species_counts[dimer]]/ total_tf_free) if total_tf_free > 0 else False
        
        # if nanog then choose nanognanog or nanogsox2
        # choose which dimer nanogsox2 or nanognanog in the case we are looking at nanog
        
        primary_site = np.random.choice(free_indices)
        self.site_is_vacant[primary_site] = False
        self.site_bind_times[primary_site] = self.t
        self.total_unbound_sites -= 1

        # bind a dimer
        if is_dimer:
                self.chromatin_lattice[primary_site] = tf_type
                self.dangling_tf_map[primary_site] = 2
                self.molecule_counts[bulk_species_counts[dimer]] -= 1
                self.molecule_counts[bulk_species_counts[dimer_single]] += 1
        # bind a monomer
        else:
            self.unpaired_monomer_mask[primary_site] = True
            self.chromatin_lattice[primary_site] = tf_type
            self.molecule_counts[bulk_species_counts[monomer]] -= 1
            self.molecule_counts[bulk_species_counts[monomer_bound]] += 1

        self.reaction_history.append((self.t, self.reaction_names[reaction_index], primary_site, secondary_site))
    
    def _handle_tf_unbinding(self, reaction_index):
        primary_site, secondary_site = -1, -1
        bulk_species_counts = self.state_map

        tf_type = 1 if reaction_index == 6 else 2 # TF unbinding is SOX2 if reaction 6 otherwise NANOG
        bound_indices = np.where(self.chromatin_lattice == tf_type)[0]
        
        if len(bound_indices) > 0:
            dissociating_site = np.random.choice(bound_indices)
            duration = self.t - self.site_bind_times[dissociating_site]
            self.site_bind_times[dissociating_site] = -1.0 
                            
            # calculate residence time
            if self.dimer_partner_map[dissociating_site] != -1:
                paired_site_type = self.chromatin_lattice[self.dimer_partner_map[dissociating_site]]
                if (tf_type == 1 and paired_site_type == 2) or (tf_type == 2 and paired_site_type == 1):
                    species_label = "SOX2:NANOG"
                else:
                    species_label = "NANOG:NANOG"
            else:
                species_label = "SOX2" if tf_type == 1 else "NANOG"

            self.residence_time_states.append([dissociating_site, duration, species_label])                
            
            self.site_bind_times[dissociating_site] = -1.0 
            self.site_is_vacant[dissociating_site] = True
            self.chromatin_lattice[dissociating_site] = 0  
            self.total_unbound_sites += 1  
            # site was a dimer bound to two sites
            if self.dimer_partner_map[dissociating_site] != -1:
                paired_site = self.dimer_partner_map[dissociating_site]
                paired_site_tf_type = self.chromatin_lattice[paired_site]
                
                # remove pair
                self.dimer_partner_map[dissociating_site] = -1
                self.dimer_partner_map[paired_site] = -1
                
                self.dangling_tf_map[paired_site] = tf_type

                if (tf_type == 1 and paired_site_tf_type == 2) or (tf_type == 2 and paired_site_tf_type == 1):
                    self.molecule_counts[bulk_species_counts['nanog_sox2_dimer_bound']] -= 1
                    self.molecule_counts[bulk_species_counts['nanog_sox2_dimer_single_bound']] += 1
                elif tf_type == 2 and paired_site_tf_type == 2:
                    self.molecule_counts[bulk_species_counts['nanog_nanog_dimer_bound']] -= 1
                    self.molecule_counts[bulk_species_counts['nanog_nanog_dimer_single_bound']] += 1
            # site was a single bound dimer
            elif self.dangling_tf_map[dissociating_site] != -1:
                tether_tf_type = self.dangling_tf_map[dissociating_site]
                self.dangling_tf_map[dissociating_site] = -1
                
                if (tf_type == 1 and tether_tf_type == 2) or (tf_type == 2 and tether_tf_type == 1):
                    self.molecule_counts[bulk_species_counts['nanog_sox2_dimer_single_bound']] -= 1
                    self.molecule_counts[bulk_species_counts['nanog_sox2_dimer_free']] += 1
                elif tf_type == 2 and tether_tf_type == 2:
                    self.molecule_counts[bulk_species_counts['nanog_nanog_dimer_single_bound']] -= 1
                    self.molecule_counts[bulk_species_counts['nanog_nanog_dimer_free']] += 1

            # site found was a monomer  / individually bound dimer
            else:
                self.unpaired_monomer_mask[dissociating_site] = False
                if tf_type == 1:
                    self.molecule_counts[bulk_species_counts['sox2_monomer_bound']] -= 1
                    self.molecule_counts[bulk_species_counts['sox2_monomer_free']] += 1
                else:
                    self.molecule_counts[bulk_species_counts['nanog_monomer_bound']] -= 1
                    self.molecule_counts[bulk_species_counts['nanog_monomer_free']] += 1
                    
            primary_site = dissociating_site
            self.reaction_history.append((self.t, self.reaction_names[reaction_index], primary_site, secondary_site))      

    def _execute_site_dimerise(self, reaction_index):
        """Reaction 10: Handles site-site dimerisation on the chromatin lattice."""
        total_w = np.sum(self.current_pair_weights)        
        
        if total_w > 0:
            dwell_site, next_site = self._sample_spatial_pair(self.current_pair_weights, total_w)
            
            tf1_type = self.chromatin_lattice[dwell_site]
            tf2_type = self.chromatin_lattice[next_site]
            
            # Update bulk counts based on dimer type
            bulk = self.state_map
            if self._is_heterodimer(tf1_type, tf2_type):
                self.molecule_counts[bulk['nanog_sox2_dimer_bound']] += 1
                self.molecule_counts[bulk['sox2_monomer_bound']] -= 1
                self.molecule_counts[bulk['nanog_monomer_bound']] -= 1
            else:
                self.molecule_counts[bulk['nanog_nanog_dimer_bound']] += 1
                self.molecule_counts[bulk['nanog_monomer_bound']] -= 2                
            
            # Link sites spatially
            self._link_dimer_sites(dwell_site, next_site)
            self.reaction_history.append((self.t, self.reaction_names[reaction_index], dwell_site, next_site))

    def _execute_bulk_dimerise(self, reaction_index):
        """Reaction 11: Handles dimerisation of free monomers in the bulk."""
        bulk_species_counts = self.state_map
        s_free = self.molecule_counts[bulk_species_counts['sox2_monomer_free']]
        n_free = self.molecule_counts[bulk_species_counts['nanog_monomer_free']]
        
        nn_pairs = n_free * (n_free - 1) / 2.0
        sn_pairs = s_free * n_free
        total_pairs = nn_pairs + sn_pairs
        
        if total_pairs > 0:
            is_nanog = (np.random.rand() < nn_pairs / total_pairs)
            
            if is_nanog and n_free >= 2:
                self.molecule_counts[bulk_species_counts['nanog_nanog_dimer_free']] += 1
                self.molecule_counts[bulk_species_counts['nanog_monomer_free']] -= 2
            elif not is_nanog and s_free >= 1 and n_free >= 1:
                self.molecule_counts[bulk_species_counts['nanog_sox2_dimer_free']] += 1
                self.molecule_counts[bulk_species_counts['sox2_monomer_free']] -= 1
                self.molecule_counts[bulk_species_counts['nanog_monomer_free']] -= 1
        
        self.reaction_history.append((self.t, self.reaction_names[reaction_index], -1, -1))

    def _execute_tether_bind(self, reaction_index):
        """Reaction 12: Handles a single-bound dimer binding its free head to a vacant site."""
        total_w = np.sum(self.current_tether_weights)
        
        if total_w > 0:
            tethered_site, target_vacant_site = self._sample_spatial_pair(self.current_tether_weights, total_w)
            
            tether_tf_type = self.dangling_tf_map[tethered_site]
            bound_tf_type = self.chromatin_lattice[tethered_site]
            
            # Bind the free head to the new site
            self.site_is_vacant[target_vacant_site] = False
            self.site_bind_times[target_vacant_site] = self.t
            self.chromatin_lattice[target_vacant_site] = tether_tf_type
            self.total_unbound_sites -= 1
            
            # Form full bridge and remove tether status
            self._link_dimer_sites(tethered_site, target_vacant_site)
            self.dangling_tf_map[tethered_site] = -1
            
            # Update bulk parameters 
            bulk = self.state_map
            if self._is_heterodimer(bound_tf_type, tether_tf_type):
                self.molecule_counts[bulk['nanog_sox2_dimer_single_bound']] -= 1
                self.molecule_counts[bulk['nanog_sox2_dimer_bound']] += 1
            else:
                self.molecule_counts[bulk['nanog_nanog_dimer_single_bound']] -= 1
                self.molecule_counts[bulk['nanog_nanog_dimer_bound']] += 1

            self.reaction_history.append((self.t, self.reaction_names[reaction_index], tethered_site, target_vacant_site))

    def _execute_site_dedimerise(self, reaction_index):
        """Reaction 13: Breaks a bound dimer back into individual bound monomers."""
        paired_indices = np.where(self.dimer_partner_map != -1)[0]
        
        if len(paired_indices) > 0:
            dissociating_site = np.random.choice(paired_indices)
            partner_site = self.dimer_partner_map[dissociating_site]
            
            # Unlink spatially
            self._unlink_dimer_sites(dissociating_site, partner_site)
            
            tf1_type = self.chromatin_lattice[dissociating_site]
            tf2_type = self.chromatin_lattice[partner_site]
            
            # Update bulk counts
            bulk = self.state_map
            if self._is_heterodimer(tf1_type, tf2_type):
                self.molecule_counts[bulk['nanog_sox2_dimer_bound']] -= 1
                self.molecule_counts[bulk['sox2_monomer_bound']] += 1
                self.molecule_counts[bulk['nanog_monomer_bound']] += 1
            else:
                self.molecule_counts[bulk['nanog_nanog_dimer_bound']] -= 1
                self.molecule_counts[bulk['nanog_monomer_bound']] += 2
                
            self.reaction_history.append((self.t, self.reaction_names[reaction_index], dissociating_site, partner_site))    
    
    def _execute_bulk_dedimerise(self, reaction_index):
        """Reaction 14: Breaks a free bulk dimer into two free bulk monomers."""
        bulk_species_counts = self.state_map
        ns_free = self.molecule_counts[bulk_species_counts['nanog_sox2_dimer_free']]
        nn_free = self.molecule_counts[bulk_species_counts['nanog_nanog_dimer_free']]
        total_dimers = ns_free + nn_free
        
        if total_dimers > 0:
            is_ns = (np.random.rand() < ns_free / total_dimers)
            
            if is_ns and ns_free >= 1:
                self.molecule_counts[bulk_species_counts['nanog_sox2_dimer_free']] -= 1
                self.molecule_counts[bulk_species_counts['sox2_monomer_free']] += 1
                self.molecule_counts[bulk_species_counts['nanog_monomer_free']] += 1
            elif not is_ns and nn_free >= 1:
                self.molecule_counts[bulk_species_counts['nanog_nanog_dimer_free']] -= 1
                self.molecule_counts[bulk_species_counts['nanog_monomer_free']] += 2
        
        self.reaction_history.append((self.t, self.reaction_names[reaction_index], -1, -1))
    
    def _execute_spatial_reaction(self, reaction_index):
        """Reaction handler for a given reaction determined during run_trajectory call.
        
        The reactions this handles are:
            1. Production of SOX2
            2. Production of NANOG
            3. Binding of SOX2 (handles both binding of dimers and monomers)
            4. Binding of NANOG (handles both binding of dimers and monomers)
            5. Degradation of SOX2
            6. Degradation of NANOG
            7. Unbinding of SOX2 (handles unbinding of dimers and monomers)
            8. Unbinding of NANOG (handles unbinding of dimers and monomers)
            9. Production of mRNA (ON/OFF depending on promoter status)
            10. Degradation of mRNA (proportional to mrNA count)
            11. Dimerisation of site-bound monomers
            12. Dimerisation of bulk monomers
            13. Binding of a dimerised molecule to one site.
            14. Dissociation of sitebound dimer. 
            15. Dissociation of bulk dimer.
        
        Args:
            reaction_index (int): index value referring to the numpy array propensities. 
        
        """
        primary_site, secondary_site = -1, -1
        bulk_species_counts = self.state_map
        
        
        #TODO: remove reaction decision logic and use lookup table to select appropriate reaction.
        reaction_index_map = {
            "SOX2_in": None,
            "NANOG_in": None,
            "SOX2_binding": self._handle_tf_binding(reaction_index),
            "NANOG_binding": self._handle_tf_binding(reaction_index),
            "mRNA_production": None,
            "mRNA_degradation": None,
            "dimerise_sites": self._execute_site_dimerise(reaction_index),
            "dimerise_bulk": self._execute_bulk_dimerise(reaction_index),
            "bind_dimer_head": self._execute_tether_bind(reaction_index),
            "dissociate_dimer_sites": self._execute_site_dedimerise(reaction_index),
            "dissociate_dimer_bulk": self._execute_bulk_dedimerise(reaction_index)
        }
        
        # mRNA production/degradation
        if reaction_index in [8, 9]: 
            if reaction_index == 8: self.molecule_counts[bulk_species_counts['mRNA']] += 1
            else: self.molecule_counts[bulk_species_counts['mRNA']] -= 1
            
            self.reaction_history.append((self.t, self.reaction_names[reaction_index], -1, -1))
        else:
            reaction_index_map[reaction_index]()
        

    def _record_snapshot(self):
        """Record the current states of the simulation."""
        self.times.append(self.t)
        self.bulk_states.append(self.molecule_counts.copy())
        self.spatial_states.append(self.chromatin_lattice.copy())
        self.dimer_partner_states.append(self.dimer_partner_map.copy())
        self.tethered_states.append(self.dangling_tf_map.copy())   
    def _generate_dataframes(self):
        """Setup trajectory output dataframes.
        
        Returns:
            sim_variable_states_df: Polars dataframe containing species counts columns (e.g. sox2_monomer_free, nanog_monomer_free, mRNA) with column time.
                {
                    sox2_monomer_free (i32): [1, 0, ... ]
                    (...) <- all tracked species in simulation. see variable self.state_names to see all these variables. 
                    time (f64): [0.0, 1.0, ... ]
                }
            sim_site_dwell_times_df: Polars dataframe containing binding duration for all bound molecules during the run_trajectory call. 
                {
                    dwell_site (i64): [1, 2, 3, ... ]
                    dwell_time (f64): [14, 21, 2, ... ]
                    species (str): [NANOG, SOX2, NANOG:NANOG, ... ]
                }
            sim_reaction_history_df: Polars dataframe detailing all reactions that occurred during the run_trajectory call.
                {
                    time (f64): [0.0, 1.0, 2.0, ... ]
                    reaction_type (str): [bulk_dimerise, bind_s, unbind_s, ... ]
                    primary_site (i64): [-1, 1, 5, ... ]
                    secondary_site (i64): [ -1, 2, 4, ... ]
                }
        """
        # edge case: calculate residence time at the end of the simulation for existing bound
        for i in range(self.n_binding_sites):
                        if not self.site_is_vacant[i] and self.site_bind_times[i] != -1.0:
                            tf_type = self.chromatin_lattice[i]
                        
                            if self.dimer_partner_map[i] != -1:
                                partner = self.dimer_partner_map[i]
                                partner_type = self.chromatin_lattice[partner]
                                
                                if i > partner: 
                                    continue 
                                    
                                if (tf_type == 1 and partner_type == 2) or (tf_type == 2 and partner_type == 1):
                                    species_label = "SOX2:NANOG"
                                else:
                                    species_label = "NANOG:NANOG"
                            elif self.dangling_tf_map[i] != -1:
                                tether_type = self.dangling_tf_map[i]
                                species_label = "SOX2:NANOG" if (tf_type == 1 and tether_type == 2) or (tf_type == 2 and tether_type == 1) else "NANOG:NANOG"
                            else:
                                species_label = "SOX2" if tf_type == 1 else "NANOG"

                            self.residence_time_states.append([i, self.t - self.site_bind_times[i], species_label])                    
        state_data = {
                name: [s[idx] for s in self.bulk_states]
            for name, idx in self.state_map.items()
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
            self._generate_dataframes(): self.sim_variable_states_df, self.sim_site_dwell_times_df, self.sim_reaction_history_df
        """
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
            self.molecule_counts += self.stoichiometry_matrix[:, reaction_index]

            self._execute_spatial_reaction(reaction_index)

        return self._generate_dataframes()
    
    
model_var = {
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

model_param = {
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
        model_param=model_param, 
        model_var=model_var, 
        model_binding_sites=10, 
        sim_max_time=100, 
        track_history=False
    )
    print(model)
