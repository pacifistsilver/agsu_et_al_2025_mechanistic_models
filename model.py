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
import numpy as np
import polars as pl
import pandas as pd
import simulation_config as config

SPECIES_MAP = {name: i for i, name in enumerate(config.SPECIES_NAMES)}


class ModelState:
    """Tracks the state of the simulation and includes commonly called methods. 
    
        Attributes:
            rate_constants (dict): Reaction kinetic parameters (k_prod_s, k_bind_s, etc.).
            initial_species_states (dict): Initial molecule counts for the simulation.
            total_binding_sites (int): Total number of binding sites in the chromatin lattice.
            record_interval (float): Simulation time interval to record reactions.
            promoter_site (int): The index of the primary promoter site.
    """
    def __init__(self, total_binding_sites: int, initial_species_states: dict, promoter_site: int = None):
        self.total_sites = total_binding_sites
        self.promoter_site = promoter_site if promoter_site is not None else int((total_binding_sites - 1) / 2)
        
        self.species_counts = np.zeros(len(config.SPECIES_NAMES), dtype=np.int32)
        for name, idx in SPECIES_MAP.items():
            self.species_counts[idx] = initial_species_states.get(name, 0)
            
        # chromatin tracking arrays
        self.chromatin_lattice = np.zeros(self.total_sites, dtype=np.int8)
        self.chromatin_site_is_vacant = np.ones(self.total_sites, dtype=bool)
        self.chromatin_all_undimered_monomers = np.zeros(self.total_sites, dtype=bool)
        self.chromatin_all_dimers = np.full(self.total_sites, -1, dtype=np.int32) 
        self.chromatin_all_dangling_tfs = np.full(self.total_sites, -1, dtype=np.int32) 
        self.chromatin_site_bind_times = np.full(self.total_sites, -1.0)

        self.vacant_chromatin_sites = set(range(self.total_sites))
        self.bound_sox2_sites = set()
        self.bound_nanog_sites = set()
        self.dimered_dimer_sites = set()
        self.undimered_sox2_sites = set()
        self.undimered_nanog_sites = set()
        self.dangling_sox2_count = 0
        self.dangling_nanog_count = 0
        
        # spatial distance kernel
        chromatin_lattice_indicies = np.arange(self.total_sites)
        chromatin_potential_dimer_partners = np.abs(chromatin_lattice_indicies[:, None] - chromatin_lattice_indicies[None, :]) 
        self.dist_weighted_dimer_partners = np.exp(-chromatin_potential_dimer_partners / 1)
        np.fill_diagonal(self.dist_weighted_dimer_partners, 0.0)

        # increment weights
        self.dimer_weight_matrix = np.zeros_like(self.dist_weighted_dimer_partners)
        self.tether_weight_matrix = np.zeros_like(self.dist_weighted_dimer_partners)
        self.total_dimer_weight_symmetric = 0.0
        self.total_tether_weight = 0.0

    def update_site_weights(self, site):
        """Update spatial weight matrices to decide dimerisation partners."""

        # update dimer weights
        old_dimer_row_sum = np.sum(self.dimer_weight_matrix[site, :])
        new_dimer_row = np.zeros(self.total_sites)
        
        if self.chromatin_all_undimered_monomers[site]:
            tf_type = self.chromatin_lattice[site]
            if tf_type == 1: valid_mask = (self.chromatin_lattice == 2) & self.chromatin_all_undimered_monomers
            elif tf_type == 2: valid_mask = ((self.chromatin_lattice == 1) | (self.chromatin_lattice == 2)) & self.chromatin_all_undimered_monomers
            else: valid_mask = np.zeros(self.total_sites, dtype=bool)
            
            valid_mask[site] = False
            new_dimer_row[valid_mask] = self.dist_weighted_dimer_partners[site, valid_mask]
        
        self.dimer_weight_matrix[site, :] = new_dimer_row
        self.dimer_weight_matrix[:, site] = new_dimer_row
        self.total_dimer_weight_symmetric += (np.sum(new_dimer_row) - old_dimer_row_sum) * 2

        # update tethered weights
        old_tether_row_sum = np.sum(self.tether_weight_matrix[site, :])
        new_tether_row = np.zeros(self.total_sites)
        if self.chromatin_all_dangling_tfs[site] != -1:
            vacant_mask = self.chromatin_site_is_vacant
            new_tether_row[vacant_mask] = self.dist_weighted_dimer_partners[site, vacant_mask]
            new_tether_row[site] = 0.0
        
        self.tether_weight_matrix[site, :] = new_tether_row
        self.total_tether_weight += (np.sum(new_tether_row) - old_tether_row_sum)
        
        old_tether_col_sum = np.sum(self.tether_weight_matrix[:, site])
        new_tether_col = np.zeros(self.total_sites)
        if self.chromatin_site_is_vacant[site]:
            tether_mask = (self.chromatin_all_dangling_tfs != -1)
            new_tether_col[tether_mask] = self.dist_weighted_dimer_partners[tether_mask, site]
            new_tether_col[site] = 0.0
            
        self.tether_weight_matrix[:, site] = new_tether_col
        self.total_tether_weight += (np.sum(new_tether_col) - old_tether_col_sum)

    def set_site_state(self, site, is_vacant=None, tf_type=None, is_undimered=None, dangling_tf_type=None, dimer_partner=None):
        """Update status of a site on the chromatin array when a binding/unbinding/dimerisation reaction occurs. 

        Args:
            site (int): index referring to position in chromatin array
            is_vacant (bool, optional): Update site occupancy status. Defaults to None.
            tf_type (int, optional): Form of TF. i.e. SOX2, NANOG. Defaults to None.
            is_undimered (bool, optional): Track dimer status. Defaults to None.
            dangling_tf_type (int, optional): For a dimerised molecule, this refers to the TF that is not bound to any site. i.e. it is bound to this TF but not bound to anything else. Defaults to None.
            dimer_partner (int, optional): index referring to the dimerised TF partner on the chromatin array. Defaults to None.
        """
        # update bound/unbound sites
        if tf_type is not None:
            old_tf = self.chromatin_lattice[site]
            if old_tf == 1: self.bound_sox2_sites.discard(site)
            elif old_tf == 2: self.bound_nanog_sites.discard(site)
            self.chromatin_lattice[site] = tf_type
            if tf_type == 1: self.bound_sox2_sites.add(site)
            elif tf_type == 2: self.bound_nanog_sites.add(site)

        # update vacancy information for sites
        if is_vacant is not None:
            self.chromatin_site_is_vacant[site] = is_vacant
            if is_vacant: self.vacant_chromatin_sites.add(site)
            else: self.vacant_chromatin_sites.discard(site)

        # dimerisation logic
        if is_undimered is not None:
            self.chromatin_all_undimered_monomers[site] = is_undimered
            current_tf = self.chromatin_lattice[site]
            if is_undimered:
                if current_tf == 1:
                    self.undimered_sox2_sites.add(site)
                    self.undimered_nanog_sites.discard(site)
                elif current_tf == 2:
                    self.undimered_nanog_sites.add(site)
                    self.undimered_sox2_sites.discard(site)
            else:
                self.undimered_sox2_sites.discard(site)
                self.undimered_nanog_sites.discard(site)

        # single-site bound dimer logic
        if dangling_tf_type is not None:
            old_dangling = self.chromatin_all_dangling_tfs[site]
            if old_dangling == 1: self.dangling_sox2_count -= 1
            elif old_dangling == 2: self.dangling_nanog_count -= 1
            self.chromatin_all_dangling_tfs[site] = dangling_tf_type
            if dangling_tf_type == 1: self.dangling_sox2_count += 1
            elif dangling_tf_type == 2: self.dangling_nanog_count += 1

        # dissociation from dimerised state
        if dimer_partner is not None:
            self.chromatin_all_dimers[site] = dimer_partner
            if dimer_partner != -1: self.dimered_dimer_sites.add(site)
            else: self.dimered_dimer_sites.discard(site)
                
        self.update_site_weights(site)

class ModelLogger():
    """_summary_ Methods for recording reactions, and outputting DFs containing time course data for sites, residence times, and a list of all reactions that occur.
    
    Attributes:
        state (class): inherits ModelState class methods and attributes to record all reaction information. 
    """
    def __init__(self, state: ModelState):
        self.state = state
        self.times, self.bulk_states, self.spatial_states = [], [], []
        self.dimer_partner_states, self.tethered_states = [], []
        self.reaction_history, self.residence_time_states = [], []
        
    def record_reaction(self, current_t, reaction_index, primary_site=-1, secondary_site=-1):
        self.reaction_history.append((current_t, config.REACTION_NAMES[reaction_index], primary_site, secondary_site))

    def record_snapshot(self, current_t):
        self.times.append(current_t)
        self.bulk_states.append(self.state.species_counts.copy())
        self.spatial_states.append(self.state.chromatin_lattice.copy())
        self.dimer_partner_states.append(self.state.chromatin_all_dimers.copy())
        self.tethered_states.append(self.state.chromatin_all_dangling_tfs.copy())   

    def generate_dataframes(self, final_t):
        """_summary_ Outputs three dataframes containing simulation data.
        
        Return:
            polars dataframes df_states, df_dwell df_rxns.
        """
        for i in range(self.state.total_sites):
            if not self.state.chromatin_site_is_vacant[i] and self.state.chromatin_site_bind_times[i] != -1.0:
                tf_type = self.state.chromatin_lattice[i]
                if self.state.chromatin_all_dimers[i] != -1:
                    partner = self.state.chromatin_all_dimers[i]
                    partner_type = self.state.chromatin_lattice[partner]
                    if i > partner: continue 
                    species_label = "SOX2:NANOG" if (tf_type == 1 and partner_type == 2) or (tf_type == 2 and partner_type == 1) else "NANOG:NANOG"
                elif self.state.chromatin_all_dangling_tfs[i] != -1:
                    tether_type = self.state.chromatin_all_dangling_tfs[i]
                    species_label = "SOX2:NANOG" if (tf_type == 1 and tether_type == 2) or (tf_type == 2 and tether_type == 1) else "NANOG:NANOG"
                else:
                    species_label = "SOX2" if tf_type == 1 else "NANOG"
                self.residence_time_states.append([final_t - self.state.chromatin_site_bind_times[i], i, species_label])                    
        
        df_states = pl.DataFrame({
            "time": self.times,
            **{name: [s[idx] for s in self.bulk_states] for name, idx in SPECIES_MAP.items()}
        })
           
        df_dwell = pl.DataFrame(
            {"time": [s[0] for s in self.residence_time_states], "dwell_site": [s[1] for s in self.residence_time_states], "species": [s[2] for s in self.residence_time_states]},
            schema={"time": pl.Float64, "dwell_site": pl.Int64, "species": str},
        )
        df_rxns = pl.DataFrame(self.reaction_history, schema={"time": pl.Float64, "reaction_type": pl.String, "primary_site": pl.Int64, "secondary_site": pl.Int64}, orient="row")        
        return df_states, df_dwell, df_rxns   

class ModelReactions():
    """Contains reaction logic methods to update ModelState attributes as dictated by the class ModelCall.
    
    Attributes:
        state (class): inherits simulation state information and important state update methods from ModelState.
        logger (class): inherits ModelLogger methods to track reactions, dwelltimes, etc.
    """
    def __init__(self, state: ModelState, logger: ModelLogger):
        self.state = state
        self.logger = logger
        self.select_reaction_map = {
            0: self._log_only, 1: self._log_only, 4: self._log_only, 5: self._log_only, 8: self._log_only, 9: self._log_only,
            2: self._handle_tf_binding, 3: self._handle_tf_binding, 12: self._handle_tf_binding,
            6: self._handle_tf_unbinding, 7: self._handle_tf_unbinding,
            10: self._execute_site_dimerise, 11: self._execute_bulk_dimerise,
            13: self._execute_site_dedimerise, 14: self._execute_bulk_dedimerise, 15: self._execute_site_bulk_dimerise
        }
    
    def execute(self, current_t, reaction_index):
        self.select_reaction_map[reaction_index](current_t, reaction_index)

    def _log_only(self, current_t, reaction_index):
        self.logger.record_reaction(current_t, reaction_index)

    def _sample_spatial_dimer(self, weight_matrix, is_tether=False):
        """Method selects a random site obtained from weight_matrix to dimerise.

        Args:
            weight_matrix (NDarray): 2D NxN matrix containing site to site weights.
            is_tether (bool, optional): Defaults to False.

        Returns:
            tuple: Randomised index in weight_matrix 
        """
        if not is_tether: weight_matrix = np.triu(weight_matrix, k=1)
        flat_weights = weight_matrix.flatten()
        chosen_idx = np.random.choice(len(flat_weights), p=(flat_weights / np.sum(flat_weights)))
        return divmod(chosen_idx, self.state.total_sites)

    def _link_dimer_sites(self, site1, site2):
        """Updates two indexes on chromatin_lattice by updating chromatin_all_undimered_monomers.

        Args:
            site1 (int): index referring to first half of dimer
            site2 (int): index referring to last half of dimer
        """
        self.state.set_site_state(site1, is_undimered=False, dimer_partner=site2)
        self.state.set_site_state(site2, is_undimered=False, dimer_partner=site1)

    def _is_heterodimer(self, tf1_type, tf2_type):
        """Determines if two dimerised TFs are either hetero (S2-N) (N-S2)/homo (N-N).

        Args:
            tf1_type (int): type of TF
            tf2_type (int): type of TF

        Returns:
            bool: True if heterodimer. False otherwise.
        """
        return (tf1_type == 1 and tf2_type == 2) or (tf1_type == 2 and tf2_type == 1)

    def _handle_tf_binding(self, current_t, reaction_index):
        """Reaction logic for a dimerised/monomer TF binding to a site on chromatin array.

        Args:
            current_t (int): current time of simulation 
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        free_indices = list(self.state.vacant_chromatin_sites)
        if len(free_indices) == 0: return
        
        if reaction_index == 2: 
            tf_type, monomer, monomer_bound = 1, "sox2_monomer_free", "sox2_monomer_bound"
            dimer, dimer_single = "nanog_sox2_dimer_free", "nanog_sox2_dimer_single_bound"
        else: 
            tf_type, monomer, monomer_bound = 2, "nanog_monomer_free", "nanog_monomer_bound"
            dimer, dimer_single = "nanog_nanog_dimer_free", "nanog_nanog_dimer_single_bound"

        count_monomer = self.state.species_counts[SPECIES_MAP[monomer]]
        count_dimer = self.state.species_counts[SPECIES_MAP[dimer]]
        count_tether = self.state.dangling_sox2_count if tf_type == 1 else self.state.dangling_nanog_count
        
        # choose random tf (monomer, dimer, single bound dimer) to bind
        total_tf_pool = count_monomer + count_dimer + count_tether
        if total_tf_pool <= 0: return
        rand_val = np.random.rand() * total_tf_pool

        # an already bound dimer (that is bound to one and only one site) binds an additional site.
        if rand_val >= (count_monomer + count_dimer) and count_tether > 0:
            if self.state.total_tether_weight > 0:
                tethered_site, target_vacant_site = self._sample_spatial_dimer(self.state.tether_weight_matrix, is_tether=True)
                bound_tf_type = self.state.chromatin_lattice[tethered_site]
                
                self.state.set_site_state(target_vacant_site, is_vacant=False, tf_type=tf_type)
                self.state.chromatin_site_bind_times[target_vacant_site] = current_t
                self._link_dimer_sites(tethered_site, target_vacant_site)
                self.state.set_site_state(tethered_site, dangling_tf_type=-1)
                
                if self._is_heterodimer(bound_tf_type, tf_type):
                    self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_single_bound']] -= 1
                    self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_bound']] += 1
                else:
                    self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_single_bound']] -= 1
                    self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_bound']] += 1

                self.logger.record_reaction(current_t, reaction_index, tethered_site, target_vacant_site)
            return

        # if a dimer was chosen to bind
        is_dimer = (rand_val >= count_monomer)
        primary_site = np.random.choice(free_indices)
        self.state.chromatin_site_bind_times[primary_site] = current_t
        
        # the bound dimer is now a single site bound dimer
        if is_dimer:
            self.state.set_site_state(primary_site, is_vacant=False, tf_type=tf_type, dangling_tf_type=2)
            self.state.species_counts[SPECIES_MAP[dimer]] -= 1
            self.state.species_counts[SPECIES_MAP[dimer_single]] += 1
        else:
            self.state.set_site_state(primary_site, is_vacant=False, tf_type=tf_type, is_undimered=True)
            self.state.species_counts[SPECIES_MAP[monomer]] -= 1
            self.state.species_counts[SPECIES_MAP[monomer_bound]] += 1

        self.logger.record_reaction(current_t, reaction_index, primary_site)
    
    def _handle_tf_unbinding(self, current_t, reaction_index):
        """Reaction logic for a TF unbinding from a site on the chromatin array.

        Args:
            current_t (int): current time of simulation 
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        tf_type = 1 if reaction_index == 6 else 2 
        bound_indices = list(self.state.bound_sox2_sites) if tf_type == 1 else list(self.state.bound_nanog_sites)
        
        if len(bound_indices) > 0:
            dissociating_site = np.random.choice(bound_indices)
            duration = current_t - self.state.chromatin_site_bind_times[dissociating_site]
            
            # check for dimerisation partners/tethered status
            dimer_partner_index = self.state.chromatin_all_dimers[dissociating_site]
            old_dangling = self.state.chromatin_all_dangling_tfs[dissociating_site]
                        
            # for dissociating_site, there exists a dimer at some index.
            if dimer_partner_index != -1:
                dimered_type = self.state.chromatin_lattice[dimer_partner_index]
                species_label = "SOX2:NANOG" if self._is_heterodimer(tf_type, dimered_type) else "NANOG:NANOG"
            else:
                species_label = "SOX2" if tf_type == 1 else "NANOG"

            self.logger.residence_time_states.append([duration, dissociating_site, species_label])                
            self.state.chromatin_site_bind_times[dissociating_site] = -1.0 
            self.state.set_site_state(dissociating_site, is_vacant=True, tf_type=0, is_undimered=False, dangling_tf_type=-1, dimer_partner=-1)

            # update species counts accordingly.
            if dimer_partner_index != -1:
                self.state.set_site_state(dimer_partner_index, dimer_partner=-1, dangling_tf_type=tf_type)
                if self._is_heterodimer(tf_type, dimered_type):
                    self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_bound']] -= 1
                    self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_single_bound']] += 1
                else:
                    self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_bound']] -= 1
                    self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_single_bound']] += 1
            elif old_dangling != -1:
                if self._is_heterodimer(tf_type, old_dangling):
                    self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_single_bound']] -= 1
                    self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_free']] += 1
                else:
                    self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_single_bound']] -= 1
                    self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_free']] += 1
            else:
                if tf_type == 1:
                    self.state.species_counts[SPECIES_MAP['sox2_monomer_bound']] -= 1
                    self.state.species_counts[SPECIES_MAP['sox2_monomer_free']] += 1
                else:
                    self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] -= 1
                    self.state.species_counts[SPECIES_MAP['nanog_monomer_free']] += 1
                    
            self.logger.record_reaction(current_t, reaction_index, dissociating_site)      

    def _execute_site_dimerise(self, current_t, reaction_index):
        """Reaction logic for an already bound TF to dimerise with another TF at some index in chromatin_lattice.

        Args:
            current_t (int): current time of simulation 
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        if self.state.total_dimer_weight_symmetric > 0:
            dimer_index, dimer_partner_index = self._sample_spatial_dimer(self.state.dimer_weight_matrix)
            tf1_type = self.state.chromatin_lattice[dimer_index]
            tf2_type = self.state.chromatin_lattice[dimer_partner_index]
            
            # update species counts accordingly
            if self._is_heterodimer(tf1_type, tf2_type):
                self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_bound']] += 1
                self.state.species_counts[SPECIES_MAP['sox2_monomer_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] -= 1
            else:
                self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_bound']] += 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] -= 2                
            
            self._link_dimer_sites(dimer_index, dimer_partner_index)
            self.logger.record_reaction(current_t, reaction_index, dimer_index, dimer_partner_index)

    def _execute_bulk_dimerise(self, current_t, reaction_index):
        """Updates species_counts for all non-bound TFs (i.e. only *_free tagged variables)

        Args:
            current_t (int): current time of simulation 
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        s_free = self.state.species_counts[SPECIES_MAP['sox2_monomer_free']]
        n_free = self.state.species_counts[SPECIES_MAP['nanog_monomer_free']]
        # check if there are any valid reactions 
        nn_dimers = n_free * (n_free - 1) / 2.0
        sn_dimers = s_free * n_free
        total_dimers = nn_dimers + sn_dimers
        
        if total_dimers > 0:
            if (np.random.rand() < nn_dimers / total_dimers) and n_free >= 2:
                self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_free']] += 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_free']] -= 2
            elif s_free >= 1 and n_free >= 1:
                self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_free']] += 1
                self.state.species_counts[SPECIES_MAP['sox2_monomer_free']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_free']] -= 1
        self.logger.record_reaction(current_t, reaction_index)

    def _execute_site_bulk_dimerise(self, current_t, reaction_index):
        """Updates species_counts and chromatin arrays for a reaction between a *_free species and a *_bound species.

        Args:
            current_t (int): current time of simulation 
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        s_bound = self.state.species_counts[SPECIES_MAP['sox2_monomer_bound']]
        n_bound = self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']]
        s_free = self.state.species_counts[SPECIES_MAP['sox2_monomer_free']]
        n_free = self.state.species_counts[SPECIES_MAP['nanog_monomer_free']]

        sn_dimers, ns_dimers, nn_dimers = s_bound * n_free, n_bound * s_free, n_bound * n_free
        total_dimers = sn_dimers + ns_dimers + nn_dimers

        if total_dimers > 0:
            rand_val = np.random.rand() * total_dimers
            site = -1
            if rand_val < sn_dimers and len(self.state.undimered_sox2_sites) > 0:
                site = np.random.choice(list(self.state.undimered_sox2_sites))
                self.state.set_site_state(site, is_undimered=False, dangling_tf_type=2)
                self.state.species_counts[SPECIES_MAP['sox2_monomer_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_free']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_single_bound']] += 1
            elif rand_val < sn_dimers + ns_dimers and len(self.state.undimered_nanog_sites) > 0:
                site = np.random.choice(list(self.state.undimered_nanog_sites))
                self.state.set_site_state(site, is_undimered=False, dangling_tf_type=1)
                self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['sox2_monomer_free']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_single_bound']] += 1
            elif len(self.state.undimered_nanog_sites) > 0:
                site = np.random.choice(list(self.state.undimered_nanog_sites))
                self.state.set_site_state(site, is_undimered=False, dangling_tf_type=2)
                self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_free']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_single_bound']] += 1

            self.logger.record_reaction(current_t, reaction_index, site)

    def _execute_site_dedimerise(self, current_t, reaction_index):
        """Updates species_counts and chromatin arrays when removing dimer status between two bound indices.

        Args:
            current_t (int): current time of simulation.
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        dimered_indices = list(self.state.dimered_dimer_sites)
        if len(dimered_indices) > 0:
            site1 = np.random.choice(dimered_indices)
            site2 = self.state.chromatin_all_dimers[site1]
            tf1, tf2 = self.state.chromatin_lattice[site1], self.state.chromatin_lattice[site2]
            
            self.state.set_site_state(site1, is_undimered=True, dimer_partner=-1)
            self.state.set_site_state(site2, is_undimered=True, dimer_partner=-1)
            
            if self._is_heterodimer(tf1, tf2):
                self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['sox2_monomer_bound']] += 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] += 1
            else:
                self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] += 2
                
            self.logger.record_reaction(current_t, reaction_index, site1, site2)    
    
    def _execute_bulk_dedimerise(self, current_t, reaction_index):
        """Updates species_counts and chromatin arrays when removing dimer status between two *_free variables.

        Args:
            current_t (int): current time of simulation.
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        ns_free = self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_free']]
        nn_free = self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_free']]
        total_dimers = ns_free + nn_free
        
        if total_dimers > 0:
            if (np.random.rand() < ns_free / total_dimers) and ns_free >= 1:
                self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_free']] -= 1
                self.state.species_counts[SPECIES_MAP['sox2_monomer_free']] += 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_free']] += 1
            elif nn_free >= 1:
                self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_free']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_free']] += 2
        self.logger.record_reaction(current_t, reaction_index)
  
class ModelCall():
    """Class handles building simulation by calling ModelState, Logger, and Reactions. In addition to handling Gillespie algorithm logic and propensity calculations.
    
    Attributes:
        model_param (dict): Reaction kinetic parameters (k_prod_s, k_bind_s, etc.).
        model_var (dict): Initial molecule counts for the simulation.
        model_binding_sites (int): Total number of binding sites in the chromatin lattice.
        record_interval (float): Simulation time interval to record reactions.
        sim_max_time (int): maximum simulation runtime. 
 
    """
    def __init__(self, model_param: dict, model_var: dict, model_binding_sites: int, sim_max_time, record_interval: float = 1.0):
        self.record_interval = record_interval
        self.rates = model_param
        self.max_time = sim_max_time
        
        self.state = ModelState(model_binding_sites, model_var)
        self.logger = ModelLogger(self.state)
        self.reactions =  ModelReactions(self.state, self.logger)

    
        self.stoichiometry_matrix = np.zeros((len(config.SPECIES_NAMES), len(config.REACTION_NAMES)), dtype=np.int32)
        self.stoichiometry_matrix[SPECIES_MAP["sox2_monomer_free"], 0] = 1   
        self.stoichiometry_matrix[SPECIES_MAP["nanog_monomer_free"], 1] = 1  
        self.stoichiometry_matrix[SPECIES_MAP["sox2_monomer_free"], 4] = -1  
        self.stoichiometry_matrix[SPECIES_MAP["nanog_monomer_free"], 5] = -1 
        self.stoichiometry_matrix[SPECIES_MAP["mRNA"], 8] = 1                
        self.stoichiometry_matrix[SPECIES_MAP["mRNA"], 9] = -1  

    
    def _calculate_propensities(self):
        """Calculates current reaction propensities based on the system state.
        
        Returns:
            tuple: A tuple containing:
                - propensities (np.ndarray): 1D array of reaction propensities.
                - total_propensity (float): The sum of all propensities.
        """
        c = self.state.species_counts     
        
        bulk_nn = c[1] * (c[1] - 1) / 2.0
        bulk_sn = c[0] * c[1]
        site_bulk = (c[2] * c[1]) + (c[3] * c[0]) + (c[3] * c[1])

        p_site = self.state.promoter_site
        promoter_on = (self.state.chromatin_lattice[p_site] == 1) or \
                      (self.state.chromatin_lattice[p_site] == 2 and self.state.chromatin_all_dimers[p_site] != -1 and self.state.chromatin_lattice[self.state.chromatin_all_dimers[p_site]] == 1)

        propensities = np.array([
            self.rates.get('k_s_in', 0.0),                                                                          # 0
            self.rates.get('k_n_in', 0.0),                                                                          # 1
            self.rates.get('k_bind_s', 0.0) * (c[0] + c[6] + self.state.dangling_sox2_count) * len(self.state.vacant_chromatin_sites), # 2
            self.rates.get('k_bind_n', 0.0) * (c[1] + c[7] + self.state.dangling_nanog_count) * len(self.state.vacant_chromatin_sites),# 3
            self.rates.get('k_s_out', 0.0) * c[0],                                                                  # 4
            self.rates.get('k_n_out', 0.0) * c[1],                                                                  # 5
            self.rates.get('k_unbind_s', 0.0) * (c[2] + c[4]),                                                      # 6
            self.rates.get('k_unbind_n', 0.0) * (c[3] + c[5]),                                                      # 7
            self.rates.get('k_prod_m', 0.0) if promoter_on else 0.0,                                                # 8
            self.rates.get('k_deg_m', 0.0) * c[10],                                                                 # 9
            self.rates.get('k_dimerise', 0.0) * (self.state.total_dimer_weight_symmetric / 2.0),                     # 10
            self.rates.get('k_dimerise', 0.0) * (bulk_nn + bulk_sn),                                                # 11
            0.0,                                                                                                    # 12
            self.rates.get('k_dissociate', 0.0) * (c[4] + c[5]),                                                    # 13
            self.rates.get('k_dissociate', 0.0) * (c[6] + c[7]),                                                    # 14
            self.rates.get('k_dimerise', 0.0) * site_bulk                                                           # 15
        ])
        return propensities, np.sum(propensities)

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
        current_t = 0.0

        while current_t < self.max_time:
            propensities, total_prop = self._calculate_propensities()
            if total_prop == 0: break
            
            r1, r2 = np.random.random(2)
            current_t += (1.0 / total_prop) * np.log(1.0 / r1)
            
            self.logger.record_snapshot(current_t)
            cumulative_props = np.cumsum(propensities)
            reaction_index = np.searchsorted(cumulative_props, r2 * total_prop)
            
            self.state.species_counts += self.stoichiometry_matrix[:, reaction_index]
            self.reactions.execute(current_t, reaction_index)

        return self.logger.generate_dataframes(current_t)

