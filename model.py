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
        self.chromatin_site_bind_times = np.full(self.total_sites, -1.0)
        self.vacant_chromatin_sites = set(range(self.total_sites))
        
        self.chromatin_partner_state = np.full(self.total_sites, -1, dtype=np.int32)
        
        self.bound_sox2_sites = set()
        self.bound_nanog_sites = set()
        self.dimered_dimer_sites = set()
        self.undimered_sox2_sites = set()
        self.undimered_nanog_sites = set()
        self.monovalent_sox2_dimer_count = 0
        self.dangling_nanog_count = 0
        
        # spatial distance kernel
        chromatin_lattice_indicies = np.arange(self.total_sites)
        chromatin_potential_dimer_partners = np.abs(chromatin_lattice_indicies[:, None] - chromatin_lattice_indicies[None, :]) 
        self.dist_weighted_dimer_partners = np.exp(-chromatin_potential_dimer_partners / 1)
        np.fill_diagonal(self.dist_weighted_dimer_partners, 0.0)
        
        
        self.total_dimer_weight_symmetric = 0.0
        self.total_tether_weight_s = 0.0  # Weight of dangling SOX2s finding a site
        self.total_tether_weight_n = 0.0  # Weight of dangling NANOGs finding a site

        # increment weights
        self.dimer_weight_matrix = np.zeros_like(self.dist_weighted_dimer_partners)
        self.bivalent_transition_matrix = np.zeros_like(self.dist_weighted_dimer_partners)
        self.total_dimer_weight_symmetric = 0.0
        self.total_tether_weight = 0.0
        
    def get_species_label(self, site):
        if site == -1 or self.chromatin_site_is_vacant[site]:
            return "EMPTY"
        
        tf_type = self.chromatin_lattice[site]
        base_str = "SOX2b" if tf_type == 1 else "NANOGb"
        partner = self.chromatin_partner_state[site]
        
        if partner == -1: return base_str
        elif partner == -2: return f"{base_str}:SOX2f"
        elif partner == -3: return f"{base_str}:NANOGf"
        elif partner >= 0:
            partner_type = self.chromatin_lattice[partner]
            if tf_type == 1 and partner_type == 2: return "SOX2b:NANOGb"
            elif tf_type == 2 and partner_type == 1: return "SOX2b:NANOGb" # Maintain standard orientation
            else: return f"{base_str}:{'SOX2b' if partner_type == 1 else 'NANOGb'}"
        return "UNKNOWN"

    def update_site_weights(self, site):
        """Update spatial weight matrices to decide dimerisation partners."""
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

        new_tether_row = np.zeros(self.total_sites)
        
        if self.chromatin_partner_state[site] in (-2, -3):
            vacant_mask = self.chromatin_site_is_vacant
            new_tether_row[vacant_mask] = self.dist_weighted_dimer_partners[site, vacant_mask]
            new_tether_row[site] = 0.0
        
        old_tether_col_sum = np.sum(self.bivalent_transition_matrix[:, site])
        new_tether_col = np.zeros(self.total_sites)
        
        if self.chromatin_site_is_vacant[site]:
            tether_mask = (self.chromatin_partner_state < -1) 
            new_tether_col[tether_mask] = self.dist_weighted_dimer_partners[tether_mask, site]
            new_tether_col[site] = 0.0
            
        self.bivalent_transition_matrix[:, site] = new_tether_col
        self.total_tether_weight += (np.sum(new_tether_col) - old_tether_col_sum)
        self.total_tether_weight_s = np.sum(self.bivalent_transition_matrix[self.chromatin_partner_state == -2, :])
        self.total_tether_weight_n = np.sum(self.bivalent_transition_matrix[self.chromatin_partner_state == -3, :])
        
    def set_site_state(self, site, is_vacant=None, tf_type=None, is_undimered=None, partner_state=None):
        """Update status of a site on the chromatin array when a binding/unbinding/dimerisation reaction occurs. 

        Args:
            site (int): index referring to position in chromatin array
            is_vacant (bool, optional): Update site occupancy status. Defaults to None.
            tf_type (int, optional): Form of TF. i.e. SOX2, NANOG. Defaults to None.
            is_undimered (bool, optional): Track dimer status. Defaults to None.
            dangling_tf_type (int, optional): For a dimerised molecule, this refers to the TF that is not bound to any site. i.e. it is bound to this TF but not bound to anything else. Defaults to None.
            dimer_partner (int, optional): index referring to the dimerised TF partner on the chromatin array. Defaults to None.
        """
        if tf_type is not None:
            old_tf = self.chromatin_lattice[site]
            if old_tf == 1: self.bound_sox2_sites.discard(site)
            elif old_tf == 2: self.bound_nanog_sites.discard(site)
            self.chromatin_lattice[site] = tf_type
            if tf_type == 1: self.bound_sox2_sites.add(site)
            elif tf_type == 2: self.bound_nanog_sites.add(site)

        if is_vacant is not None:
            self.chromatin_site_is_vacant[site] = is_vacant
            if is_vacant: self.vacant_chromatin_sites.add(site)
            else: self.vacant_chromatin_sites.discard(site)

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
                
        if partner_state is not None:
            old_state = self.chromatin_partner_state[site]
            
            if old_state == -2: self.monovalent_sox2_dimer_count -= 1
            elif old_state == -3: self.dangling_nanog_count -= 1
            elif old_state >= 0: self.dimered_dimer_sites.discard(site)
            
            self.chromatin_partner_state[site] = partner_state
            
            if partner_state == -2: self.monovalent_sox2_dimer_count += 1
            elif partner_state == -3: self.dangling_nanog_count += 1
            elif partner_state >= 0: self.dimered_dimer_sites.add(site)
                
        self.update_site_weights(site)

class ModelLogger():
    """_summary_ Methods for recording reactions, and outputting DFs containing time course data for sites, residence times, and a list of all reactions that occur.
    
    Attributes:
        state (class): inherits ModelState class methods and attributes to record all reaction information. 
    """
    def __init__(self, state: ModelState):
        self.state = state
        self.times, self.bulk_states, self.spatial_states = [], [], []
        self.partner_states = []
        self.reaction_history, self.residence_time_states = [], []
        
    def record_reaction(self, current_t, reaction_index, primary_site=-1, secondary_site=-1):
        reaction_name = config.REACTION_NAMES[reaction_index] if reaction_index < len(config.REACTION_NAMES) else f"RXN_{reaction_index}"
        self.reaction_history.append((current_t, reaction_name, primary_site, secondary_site))

    def record_snapshot(self, current_t):
        self.times.append(current_t)
        self.bulk_states.append(self.state.species_counts.copy())
        self.spatial_states.append(self.state.chromatin_lattice.copy())
        self.partner_states.append(self.state.chromatin_partner_state.copy())

    def generate_dataframes(self, final_t):
        """Outputs three dataframes containing simulation data.
        
        Return:
            polars dataframes df_states, df_dwell df_rxns.
        """
        for i in range(self.state.total_sites):
            if not self.state.chromatin_site_is_vacant[i] and self.state.chromatin_site_bind_times[i] != -1.0:
                
                partner_state = self.state.chromatin_partner_state[i]
                if partner_state >= 0:
                    if i > partner_state: continue # Skip duplicates for bivalent dimers
                    paired_site = partner_state
                else:
                    paired_site = -1
                
                species_label = self.state.get_species_label(i)
                
                self.residence_time_states.append([
                    final_t - self.state.chromatin_site_bind_times[i], 
                    i, paired_site, species_label, "STILL_BOUND", "END_OF_SIMULATION", True
                ])        
                df_states = pl.DataFrame({
                    "time": self.times,
                    **{name: [s[idx] for s in self.bulk_states] for name, idx in SPECIES_MAP.items()}
                })
           
        df_dwell = pl.DataFrame(
            self.residence_time_states,
            schema={
                "event_duration": pl.Float64, 
                "dwell_site": pl.Int64, 
                "paired_site": pl.Int64,
                "old_species": pl.String, 
                "new_species": pl.String, 
                "reaction_name": pl.String,
                "is_bound": pl.Boolean
            },
            orient="row"
        )
        
        df_rxns = pl.DataFrame(
            self.reaction_history, 
            schema={"time": pl.Float64, "reaction_type": pl.String, "primary_site": pl.Int64, "secondary_site": pl.Int64}, 
            orient="row"
        )        
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
            2: self._handle_tf_binding, 3: self._handle_tf_binding, 
            6: self._handle_tf_unbinding, 7: self._handle_tf_unbinding,
            10: self._execute_site_dimerise, 11: self._execute_bulk_dimerise,
            12: self._execute_site_dedimerise, 13: self._execute_bulk_dedimerise, 14: self._execute_site_bulk_dimerise,
            15: self._execute_single_bound_dedimerise 
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
    
    def _update_site_times(self, event_end_time, site, old_species_label, new_species_label, reaction, paired_site=-1, is_bound=True, time_reset_value=-1.0):
        event_start_time = self.state.chromatin_site_bind_times[site]
        duration = (event_end_time - event_start_time) if event_start_time != -1.0 else (event_end_time - 0)
        
        reaction_name = config.REACTION_NAMES[reaction] if reaction < len(config.REACTION_NAMES) else f"RXN_{reaction}"
        
        self.logger.residence_time_states.append([
            duration, site, paired_site, old_species_label, new_species_label, reaction_name, is_bound
        ])
        self.state.chromatin_site_bind_times[site] = time_reset_value
    
    def _handle_tf_binding(self, current_t, reaction_index):
        """Unified binding logic handling both Bulk (Intermolecular) and Tethered (Intramolecular) binding."""
        vacant_sites = list(self.state.vacant_chromatin_sites)
        num_vacant = len(vacant_sites)
        c = self.state.species_counts

        if reaction_index == 2:  # SOX2-mediated binding
            tf_type = 1
            bulk_s = c[SPECIES_MAP["sox2_monomer_free"]] * num_vacant
            bulk_ns = c[SPECIES_MAP["nanog_sox2_dimer_free"]] * num_vacant
            tether_w = self.state.total_tether_weight_s
            
            total_pool = bulk_s + bulk_ns + tether_w
            if total_pool <= 0: return
            
            rand_val = np.random.rand() * total_pool
            
            # --- BULK BINDING ---
            if rand_val < bulk_s + bulk_ns:  
                primary_site = np.random.choice(vacant_sites)
                if rand_val < bulk_s:
                    self.state.set_site_state(primary_site, is_vacant=False, tf_type=tf_type, is_undimered=True, partner_state=-1)
                    c[SPECIES_MAP["sox2_monomer_free"]] -= 1
                    c[SPECIES_MAP["sox2_monomer_bound"]] += 1
                    species_label = "SOX2b"
                else:
                    self.state.set_site_state(primary_site, is_vacant=False, tf_type=tf_type, partner_state=-3) # NANOG dangles
                    c[SPECIES_MAP["nanog_sox2_dimer_free"]] -= 1
                    c[SPECIES_MAP["nanog_sox2_dimer_single_bound"]] += 1
                    species_label = "SOX2b:NANOGf"
                    
                self.state.chromatin_site_bind_times[primary_site] = current_t
                self._update_site_times(current_t, primary_site, "EMPTY", species_label, reaction_index, time_reset_value=current_t, is_bound=True)
                self.logger.record_reaction(current_t, reaction_index, primary_site)
                
            # --- TETHERED BINDING ---
            else:  
                self._execute_tethered_binding_logic(current_t, reaction_index, dangling_type=1, dangling_state_marker=-2)
                
        else:  # NANOG-mediated binding (reaction_index == 3)
            tf_type = 2
            bulk_n = c[SPECIES_MAP["nanog_monomer_free"]] * num_vacant
            bulk_nn = (2 * c[SPECIES_MAP["nanog_nanog_dimer_free"]]) * num_vacant
            bulk_ns = c[SPECIES_MAP["nanog_sox2_dimer_free"]] * num_vacant
            tether_w = self.state.total_tether_weight_n
            
            total_pool = bulk_n + bulk_nn + bulk_ns + tether_w
            if total_pool <= 0: return
            
            rand_val = np.random.rand() * total_pool
            
            # --- BULK BINDING ---
            if rand_val < bulk_n + bulk_nn + bulk_ns: 
                primary_site = np.random.choice(vacant_sites)
                if rand_val < bulk_n:
                    self.state.set_site_state(primary_site, is_vacant=False, tf_type=tf_type, is_undimered=True, partner_state=-1)
                    c[SPECIES_MAP["nanog_monomer_free"]] -= 1
                    c[SPECIES_MAP["nanog_monomer_bound"]] += 1
                    species_label = "NANOGb"
                elif rand_val < bulk_n + bulk_nn:
                    self.state.set_site_state(primary_site, is_vacant=False, tf_type=tf_type, partner_state=-3) # NANOG dangles
                    c[SPECIES_MAP["nanog_nanog_dimer_free"]] -= 1
                    c[SPECIES_MAP["nanog_nanog_dimer_single_bound"]] += 1
                    species_label = "NANOGb:NANOGf"
                else:
                    self.state.set_site_state(primary_site, is_vacant=False, tf_type=tf_type, partner_state=-2) # SOX2 dangles
                    c[SPECIES_MAP["nanog_sox2_dimer_free"]] -= 1
                    c[SPECIES_MAP["nanog_sox2_dimer_single_bound"]] += 1
                    species_label = "NANOGb:SOX2f"

                self.state.chromatin_site_bind_times[primary_site] = current_t
                self._update_site_times(current_t, primary_site, "EMPTY", species_label, reaction_index, time_reset_value=current_t, is_bound=True)
                self.logger.record_reaction(current_t, reaction_index, primary_site)
                
            # --- TETHERED BINDING ---
            else: 
                self._execute_tethered_binding_logic(current_t, reaction_index, dangling_type=2, dangling_state_marker=-3)            
    def _handle_tf_unbinding(self, current_t, reaction_index):
        """Reaction logic for a TF unbinding from a site on the chromatin array.

        Args:
            current_t (int): current time of simulation 
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        tf_type = 1 if reaction_index == 6 else 2 
        bound_indices = list(self.state.bound_sox2_sites) if tf_type == 1 else list(self.state.bound_nanog_sites)
        if len(bound_indices) == 0: return
        
        dissociating_site = np.random.choice(bound_indices)
        partner_state = self.state.chromatin_partner_state[dissociating_site]
        
        old_label = self.state.get_species_label(dissociating_site)
        self.state.set_site_state(dissociating_site, is_vacant=True, tf_type=0, is_undimered=False, partner_state=-1)

        # update the partner/bulk state
        if partner_state >= 0:
            # partner was fully bound, is now dangling
            self.state.set_site_state(partner_state, partner_state=(-2 if tf_type == 1 else -3))
            partner_tf_type = self.state.chromatin_lattice[partner_state]
            if self._is_heterodimer(tf_type, partner_tf_type):
                self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_single_bound']] += 1
            else:
                self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_single_bound']] += 1
                
        elif partner_state in (-2, -3):
            # it was single-bound, dangling arm is now completely free
            if self._is_heterodimer(tf_type, 1 if partner_state == -2 else 2):
                self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_single_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_free']] += 1
            else:
                self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_single_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_free']] += 1
        else: # was a Monomer
            if tf_type == 1:
                self.state.species_counts[SPECIES_MAP['sox2_monomer_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['sox2_monomer_free']] += 1
            else:
                self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_free']] += 1
        
        self._update_site_times(current_t, dissociating_site, old_label, "EMPTY", reaction_index, paired_site=-1, time_reset_value=current_t, is_bound=False)
        self.logger.record_reaction(current_t, reaction_index, dissociating_site)
    
    def _execute_tethered_binding_logic(self, current_t, reaction_index, dangling_type, dangling_state_marker):
        """Helper to execute intramolecular binding when selected via Gillespie."""
        # 1. Mask the matrix so we ONLY sample from the correct dangling pool (SOX2 vs NANOG)
        masked_matrix = np.copy(self.state.bivalent_transition_matrix)
        masked_matrix[self.state.chromatin_partner_state != dangling_state_marker, :] = 0
        
        tethered_site, target_vacant_site = self._sample_spatial_dimer(masked_matrix, is_tether=True)
        bound_tf_type = self.state.chromatin_lattice[tethered_site]
        old_tether_label = self.state.get_species_label(tethered_site)
        
        # 2. Update states
        self.state.set_site_state(target_vacant_site, is_vacant=False, tf_type=dangling_type)
        self.state.set_site_state(tethered_site, partner_state=target_vacant_site)
        self.state.set_site_state(target_vacant_site, is_undimered=False, partner_state=tethered_site)
        new_label = self.state.get_species_label(tethered_site)
        
        # 3. Update species pool
        if self._is_heterodimer(bound_tf_type, dangling_type):
            self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_single_bound']] -= 1
            self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_bound']] += 1
        else:
            self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_single_bound']] -= 1
            self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_bound']] += 1

        self._update_site_times(current_t, tethered_site, old_tether_label, new_label, reaction_index, paired_site=target_vacant_site, time_reset_value=current_t, is_bound=True)
        self._update_site_times(current_t, target_vacant_site, "EMPTY", new_label, reaction_index, paired_site=tethered_site, time_reset_value=current_t, is_bound=True)
        self.logger.record_reaction(current_t, reaction_index, tethered_site, target_vacant_site)
    
    def _execute_site_dimerise(self, current_t, reaction_index):
        """Reaction logic for an already bound TF to dimerise with another TF at some index in chromatin_lattice.

        Args:
            current_t (int): current time of simulation 
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        if self.state.total_dimer_weight_symmetric > 0:
            dimer_index, dimer_partner_index = self._sample_spatial_dimer(self.state.dimer_weight_matrix)
        
            old_label_index = self.state.get_species_label(dimer_index)
            old_label_partner = self.state.get_species_label(dimer_partner_index)

            self.state.set_site_state(dimer_index, is_undimered=False, partner_state=dimer_partner_index)
            self.state.set_site_state(dimer_partner_index, is_undimered=False, partner_state=dimer_index)

            new_label_index = self.state.get_species_label(dimer_index)
            new_label_partner = self.state.get_species_label(dimer_partner_index)        
            
            dimer_index_tf_type = self.state.chromatin_lattice[dimer_index]
            dimer_partner_index_tf_type = self.state.chromatin_lattice[dimer_partner_index]    
            
            
            # update species counts accordingly
            if self._is_heterodimer(dimer_index_tf_type, dimer_partner_index_tf_type):
                self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_bound']] += 1
                self.state.species_counts[SPECIES_MAP['sox2_monomer_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] -= 1
            else:
                self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_bound']] += 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] -= 2                

            self._update_site_times(current_t, dimer_index, old_label_index, new_label_index, reaction_index, paired_site=dimer_partner_index, time_reset_value=current_t, is_bound=True)
            self._update_site_times(current_t, dimer_partner_index, old_label_partner, new_label_partner, reaction_index, paired_site=dimer_index, time_reset_value=current_t, is_bound=True)
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
        s_free = self.state.species_counts[SPECIES_MAP['sox2_monomer_free']]
        n_free = self.state.species_counts[SPECIES_MAP['nanog_monomer_free']]

        sn_dimers = len(self.state.undimered_sox2_sites) * n_free
        ns_dimers = len(self.state.undimered_nanog_sites) * s_free
        nn_dimers = len(self.state.undimered_nanog_sites) * n_free
        
        total_dimers = sn_dimers + ns_dimers + nn_dimers
        if total_dimers <= 0: return
        
        rand_val = np.random.rand() * total_dimers
        site = -1    

        if rand_val < sn_dimers and len(self.state.undimered_sox2_sites) > 0:
            site = np.random.choice(list(self.state.undimered_sox2_sites))
            old_label = self.state.get_species_label(site)
            self.state.set_site_state(site, is_undimered=False, partner_state=-3) # SOX2 binds dangling NANOG
            new_label = self.state.get_species_label(site)
            self.state.species_counts[SPECIES_MAP['sox2_monomer_bound']] -= 1
            self.state.species_counts[SPECIES_MAP['nanog_monomer_free']] -= 1
            self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_single_bound']] += 1
            
        elif rand_val < sn_dimers + ns_dimers and len(self.state.undimered_nanog_sites) > 0:
            site = np.random.choice(list(self.state.undimered_nanog_sites))
            old_label = self.state.get_species_label(site)
            self.state.set_site_state(site, is_undimered=False, partner_state=-2) # NANOG binds dangling SOX2
            new_label = self.state.get_species_label(site)
            self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] -= 1
            self.state.species_counts[SPECIES_MAP['sox2_monomer_free']] -= 1
            self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_single_bound']] += 1
            
        elif len(self.state.undimered_nanog_sites) > 0:
            site = np.random.choice(list(self.state.undimered_nanog_sites))
            old_label = self.state.get_species_label(site)
            self.state.set_site_state(site, is_undimered=False, partner_state=-3) # NANOG binds dangling NANOG
            new_label = self.state.get_species_label(site)
            self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] -= 1
            self.state.species_counts[SPECIES_MAP['nanog_monomer_free']] -= 1
            self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_single_bound']] += 1

        if site != -1:
            self._update_site_times(current_t, site, old_label, new_label, reaction_index, time_reset_value=current_t, is_bound=True)
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
            site2 = self.state.chromatin_partner_state[site1]
            dimer_index_tf_type = self.state.chromatin_lattice[site1]
            dimer_partner_index_tf_type = self.state.chromatin_lattice[site2]
            
            # Read OLD label before breaking them apart
            old_label = self.state.get_species_label(site1)
            
            # Break them apart
            self.state.set_site_state(site1, is_undimered=True, partner_state=-1)
            self.state.set_site_state(site2, is_undimered=True, partner_state=-1)
            
            # Read NEW labels
            new_sp1 = self.state.get_species_label(site1)
            new_sp2 = self.state.get_species_label(site2)
                                
            if self._is_heterodimer(dimer_index_tf_type, dimer_partner_index_tf_type):
                self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['sox2_monomer_bound']] += 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] += 1
            else:
                self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_bound']] -= 1
                self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] += 2

            self._update_site_times(current_t, site1, old_label, new_sp1, reaction_index, paired_site=site2, time_reset_value=current_t, is_bound=True)
            self._update_site_times(current_t, site2, old_label, new_sp2, reaction_index, paired_site=site2, time_reset_value=current_t, is_bound=True)
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
             
    def _execute_single_bound_dedimerise(self, current_t, reaction_index):
        """Reaction logic for a dangling dimer snapping apart, dropping a monomer back to bulk."""
        dangling_sites = [s for s in range(self.state.total_sites) if self.state.chromatin_partner_state[s] in (-2, -3)]
        if not dangling_sites: return
        
        site = np.random.choice(dangling_sites)
        bound_tf_type = self.state.chromatin_lattice[site]
        partner_state = self.state.chromatin_partner_state[site]
        dangling_tf_type = 1 if partner_state == -2 else 2
        
        old_label = self.state.get_species_label(site)
        self.state.set_site_state(site, is_undimered=True, partner_state=-1)
        new_label = self.state.get_species_label(site)
        
        if dangling_tf_type == 1:
            self.state.species_counts[SPECIES_MAP['sox2_monomer_free']] += 1
        else:
            self.state.species_counts[SPECIES_MAP['nanog_monomer_free']] += 1
            
        if self._is_heterodimer(bound_tf_type, dangling_tf_type):
            self.state.species_counts[SPECIES_MAP['nanog_sox2_dimer_single_bound']] -= 1
            if bound_tf_type == 1:
                self.state.species_counts[SPECIES_MAP['sox2_monomer_bound']] += 1
            else:
                self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] += 1
        else:
            self.state.species_counts[SPECIES_MAP['nanog_nanog_dimer_single_bound']] -= 1
            self.state.species_counts[SPECIES_MAP['nanog_monomer_bound']] += 1
            
        self._update_site_times(current_t, site, old_label, new_label, reaction_index, paired_site=-1, time_reset_value=current_t, is_bound=True)
        self.logger.record_reaction(current_t, reaction_index, site)  
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
        
        s_free = c[SPECIES_MAP['sox2_monomer_free']]
        n_free = c[SPECIES_MAP['nanog_monomer_free']]
        ns_free = c[SPECIES_MAP['nanog_sox2_dimer_free']]
        nn_free = c[SPECIES_MAP['nanog_nanog_dimer_free']]
        
        bulk_nn = n_free * (n_free - 1) / 2.0
        bulk_sn = s_free * n_free

        site_bulk = (len(self.state.undimered_sox2_sites) * n_free) + \
                    (len(self.state.undimered_nanog_sites) * s_free) + \
                    (len(self.state.undimered_nanog_sites) * n_free)

        p_site = self.state.promoter_site
        partner_site = self.state.chromatin_partner_state[p_site]
        
        promoter_on = (self.state.chromatin_lattice[p_site] == 1) or \
                      (self.state.chromatin_lattice[p_site] == 2 and partner_site >= 0 and self.state.chromatin_lattice[partner_site] == 1)
        
        propensities = np.array([
            self.rates.get('k_s_in', 0.0),                                                                  # 0: S Prod
            self.rates.get('k_n_in', 0.0),                                                                  # 1: N Prod
            self.rates.get('k_bind_s', 0.0) * (((s_free + ns_free) * len(self.state.vacant_chromatin_sites)) + self.state.total_tether_weight_s),  # 2: Bulk S/NS Binding
            self.rates.get('k_bind_n', 0.0) * (((n_free + (nn_free * 2) + ns_free) * len(self.state.vacant_chromatin_sites)) + self.state.total_tether_weight_n),  # 3: Bulk N/NN Binding
            self.rates.get('k_s_out', 0.0) * s_free,                                                        # 4: S Deg
            self.rates.get('k_n_out', 0.0) * n_free,                                                        # 5: N Deg
            self.rates.get('k_unbind_s', 0.0) * len(self.state.bound_sox2_sites),                           # 6: S Unbind
            self.rates.get('k_unbind_n', 0.0) * len(self.state.bound_nanog_sites),                          # 7: N Unbind
            self.rates.get('k_prod_m', 0.0) if promoter_on else 0.0,                                        # 8: mRNA Prod
            self.rates.get('k_deg_m', 0.0) * c[SPECIES_MAP.get('mRNA', 10)],                                # 9: mRNA Deg
            self.rates.get('k_dimerise', 0.0) * (self.state.total_dimer_weight_symmetric / 2.0),            # 10: Site-Site Dimerise
            self.rates.get('k_dimerise', 0.0) * (bulk_nn + bulk_sn),                                        # 11: Bulk-Bulk Dimerise
            self.rates.get('k_dissociate', 0.0) * (len(self.state.dimered_dimer_sites) / 2.0),              # 12: Site-Site Dedimerise
            self.rates.get('k_dissociate', 0.0) * (ns_free + nn_free),                                      # 13: Bulk-Bulk Dedimerise
            self.rates.get('k_dimerise', 0.0) * site_bulk,                                                  # 14: Site-Bulk Dimerise
            self.rates.get('k_dissociate', 0.0) * (self.state.monovalent_sox2_dimer_count + self.state.dangling_nanog_count) # 15: Single-Bound Dedimerise
        ])
        
        propensities[propensities < 0] = 0.0
        
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

