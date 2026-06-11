"""Gillespie model of gene expression given spatiotemporal dynamics.

This module obtains residence time and mRNA expression from a single realisation of a
single trajectory of the chemical master equation. 

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
from .model_utils import TranscriptionFactor
from .model_state import ModelState
from .model_logger import ModelLogger
from .model_reactions import ModelReactions
from typing import List



class ModelCall:
    """Class handles building simulation by calling ModelState, Logger, and Reactions. In addition to handling Gillespie algorithm logic and propensity calculations.

    Attributes:
        model_param (dict): Reaction kinetic parameters (k_prod_s, k_bind_s, etc.).
        model_var (dict): Initial molecule counts for the simulation.
        model_binding_sites (int): Total number of binding sites in the chromatin lattice.
        record_interval (float): Simulation time interval to record reactions.
        sim_max_time (int): maximum simulation runtime.

    """

    def __init__(
        self,
        tfs: List[TranscriptionFactor],
        model_param: dict,
        model_var: dict,
        model_binding_sites: int,
        sim_max_time,
        record_interval: float = 1.0,
    ):
        self.record_interval = record_interval
        self.rates = model_param
        self.max_time = sim_max_time

        self.state = ModelState(
            tfs=tfs,
            total_binding_sites=model_binding_sites,
            initial_species_states=model_var,
        )
        
        self.observers = []
        
        self.logger = ModelLogger(self.state)
        self.add_observer(self.logger)
        
        self.reactions = ModelReactions(self.state)

        self.stoichiometry_matrix = np.zeros(
            (len(self.state.species_names), len(self.state.reaction_names)), dtype=np.int32
        )
        self.stoichiometry_matrix[self.state.species_map["sox2_monomer_free"], 0] = 1
        self.stoichiometry_matrix[self.state.species_map["nanog_monomer_free"], 1] = 1
        self.stoichiometry_matrix[self.state.species_map["sox2_monomer_free"], 4] = -1
        self.stoichiometry_matrix[self.state.species_map["nanog_monomer_free"], 5] = -1
        self.stoichiometry_matrix[self.state.species_map["mRNA"], 8] = 1
        self.stoichiometry_matrix[self.state.species_map["mRNA"], 9] = -1

        self._propensities = np.zeros(16, dtype=np.float64)

    def add_observer(self, observer):
        """_summary_ Adds an observer to the simulation's event stream."""
        self.observers.append(observer)

    def notify_snapshot(self, current_t):
        """_summary_ Emits a state snapshot event to all observers."""
        for obs in self.observers:
            obs.on_snapshot(current_t, self.state)

    def notify_reaction(self, current_t, reaction_index, primary_site=-1, secondary_site=-1):
        """_summary_ Emits a reaction fired event to all observers."""
        for obs in self.observers:
            obs.on_reaction_fired(current_t, reaction_index, primary_site, secondary_site)

    def notify_site_state_changed(self, current_t, site, old_species, new_species, reaction_name, duration, paired_site=-1, is_bound=True):
        """_summary_ Emits a site state change event for residence time logging."""
        for obs in self.observers:
            obs.on_site_state_changed(current_t, site, old_species, new_species, reaction_name, duration, paired_site, is_bound)

    def _calculate_propensities(self):
        """_summary_ Calculates current reaction propensities based on the system state.
        
        Returns:
            propensities (numpy.ndarray): updated array of reaction propensities.
            total_propensity (float): sum of all propensities.
        """
        c = self.state.species_counts

        s_free = c[self.state.species_map["sox2_monomer_free"]]
        n_free = c[self.state.species_map["nanog_monomer_free"]]
        ns_free = c[self.state.species_map["nanog_sox2_dimer_free"]]
        nn_free = c[self.state.species_map["nanog_nanog_dimer_free"]]

        # calculate bulk dimer counts
        bulk_nn = n_free * (n_free - 1) / 2.0
        bulk_sn = s_free * n_free

        len_undimered_s = self.state.undimered_count[1]
        len_undimered_n = self.state.undimered_count[2]

        # bulk dimerisation with site bound monomers
        site_bulk = (
            (len_undimered_s * n_free)
            + (len_undimered_n * s_free)
            + (len_undimered_n * n_free)
        )

        promoter_on = self.state.is_promoter_active()

        # dangling arms for dedimerisation
        total_dangling = self.state.dangling_counts[1] + self.state.dangling_counts[2]

        # update all reaction propensities
        self._propensities[0] = self.rates.get("k_s_in", 0.0)
        self._propensities[1] = self.rates.get("k_n_in", 0.0)
        self._propensities[2] = self.rates.get("k_bind_s", 0.0) * (((s_free + ns_free) * self.state.vacant_count) + self.state.total_tether_weight_s)
        self._propensities[3] = self.rates.get("k_bind_n", 0.0) * (((n_free + (nn_free * 2) + ns_free) * self.state.vacant_count) + self.state.total_tether_weight_n)
        self._propensities[4] = self.rates.get("k_s_out", 0.0) * s_free
        self._propensities[5] = self.rates.get("k_n_out", 0.0) * n_free
        self._propensities[6] = self.rates.get("k_unbind_s", 0.0) * self.state.bound_count[1]
        self._propensities[7] = self.rates.get("k_unbind_n", 0.0) * self.state.bound_count[2]
        self._propensities[8] = self.rates.get("k_prod_m", 0.0) if promoter_on else 0.0
        self._propensities[9] = self.rates.get("k_deg_m", 0.0) * c[self.state.species_map.get("mRNA", 10)]
        self._propensities[10] = self.rates.get("k_dimerise", 0.0) * (self.state.total_dimer_weight_symmetric / 2.0)
        self._propensities[11] = self.rates.get("k_dimerise", 0.0) * (bulk_nn + bulk_sn)
        self._propensities[12] = self.rates.get("k_dissociate", 0.0) * (self.state.dimered_count / 2.0)
        self._propensities[13] = self.rates.get("k_dissociate", 0.0) * (ns_free + nn_free)
        self._propensities[14] = self.rates.get("k_dimerise", 0.0) * site_bulk
        self._propensities[15] = self.rates.get("k_dissociate", 0.0) * total_dangling

        self._propensities[self._propensities < 0] = 0.0

        return self._propensities.copy(), np.sum(self._propensities)

    def run_trajectory(self):
        """Run the Gillespie simulation using the Next Reaction Method (NRM).

        1. Initialise the system and calculate all initial propensities.
        2. Generate putative reaction times (tau) for all reactions.
        3. Extract the minimum tau (mu), advance time to tau_mu, and execute reaction mu.
        4. Recalculate propensities and update tau values based on NRM rules.
        5. Repeat till max_time is reached.

        Returns:
            tuple: the generated dataframes with simulation data: self.sim_variable_states_df, self.sim_site_dwell_times_df, self.sim_reaction_history_df
        """
        current_t = 0.0

        # initial propensities
        propensities, total_prop = self._calculate_propensities()
        num_rxns = len(propensities)
        
        # generate initial putative times for all reactions
        tau = np.full(num_rxns, np.inf)
        for i in range(num_rxns):
            if propensities[i] > 0:
                tau[i] = (1.0 / propensities[i]) * np.log(1.0 / np.random.rand())

        # nrm loop
        while current_t < self.max_time:
            # find next reaction
            reaction_index = np.argmin(tau)
            min_tau = tau[reaction_index]

            if min_tau == np.inf:
                break

            current_t = min_tau

            self.notify_snapshot(current_t)
            
            # execute reaction and update state
            self.state.species_counts += self.stoichiometry_matrix[:, reaction_index]
            p_site, s_site = self.reactions.execute(current_t, reaction_index)
            self.notify_reaction(current_t, reaction_index, p_site, s_site)

            # recalculate propensities
            new_propensities, _ = self._calculate_propensities()

            # update tau according to next reaction method (nrm) rules
            for i in range(num_rxns):
                a_old = propensities[i]
                a_new = new_propensities[i]
                
                if a_new == 0.0:
                    tau[i] = np.inf
                elif i == reaction_index or a_old == 0.0:
                    tau[i] = current_t + (1.0 / a_new) * np.log(1.0 / np.random.rand())
                elif a_new != a_old:
                    tau[i] = current_t + (a_old / a_new) * (tau[i] - current_t)
                    
            propensities = new_propensities

        return self.logger.generate_dataframes(current_t)
