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
        self.logger = ModelLogger(self.state)
        self.reactions = ModelReactions(self.state, self.logger)

        self.stoichiometry_matrix = np.zeros(
            (len(self.state.species_names), len(self.state.reaction_names)), dtype=np.int32
        )
        self.stoichiometry_matrix[self.state.species_map["sox2_monomer_free"], 0] = 1
        self.stoichiometry_matrix[self.state.species_map["nanog_monomer_free"], 1] = 1
        self.stoichiometry_matrix[self.state.species_map["sox2_monomer_free"], 4] = -1
        self.stoichiometry_matrix[self.state.species_map["nanog_monomer_free"], 5] = -1
        self.stoichiometry_matrix[self.state.species_map["mRNA"], 8] = 1
        self.stoichiometry_matrix[self.state.species_map["mRNA"], 9] = -1

    def _calculate_propensities(self):
        """Calculates current reaction propensities based on the system state."""
        c = self.state.species_counts

        s_free = c[self.state.species_map["sox2_monomer_free"]]
        n_free = c[self.state.species_map["nanog_monomer_free"]]
        ns_free = c[self.state.species_map["nanog_sox2_dimer_free"]]
        nn_free = c[self.state.species_map["nanog_nanog_dimer_free"]]

        bulk_nn = n_free * (n_free - 1) / 2.0
        bulk_sn = s_free * n_free

        len_undimered_s = len(self.state.undimered_sites.get(1, set()))
        len_undimered_n = len(self.state.undimered_sites.get(2, set()))

        site_bulk = (
            (len_undimered_s * n_free)
            + (len_undimered_n * s_free)
            + (len_undimered_n * n_free)
        )

        promoter_on = self.state.is_promoter_active()

        propensities = np.array(
            [
                self.rates.get("k_s_in", 0.0),
                self.rates.get("k_n_in", 0.0),
                self.rates.get("k_bind_s", 0.0)
                * (
                    ((s_free + ns_free) * len(self.state.vacant_chromatin_sites))
                    + self.state.total_tether_weight_s
                ),
                self.rates.get("k_bind_n", 0.0)
                * (
                    (
                        (n_free + (nn_free * 2) + ns_free)
                        * len(self.state.vacant_chromatin_sites)
                    )
                    + self.state.total_tether_weight_n
                ),
                self.rates.get("k_s_out", 0.0) * s_free,
                self.rates.get("k_n_out", 0.0) * n_free,
                self.rates.get("k_unbind_s", 0.0) * len(self.state.bound_sites.get(1, set())),
                self.rates.get("k_unbind_n", 0.0) * len(self.state.bound_sites.get(2, set())),
                self.rates.get("k_prod_m", 0.0) if promoter_on else 0.0,
                self.rates.get("k_deg_m", 0.0) * c[self.state.species_map.get("mRNA", 10)],
                self.rates.get("k_dimerise", 0.0) * (self.state.total_dimer_weight_symmetric / 2.0),
                self.rates.get("k_dimerise", 0.0) * (bulk_nn + bulk_sn),
                self.rates.get("k_dissociate", 0.0) * (len(self.state.dimered_dimer_sites) / 2.0),
                self.rates.get("k_dissociate", 0.0) * (ns_free + nn_free),
                self.rates.get("k_dimerise", 0.0) * site_bulk,
                self.rates.get("k_dissociate", 0.0) * (sum(self.state.dangling_counts.values())),
            ]
        )

        propensities[propensities < 0] = 0.0

        return propensities, np.sum(propensities)

    def run_trajectory(self):
        """Run the Gillespie simulation.

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
            if total_prop == 0:
                break

            r1, r2 = np.random.random(2)
            current_t += (1.0 / total_prop) * np.log(1.0 / r1)

            self.logger.record_snapshot(current_t)
            cumulative_props = np.cumsum(propensities)
            reaction_index = np.searchsorted(cumulative_props, r2 * total_prop)

            self.state.species_counts += self.stoichiometry_matrix[:, reaction_index]
            self.reactions.execute(current_t, reaction_index)

        return self.logger.generate_dataframes(current_t)
