
import numpy as np
from . import constants
from .model_state import ModelState
from .model_logger import ModelLogger
from numba import njit

@njit
def fast_sample_spatial_dimer(weight_matrix, is_tether=False):
    if not is_tether:
        weight_matrix = np.triu(weight_matrix, 1)
        
    row_sums = np.zeros(weight_matrix.shape[0])
    for i in range(weight_matrix.shape[0]):
        row_sums[i] = np.sum(weight_matrix[i, :])
        
    total_sum = np.sum(row_sums)
    if total_sum <= 0:
        return 0, 0
        
    r1 = np.random.rand() * total_sum
    cum_rows = 0.0
    row_idx = 0
    for i in range(len(row_sums)):
        cum_rows += row_sums[i]
        if cum_rows >= r1:
            row_idx = i
            break
            
    col_weights = weight_matrix[row_idx, :]
    r2 = np.random.rand() * row_sums[row_idx]
    cum_cols = 0.0
    col_idx = 0
    for i in range(len(col_weights)):
        cum_cols += col_weights[i]
        if cum_cols >= r2:
            col_idx = i
            break
            
    return row_idx, col_idx

class ModelReactions:
    """Contains reaction logic methods to update ModelState attributes as dictated by the class ModelCall.

    Attributes:
        state (class): inherits simulation state information and important state update methods from ModelState.
        logger (class): inherits ModelLogger methods to track reactions, dwelltimes, etc.
    """

    def __init__(self, state: ModelState, logger: ModelLogger):
        self.state = state
        self.logger = logger
        self.select_reaction_map = {
            2: self._handle_tf_binding,
            3: self._handle_tf_binding,
            6: self._handle_tf_unbinding,
            7: self._handle_tf_unbinding,
            10: self._execute_site_dimerise,
            11: self._execute_bulk_dimerise,
            12: self._execute_site_dedimerise,
            13: self._execute_bulk_dedimerise,
            14: self._execute_site_bulk_dimerise,
            15: self._execute_single_bound_dedimerise,
        }

    def execute(self, current_t, reaction_index):
        # Default to _log_only if reaction_index not explicitly mapped
        handler = self.select_reaction_map.get(reaction_index, self._log_only)
        handler(current_t, reaction_index)

    def _log_only(self, current_t, reaction_index):
        self.logger.record_reaction(current_t, reaction_index)



    def _link_dimer_sites(self, site1, site2):
        """Updates two indexes on chromatin_lattice by updating chromatin_all_undimered_monomers.

        Args:
            site2 (int): index referring to last half of dimer
        """
        tf1 = self.state.get_base_tf(self.state.chromatin_state[site1])
        tf2 = self.state.get_base_tf(self.state.chromatin_state[site2])
        self.state.set_site_state(site1, constants.SiteState.SOX2b_NANOGb if tf1 == 1 else (constants.SiteState.NANOGb_SOX2b if tf2 == 1 else constants.SiteState.NANOGb_NANOGb), site2)
        self.state.set_site_state(site2, constants.SiteState.SOX2b_NANOGb if tf2 == 1 else (constants.SiteState.NANOGb_SOX2b if tf1 == 1 else constants.SiteState.NANOGb_NANOGb), site1)
    def _is_heterodimer(self, tf1_type, tf2_type):
        """Determines if two dimerised TFs are either hetero (S2-N) (N-S2)/homo (N-N).

        Args:
            tf1_type (int): type of TF
            tf2_type (int): type of TF

        Returns:
            bool: True if heterodimer. False otherwise.
        """
        return (tf1_type == 1 and tf2_type == 2) or (tf1_type == 2 and tf2_type == 1)

    def _update_site_times(
        self,
        current_sim_time,
        site,
        old_species_label,
        new_species_label,
        reaction,
        paired_site=-1,
        is_bound=True,
        time_reset_value=-1.0,
    ):
        event_start_time = self.state.chromatin_site_bind_times[site]
        duration = (
            (current_sim_time - event_start_time)
            if event_start_time != -1.0
            else (current_sim_time - 0)
        )

        reaction_name = (
            self.state.reaction_names[reaction]
            if reaction < len(self.state.reaction_names)
            else f"RXN_{reaction}"
        )

        self.logger.residence_time_states.append(
            [
                duration,
                current_sim_time,
                site,
                paired_site,
                old_species_label,
                new_species_label,
                reaction_name,
                is_bound,
            ]
        )
        self.state.chromatin_site_bind_times[site] = time_reset_value

    def _handle_tf_binding(self, current_t, reaction_index):
        """Binding logic handling both Bulk (Intermolecular) and Tethered (Intramolecular) binding."""
        vacant_sites = self.state.vacant_sites[:self.state.vacant_count]
        num_vacant = len(vacant_sites)
        c = self.state.species_counts

        is_sox2 = reaction_index == 2
        tf_type = 1 if is_sox2 else 2

        bulk_monomer = c[self.state.species_map["sox2_monomer_free" if is_sox2 else "nanog_monomer_free"]] * num_vacant
        bulk_heterodimer = c[self.state.species_map["nanog_sox2_dimer_free"]] * num_vacant
        bulk_homodimer = 0 if is_sox2 else (2 * c[self.state.species_map["nanog_nanog_dimer_free"]]) * num_vacant
        tether_w = self.state.total_tether_weight_s if is_sox2 else self.state.total_tether_weight_n

        total_bulk = bulk_monomer + bulk_homodimer + bulk_heterodimer
        total_pool = total_bulk + tether_w

        if total_pool <= 0:
            return

        rand_val = np.random.rand() * total_pool

        if rand_val < total_bulk:
            primary_site = np.random.choice(vacant_sites)
            
            if rand_val < bulk_monomer:
                self.state.set_site_state(primary_site, constants.SiteState.SOX2b if is_sox2 else constants.SiteState.NANOGb)
                c[self.state.species_map["sox2_monomer_free" if is_sox2 else "nanog_monomer_free"]] -= 1
                c[self.state.species_map["sox2_monomer_bound" if is_sox2 else "nanog_monomer_bound"]] += 1
                species_label = "SOX2b" if is_sox2 else "NANOGb"
            elif not is_sox2 and rand_val < bulk_monomer + bulk_homodimer:
                self.state.set_site_state(primary_site, constants.SiteState.NANOGb_NANOGf)
                c[self.state.species_map["nanog_nanog_dimer_free"]] -= 1
                c[self.state.species_map["nanog_nanog_dimer_single_bound"]] += 1
                species_label = "NANOGb:NANOGf"
            else:
                self.state.set_site_state(primary_site, constants.SiteState.SOX2b_NANOGf if is_sox2 else constants.SiteState.NANOGb_SOX2f)
                c[self.state.species_map["nanog_sox2_dimer_free"]] -= 1
                c[self.state.species_map["nanog_sox2_dimer_single_bound"]] += 1
                species_label = "SOX2b:NANOGf" if is_sox2 else "NANOGb:SOX2f"

            self.state.chromatin_site_bind_times[primary_site] = current_t
            self._update_site_times(
                current_t,
                primary_site,
                "EMPTY",
                species_label,
                reaction_index,
                time_reset_value=current_t,
                is_bound=True,
            )
            self.logger.record_reaction(current_t, reaction_index, primary_site)

        else:
            dangling_markers = (constants.SiteState.NANOGb_SOX2f,) if is_sox2 else (constants.SiteState.NANOGb_NANOGf, constants.SiteState.SOX2b_NANOGf)
            self._execute_tethered_binding_logic(
                current_t, reaction_index, dangling_type=tf_type, dangling_state_markers=dangling_markers
            )

    def _handle_tf_unbinding(self, current_t, reaction_index):
        """Reaction logic for a TF unbinding from a site on the chromatin array.

        Args:
            current_t (int): current time of simulation
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        tf_type = 1 if reaction_index == 6 else 2
        bound_indices = self.state.bound_sites[tf_type][:self.state.bound_count[tf_type]]

        if len(bound_indices) == 0:
            return

        dissociating_site = np.random.choice(bound_indices)
        partner_state = self.state.chromatin_partner_site[dissociating_site]
        
        old_state = self.state.chromatin_state[dissociating_site]
        old_label = self.state.get_species_label(dissociating_site)
        self.state.set_site_state(dissociating_site, constants.SiteState.EMPTY)
        # update the partner/bulk state
        if partner_state >= 0:
            # partner was fully bound, is now dangling
            
            # partner becomes single-bound with a dangling monomer
            p_tf = self.state.get_base_tf(self.state.chromatin_state[partner_state])
            new_state = constants.SiteState.EMPTY
            if p_tf == 1:
                new_state = constants.SiteState.SOX2b_NANOGf # Only nanog can dangle from SOX2
            else:
                new_state = constants.SiteState.NANOGb_SOX2f if tf_type == 1 else constants.SiteState.NANOGb_NANOGf
            self.state.set_site_state(partner_state, new_state)
            partner_tf_type = self.state.get_base_tf(self.state.chromatin_state[partner_state])
            if self._is_heterodimer(tf_type, partner_tf_type):
                self.state.species_counts[self.state.species_map["nanog_sox2_dimer_bound"]] -= 1
                self.state.species_counts[
                    self.state.species_map["nanog_sox2_dimer_single_bound"]
                ] += 1
            else:
                self.state.species_counts[self.state.species_map["nanog_nanog_dimer_bound"]] -= 1
                self.state.species_counts[
                    self.state.species_map["nanog_nanog_dimer_single_bound"]
                ] += 1

        elif old_state in (constants.SiteState.NANOGb_SOX2f, constants.SiteState.NANOGb_NANOGf, constants.SiteState.SOX2b_NANOGf):
            # it was single-bound, dangling arm is now completely free
            dangling_tf_type = 1 if old_state == constants.SiteState.NANOGb_SOX2f else 2
            if self._is_heterodimer(tf_type, dangling_tf_type):
                self.state.species_counts[
                    self.state.species_map["nanog_sox2_dimer_single_bound"]
                ] -= 1
                self.state.species_counts[self.state.species_map["nanog_sox2_dimer_free"]] += 1
            else:
                self.state.species_counts[
                    self.state.species_map["nanog_nanog_dimer_single_bound"]
                ] -= 1
                self.state.species_counts[self.state.species_map["nanog_nanog_dimer_free"]] += 1
        else:  # was a Monomer
            if tf_type == 1:
                self.state.species_counts[self.state.species_map["sox2_monomer_bound"]] -= 1
                self.state.species_counts[self.state.species_map["sox2_monomer_free"]] += 1
            else:
                self.state.species_counts[self.state.species_map["nanog_monomer_bound"]] -= 1
                self.state.species_counts[self.state.species_map["nanog_monomer_free"]] += 1
        surviving_foot_site = partner_state if partner_state >= 0 else -1
        self._update_site_times(
            current_t,
            dissociating_site,
            old_label,
            "EMPTY",
            reaction_index,
            paired_site=surviving_foot_site,
            time_reset_value=current_t,
            is_bound=False,
        )
        self.logger.record_reaction(current_t, reaction_index, dissociating_site)

    def _execute_tethered_binding_logic(
        self,
        current_t,
        reaction_index: int,
        dangling_type: int,
        dangling_state_markers: tuple,
    ):
        """Helper to execute intramolecular binding."""

        masked_matrix = np.copy(self.state.tether_weight_matrix)
        mask = np.zeros(self.state.total_sites, dtype=bool)
        for marker in dangling_state_markers:
            mask |= (self.state.chromatin_state == marker)
        masked_matrix[~mask, :] = 0

        tethered_site, target_vacant_site = fast_sample_spatial_dimer(
            masked_matrix, is_tether=True
        )
        bound_tf_type = self.state.get_base_tf(self.state.chromatin_state[tethered_site])
        old_tether_label = self.state.get_species_label(tethered_site)

        pass # We set target_vacant_site with the partner state directly below
        self.state.set_site_state(tethered_site, constants.SiteState.SOX2b_NANOGb if bound_tf_type == 1 else (constants.SiteState.NANOGb_SOX2b if dangling_type == 1 else constants.SiteState.NANOGb_NANOGb), target_vacant_site)
        self.state.set_site_state(target_vacant_site, constants.SiteState.SOX2b_NANOGb if dangling_type == 1 else (constants.SiteState.NANOGb_SOX2b if bound_tf_type == 1 else constants.SiteState.NANOGb_NANOGb), tethered_site)
        new_label = self.state.get_species_label(tethered_site)

        if self._is_heterodimer(bound_tf_type, dangling_type):
            self.state.species_counts[self.state.species_map["nanog_sox2_dimer_single_bound"]] -= 1
            self.state.species_counts[self.state.species_map["nanog_sox2_dimer_bound"]] += 1
        else:
            self.state.species_counts[
                self.state.species_map["nanog_nanog_dimer_single_bound"]
            ] -= 1
            self.state.species_counts[self.state.species_map["nanog_nanog_dimer_bound"]] += 1

        self._update_site_times(
            current_t,
            tethered_site,
            old_tether_label,
            new_label,
            reaction_index,
            paired_site=target_vacant_site,
            time_reset_value=current_t,
            is_bound=True,
        )
        self._update_site_times(
            current_t,
            target_vacant_site,
            "EMPTY",
            new_label,
            reaction_index,
            paired_site=tethered_site,
            time_reset_value=current_t,
            is_bound=True,
        )
        self.logger.record_reaction(
            current_t, reaction_index, tethered_site, target_vacant_site
        )

    def _execute_site_dimerise(self, current_t, reaction_index):
        """Reaction logic for an already bound TF to dimerise with another TF at some index in chromatin_lattice.

        Args:
            current_t (int): current time of simulation
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        if self.state.total_dimer_weight_symmetric > 0:
            dimer_index, dimer_partner_index = fast_sample_spatial_dimer(
                self.state.dimer_weight_matrix
            )

            old_label_index = self.state.get_species_label(dimer_index)
            old_label_partner = self.state.get_species_label(dimer_partner_index)

            self.state.set_site_state(
                dimer_index, constants.SiteState.SOX2b_NANOGb if self.state.get_base_tf(self.state.chromatin_state[dimer_index]) == 1 else (constants.SiteState.NANOGb_SOX2b if self.state.get_base_tf(self.state.chromatin_state[dimer_partner_index]) == 1 else constants.SiteState.NANOGb_NANOGb), dimer_partner_index
            )
            self.state.set_site_state(
                dimer_partner_index, constants.SiteState.SOX2b_NANOGb if self.state.get_base_tf(self.state.chromatin_state[dimer_partner_index]) == 1 else (constants.SiteState.NANOGb_SOX2b if self.state.get_base_tf(self.state.chromatin_state[dimer_index]) == 1 else constants.SiteState.NANOGb_NANOGb), dimer_index
            )
            new_label_index = self.state.get_species_label(dimer_index)
            new_label_partner = self.state.get_species_label(dimer_partner_index)

            dimer_index_tf_type = self.state.get_base_tf(self.state.chromatin_state[dimer_index])
            dimer_partner_index_tf_type = self.state.get_base_tf(self.state.chromatin_state[
                dimer_partner_index
            ])

            # update species counts accordingly
            if self._is_heterodimer(dimer_index_tf_type, dimer_partner_index_tf_type):
                self.state.species_counts[self.state.species_map["nanog_sox2_dimer_bound"]] += 1
                self.state.species_counts[self.state.species_map["sox2_monomer_bound"]] -= 1
                self.state.species_counts[self.state.species_map["nanog_monomer_bound"]] -= 1
            else:
                self.state.species_counts[self.state.species_map["nanog_nanog_dimer_bound"]] += 1
                self.state.species_counts[self.state.species_map["nanog_monomer_bound"]] -= 2

            self._update_site_times(
                current_t,
                dimer_index,
                old_label_index,
                new_label_index,
                reaction_index,
                paired_site=dimer_partner_index,
                time_reset_value=current_t,
                is_bound=True,
            )
            self._update_site_times(
                current_t,
                dimer_partner_index,
                old_label_partner,
                new_label_partner,
                reaction_index,
                paired_site=dimer_index,
                time_reset_value=current_t,
                is_bound=True,
            )
            self.logger.record_reaction(
                current_t, reaction_index, dimer_index, dimer_partner_index
            )

    def _execute_bulk_dimerise(self, current_t, reaction_index):
        """Updates species_counts for all non-bound TFs (i.e. only *_free tagged variables)

        Args:
            current_t (int): current time of simulation
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        s_free = self.state.species_counts[self.state.species_map["sox2_monomer_free"]]
        n_free = self.state.species_counts[self.state.species_map["nanog_monomer_free"]]
        # check if there are any valid reactions
        nn_dimers = n_free * (n_free - 1) / 2.0
        sn_dimers = s_free * n_free
        total_dimers = nn_dimers + sn_dimers

        if total_dimers > 0:
            if (np.random.rand() < nn_dimers / total_dimers) and n_free >= 2:
                self.state.species_counts[self.state.species_map["nanog_nanog_dimer_free"]] += 1
                self.state.species_counts[self.state.species_map["nanog_monomer_free"]] -= 2
            elif s_free >= 1 and n_free >= 1:
                self.state.species_counts[self.state.species_map["nanog_sox2_dimer_free"]] += 1
                self.state.species_counts[self.state.species_map["sox2_monomer_free"]] -= 1
                self.state.species_counts[self.state.species_map["nanog_monomer_free"]] -= 1
        self.logger.record_reaction(current_t, reaction_index)

    def _execute_site_bulk_dimerise(self, current_t, reaction_index):
        """Updates species_counts and chromatin arrays for a reaction between a *_free species and a *_bound species.

        Args:
            current_t (int): current time of simulation
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        s_free = self.state.species_counts[self.state.species_map["sox2_monomer_free"]]
        n_free = self.state.species_counts[self.state.species_map["nanog_monomer_free"]]

        undimered_s = self.state.undimered_sites[1][:self.state.undimered_count[1]]
        undimered_n = self.state.undimered_sites[2][:self.state.undimered_count[2]]

        sn_dimers = len(undimered_s) * n_free
        ns_dimers = len(undimered_n) * s_free
        nn_dimers = len(undimered_n) * n_free

        total_dimers = sn_dimers + ns_dimers + nn_dimers
        if total_dimers <= 0:
            return

        rand_val = np.random.rand() * total_dimers
        site = -1

        if rand_val < sn_dimers and len(undimered_s) > 0:
            site = np.random.choice(list(undimered_s))
            old_label = self.state.get_species_label(site)
            self.state.set_site_state(site, constants.SiteState.SOX2b_NANOGf if old_label == 'SOX2b' else constants.SiteState.NANOGb_NANOGf)  # SOX2 binds dangling NANOG
            new_label = self.state.get_species_label(site)
            self.state.species_counts[self.state.species_map["sox2_monomer_bound"]] -= 1
            self.state.species_counts[self.state.species_map["nanog_monomer_free"]] -= 1
            self.state.species_counts[self.state.species_map["nanog_sox2_dimer_single_bound"]] += 1

        elif (
            rand_val < sn_dimers + ns_dimers
            and len(undimered_n) > 0
        ):
            site = np.random.choice(list(undimered_n))
            old_label = self.state.get_species_label(site)
            self.state.set_site_state(site, constants.SiteState.NANOGb_SOX2f)  # NANOG binds dangling SOX2
            new_label = self.state.get_species_label(site)
            self.state.species_counts[self.state.species_map["nanog_monomer_bound"]] -= 1
            self.state.species_counts[self.state.species_map["sox2_monomer_free"]] -= 1
            self.state.species_counts[self.state.species_map["nanog_sox2_dimer_single_bound"]] += 1

        elif len(undimered_n) > 0:
            site = np.random.choice(list(undimered_n))
            old_label = self.state.get_species_label(site)
            self.state.set_site_state(site, constants.SiteState.SOX2b_NANOGf if old_label == 'SOX2b' else constants.SiteState.NANOGb_NANOGf)  # NANOG binds dangling NANOG
            new_label = self.state.get_species_label(site)
            self.state.species_counts[self.state.species_map["nanog_monomer_bound"]] -= 1
            self.state.species_counts[self.state.species_map["nanog_monomer_free"]] -= 1
            self.state.species_counts[
                self.state.species_map["nanog_nanog_dimer_single_bound"]
            ] += 1

        if site != -1:
            self._update_site_times(
                current_t,
                site,
                old_label,
                new_label,
                reaction_index,
                time_reset_value=current_t,
                is_bound=True,
            )
            self.logger.record_reaction(current_t, reaction_index, site)
            
    def _execute_site_dedimerise(self, current_t, reaction_index):
        """Updates species_counts and chromatin arrays when removing dimer status between two bound indices.

        Args:
            current_t (int): current time of simulation.
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        dimered_indices = self.state.dimered_sites[:self.state.dimered_count]
        if len(dimered_indices) > 0:
            site1 = np.random.choice(dimered_indices)
            site2 = self.state.chromatin_partner_site[site1]
            dimer_index_tf_type = self.state.get_base_tf(self.state.chromatin_state[site1])
            dimer_partner_index_tf_type = self.state.get_base_tf(self.state.chromatin_state[site2])

            # Read OLD label before breaking them apart
            old_label = self.state.get_species_label(site1)

            # Break them apart
            self.state.set_site_state(site1, constants.SiteState.SOX2b if self.state.get_base_tf(self.state.chromatin_state[site1]) == 1 else constants.SiteState.NANOGb)
            self.state.set_site_state(site2, constants.SiteState.SOX2b if self.state.get_base_tf(self.state.chromatin_state[site2]) == 1 else constants.SiteState.NANOGb)

            # Read NEW labels
            new_sp1 = self.state.get_species_label(site1)
            new_sp2 = self.state.get_species_label(site2)

            if self._is_heterodimer(dimer_index_tf_type, dimer_partner_index_tf_type):
                self.state.species_counts[self.state.species_map["nanog_sox2_dimer_bound"]] -= 1
                self.state.species_counts[self.state.species_map["sox2_monomer_bound"]] += 1
                self.state.species_counts[self.state.species_map["nanog_monomer_bound"]] += 1
            else:
                self.state.species_counts[self.state.species_map["nanog_nanog_dimer_bound"]] -= 1
                self.state.species_counts[self.state.species_map["nanog_monomer_bound"]] += 2

            self._update_site_times(
                current_t,
                site1,
                old_label,
                new_sp1,
                reaction_index,
                paired_site=site2,
                time_reset_value=current_t,
                is_bound=True,
            )
            self._update_site_times(
                current_t,
                site2,
                old_label,
                new_sp2,
                reaction_index,
                paired_site=site2,
                time_reset_value=current_t,
                is_bound=True,
            )
            self.logger.record_reaction(current_t, reaction_index, site1, site2)

    def _execute_bulk_dedimerise(self, current_t, reaction_index):
        """Updates species_counts and chromatin arrays when removing dimer status between two *_free variables.

        Args:
            current_t (int): current time of simulation.
            reaction_index (int): selected reaction to occur according to Gillespie logic.
        """
        ns_free = self.state.species_counts[self.state.species_map["nanog_sox2_dimer_free"]]
        nn_free = self.state.species_counts[self.state.species_map["nanog_nanog_dimer_free"]]
        total_dimers = ns_free + nn_free

        if total_dimers > 0:
            if (np.random.rand() < ns_free / total_dimers) and ns_free >= 1:
                self.state.species_counts[self.state.species_map["nanog_sox2_dimer_free"]] -= 1
                self.state.species_counts[self.state.species_map["sox2_monomer_free"]] += 1
                self.state.species_counts[self.state.species_map["nanog_monomer_free"]] += 1
            elif nn_free >= 1:
                self.state.species_counts[self.state.species_map["nanog_nanog_dimer_free"]] -= 1
                self.state.species_counts[self.state.species_map["nanog_monomer_free"]] += 2
        self.logger.record_reaction(current_t, reaction_index)

    def _execute_single_bound_dedimerise(self, current_t, reaction_index):
        """Reaction logic for a dangling dimer snapping apart, dropping a monomer back to bulk."""
        dangling_sites = [
            s
            for s in range(self.state.total_sites)
            if self.state.chromatin_state[s] in (constants.SiteState.NANOGb_SOX2f, constants.SiteState.NANOGb_NANOGf, constants.SiteState.SOX2b_NANOGf)
        ]
        if not dangling_sites:
            return

        site = np.random.choice(dangling_sites)
        bound_tf_type = self.state.get_base_tf(self.state.chromatin_state[site])
        partner_state = self.state.chromatin_partner_site[site]
        dangling_tf_type = 1 if self.state.chromatin_state[site] == constants.SiteState.NANOGb_SOX2f else 2

        old_label = self.state.get_species_label(site)
        self.state.set_site_state(site, constants.SiteState.SOX2b if self.state.get_base_tf(self.state.chromatin_state[site]) == 1 else constants.SiteState.NANOGb)
        new_label = self.state.get_species_label(site)

        if dangling_tf_type == 1:
            self.state.species_counts[self.state.species_map["sox2_monomer_free"]] += 1
        else:
            self.state.species_counts[self.state.species_map["nanog_monomer_free"]] += 1

        if self._is_heterodimer(bound_tf_type, dangling_tf_type):
            self.state.species_counts[self.state.species_map["nanog_sox2_dimer_single_bound"]] -= 1
            if bound_tf_type == 1:
                self.state.species_counts[self.state.species_map["sox2_monomer_bound"]] += 1
            else:
                self.state.species_counts[self.state.species_map["nanog_monomer_bound"]] += 1
        else:
            self.state.species_counts[
                self.state.species_map["nanog_nanog_dimer_single_bound"]
            ] -= 1
            self.state.species_counts[self.state.species_map["nanog_monomer_bound"]] += 1

        self._update_site_times(
            current_t,
            site,
            old_label,
            new_label,
            reaction_index,
            paired_site=-1,
            time_reset_value=current_t,
            is_bound=True,
        )
        self.logger.record_reaction(current_t, reaction_index, site)
