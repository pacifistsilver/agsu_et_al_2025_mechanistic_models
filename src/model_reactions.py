import numpy as np
from . import constants
from .model_state import ModelState
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

class ReactionStrategy:
    def execute(self, state, current_t, reaction_index, context, publisher):
        raise NotImplementedError

class LogOnlyStrategy(ReactionStrategy):
    def execute(self, state, current_t, reaction_index, context, publisher):
        return -1, -1

class TfBindingStrategy(ReactionStrategy):
    def execute(self, state, current_t, reaction_index, context, publisher):
        vacant_sites = state.vacant_sites[:state.vacant_count]
        num_vacant = len(vacant_sites)
        c = state.species_counts

        is_sox2 = reaction_index == 2
        tf_type = 1 if is_sox2 else 2

        bulk_monomer = c[state.species_map["sox2_monomer_free" if is_sox2 else "nanog_monomer_free"]] * num_vacant
        bulk_heterodimer = c[state.species_map["nanog_sox2_dimer_free"]] * num_vacant
        bulk_homodimer = 0 if is_sox2 else (2 * c[state.species_map["nanog_nanog_dimer_free"]]) * num_vacant
        tether_w = state.total_tether_weight_s if is_sox2 else state.total_tether_weight_n

        total_bulk = bulk_monomer + bulk_homodimer + bulk_heterodimer
        total_pool = total_bulk + tether_w

        if total_pool <= 0:
            return -1, -1

        rand_val = np.random.rand() * total_pool

        if rand_val < total_bulk:
            primary_site = np.random.choice(vacant_sites)
            
            if rand_val < bulk_monomer:
                state.set_site_state(primary_site, constants.SiteState.SOX2b if is_sox2 else constants.SiteState.NANOGb)
                c[state.species_map["sox2_monomer_free" if is_sox2 else "nanog_monomer_free"]] -= 1
                c[state.species_map["sox2_monomer_bound" if is_sox2 else "nanog_monomer_bound"]] += 1
                species_label = "SOX2b" if is_sox2 else "NANOGb"
            elif not is_sox2 and rand_val < bulk_monomer + bulk_homodimer:
                state.set_site_state(primary_site, constants.SiteState.NANOGb_NANOGf)
                c[state.species_map["nanog_nanog_dimer_free"]] -= 1
                c[state.species_map["nanog_nanog_dimer_single_bound"]] += 1
                species_label = "NANOGb:NANOGf"
            else:
                state.set_site_state(primary_site, constants.SiteState.SOX2b_NANOGf if is_sox2 else constants.SiteState.NANOGb_SOX2f)
                c[state.species_map["nanog_sox2_dimer_free"]] -= 1
                c[state.species_map["nanog_sox2_dimer_single_bound"]] += 1
                species_label = "SOX2b:NANOGf" if is_sox2 else "NANOGb:SOX2f"

            context._update_site_times(
                current_t,
                primary_site,
                "EMPTY",
                species_label,
                reaction_index,
                publisher,
                time_reset_value=current_t,
                is_bound=True,
            )
            return primary_site, -1

        else:
            dangling_markers = (constants.SiteState.NANOGb_SOX2f,) if is_sox2 else (constants.SiteState.NANOGb_NANOGf, constants.SiteState.SOX2b_NANOGf)
            return context._execute_tethered_binding_logic(
                current_t, reaction_index, dangling_type=tf_type, dangling_state_markers=dangling_markers, publisher=publisher
            )

class TfUnbindingStrategy(ReactionStrategy):
    def execute(self, state, current_t, reaction_index, context, publisher):
        tf_type = 1 if reaction_index == 6 else 2
        bound_indices = state.bound_sites[tf_type][:state.bound_count[tf_type]]

        if len(bound_indices) == 0:
            return -1, -1

        dissociating_site = np.random.choice(bound_indices)
        partner_state = state.chromatin_partner_site[dissociating_site]
        
        old_state = state.chromatin_state[dissociating_site]
        old_label = state.get_species_label(dissociating_site)
        state.set_site_state(dissociating_site, constants.SiteState.EMPTY)
        
        if partner_state >= 0:
            p_tf = state.get_base_tf(state.chromatin_state[partner_state])
            new_state = constants.SiteState.EMPTY
            if p_tf == 1:
                new_state = constants.SiteState.SOX2b_NANOGf
            else:
                new_state = constants.SiteState.NANOGb_SOX2f if tf_type == 1 else constants.SiteState.NANOGb_NANOGf
            state.set_site_state(partner_state, new_state)
            partner_tf_type = state.get_base_tf(state.chromatin_state[partner_state])
            if context._is_heterodimer(tf_type, partner_tf_type):
                state.species_counts[state.species_map["nanog_sox2_dimer_bound"]] -= 1
                state.species_counts[state.species_map["nanog_sox2_dimer_single_bound"]] += 1
            else:
                state.species_counts[state.species_map["nanog_nanog_dimer_bound"]] -= 1
                state.species_counts[state.species_map["nanog_nanog_dimer_single_bound"]] += 1

        elif old_state in (constants.SiteState.NANOGb_SOX2f, constants.SiteState.NANOGb_NANOGf, constants.SiteState.SOX2b_NANOGf):
            dangling_tf_type = 1 if old_state == constants.SiteState.NANOGb_SOX2f else 2
            if context._is_heterodimer(tf_type, dangling_tf_type):
                state.species_counts[state.species_map["nanog_sox2_dimer_single_bound"]] -= 1
                state.species_counts[state.species_map["nanog_sox2_dimer_free"]] += 1
            else:
                state.species_counts[state.species_map["nanog_nanog_dimer_single_bound"]] -= 1
                state.species_counts[state.species_map["nanog_nanog_dimer_free"]] += 1
        else:
            if tf_type == 1:
                state.species_counts[state.species_map["sox2_monomer_bound"]] -= 1
                state.species_counts[state.species_map["sox2_monomer_free"]] += 1
            else:
                state.species_counts[state.species_map["nanog_monomer_bound"]] -= 1
                state.species_counts[state.species_map["nanog_monomer_free"]] += 1
        
        surviving_foot_site = partner_state if partner_state >= 0 else -1
        context._update_site_times(
            current_t,
            dissociating_site,
            old_label,
            "EMPTY",
            reaction_index,
            publisher,
            paired_site=surviving_foot_site,
            time_reset_value=current_t,
            is_bound=False,
        )
        return dissociating_site, -1

class SiteDimeriseStrategy(ReactionStrategy):
    def execute(self, state, current_t, reaction_index, context, publisher):
        if state.total_dimer_weight_symmetric > 0:
            dimer_index, dimer_partner_index = fast_sample_spatial_dimer(
                state.dimer_weight_matrix
            )

            old_label_index = state.get_species_label(dimer_index)
            old_label_partner = state.get_species_label(dimer_partner_index)

            state.set_site_state(
                dimer_index, constants.SiteState.SOX2b_NANOGb if state.get_base_tf(state.chromatin_state[dimer_index]) == 1 else (constants.SiteState.NANOGb_SOX2b if state.get_base_tf(state.chromatin_state[dimer_partner_index]) == 1 else constants.SiteState.NANOGb_NANOGb), dimer_partner_index
            )
            state.set_site_state(
                dimer_partner_index, constants.SiteState.SOX2b_NANOGb if state.get_base_tf(state.chromatin_state[dimer_partner_index]) == 1 else (constants.SiteState.NANOGb_SOX2b if state.get_base_tf(state.chromatin_state[dimer_index]) == 1 else constants.SiteState.NANOGb_NANOGb), dimer_index
            )
            new_label_index = state.get_species_label(dimer_index)
            new_label_partner = state.get_species_label(dimer_partner_index)

            dimer_index_tf_type = state.get_base_tf(state.chromatin_state[dimer_index])
            dimer_partner_index_tf_type = state.get_base_tf(state.chromatin_state[dimer_partner_index])

            if context._is_heterodimer(dimer_index_tf_type, dimer_partner_index_tf_type):
                state.species_counts[state.species_map["nanog_sox2_dimer_bound"]] += 1
                state.species_counts[state.species_map["sox2_monomer_bound"]] -= 1
                state.species_counts[state.species_map["nanog_monomer_bound"]] -= 1
            else:
                state.species_counts[state.species_map["nanog_nanog_dimer_bound"]] += 1
                state.species_counts[state.species_map["nanog_monomer_bound"]] -= 2

            context._update_site_times(
                current_t,
                dimer_index,
                old_label_index,
                new_label_index,
                reaction_index,
                publisher,
                paired_site=dimer_partner_index,
                time_reset_value=current_t,
                is_bound=True,
            )
            context._update_site_times(
                current_t,
                dimer_partner_index,
                old_label_partner,
                new_label_partner,
                reaction_index,
                publisher,
                paired_site=dimer_index,
                time_reset_value=current_t,
                is_bound=True,
            )
            return dimer_index, dimer_partner_index
        return -1, -1

class BulkDimeriseStrategy(ReactionStrategy):
    def execute(self, state, current_t, reaction_index, context, publisher):
        s_free = state.species_counts[state.species_map["sox2_monomer_free"]]
        n_free = state.species_counts[state.species_map["nanog_monomer_free"]]
        nn_dimers = n_free * (n_free - 1) / 2.0
        sn_dimers = s_free * n_free
        total_dimers = nn_dimers + sn_dimers

        if total_dimers > 0:
            if (np.random.rand() < nn_dimers / total_dimers) and n_free >= 2:
                state.species_counts[state.species_map["nanog_nanog_dimer_free"]] += 1
                state.species_counts[state.species_map["nanog_monomer_free"]] -= 2
            elif s_free >= 1 and n_free >= 1:
                state.species_counts[state.species_map["nanog_sox2_dimer_free"]] += 1
                state.species_counts[state.species_map["sox2_monomer_free"]] -= 1
                state.species_counts[state.species_map["nanog_monomer_free"]] -= 1
        return -1, -1

class SiteDedimeriseStrategy(ReactionStrategy):
    def execute(self, state, current_t, reaction_index, context, publisher):
        dimered_indices = state.dimered_sites[:state.dimered_count]
        if len(dimered_indices) > 0:
            site1 = np.random.choice(dimered_indices)
            site2 = state.chromatin_partner_site[site1]
            dimer_index_tf_type = state.get_base_tf(state.chromatin_state[site1])
            dimer_partner_index_tf_type = state.get_base_tf(state.chromatin_state[site2])

            old_label = state.get_species_label(site1)

            state.set_site_state(site1, constants.SiteState.SOX2b if state.get_base_tf(state.chromatin_state[site1]) == 1 else constants.SiteState.NANOGb)
            state.set_site_state(site2, constants.SiteState.SOX2b if state.get_base_tf(state.chromatin_state[site2]) == 1 else constants.SiteState.NANOGb)

            new_sp1 = state.get_species_label(site1)
            new_sp2 = state.get_species_label(site2)

            if context._is_heterodimer(dimer_index_tf_type, dimer_partner_index_tf_type):
                state.species_counts[state.species_map["nanog_sox2_dimer_bound"]] -= 1
                state.species_counts[state.species_map["sox2_monomer_bound"]] += 1
                state.species_counts[state.species_map["nanog_monomer_bound"]] += 1
            else:
                state.species_counts[state.species_map["nanog_nanog_dimer_bound"]] -= 1
                state.species_counts[state.species_map["nanog_monomer_bound"]] += 2

            context._update_site_times(
                current_t,
                site1,
                old_label,
                new_sp1,
                reaction_index,
                publisher,
                paired_site=site2,
                time_reset_value=current_t,
                is_bound=True,
            )
            context._update_site_times(
                current_t,
                site2,
                old_label,
                new_sp2,
                reaction_index,
                publisher,
                paired_site=site2,
                time_reset_value=current_t,
                is_bound=True,
            )
            return site1, site2
        return -1, -1

class BulkDedimeriseStrategy(ReactionStrategy):
    def execute(self, state, current_t, reaction_index, context, publisher):
        ns_free = state.species_counts[state.species_map["nanog_sox2_dimer_free"]]
        nn_free = state.species_counts[state.species_map["nanog_nanog_dimer_free"]]
        total_dimers = ns_free + nn_free

        if total_dimers > 0:
            if (np.random.rand() < ns_free / total_dimers) and ns_free >= 1:
                state.species_counts[state.species_map["nanog_sox2_dimer_free"]] -= 1
                state.species_counts[state.species_map["sox2_monomer_free"]] += 1
                state.species_counts[state.species_map["nanog_monomer_free"]] += 1
            elif nn_free >= 1:
                state.species_counts[state.species_map["nanog_nanog_dimer_free"]] -= 1
                state.species_counts[state.species_map["nanog_monomer_free"]] += 2
        return -1, -1

class SiteBulkDimeriseStrategy(ReactionStrategy):
    def execute(self, state, current_t, reaction_index, context, publisher):
        s_free = state.species_counts[state.species_map["sox2_monomer_free"]]
        n_free = state.species_counts[state.species_map["nanog_monomer_free"]]

        undimered_s = state.undimered_sites[1][:state.undimered_count[1]]
        undimered_n = state.undimered_sites[2][:state.undimered_count[2]]

        sn_dimers = len(undimered_s) * n_free
        ns_dimers = len(undimered_n) * s_free
        nn_dimers = len(undimered_n) * n_free

        total_dimers = sn_dimers + ns_dimers + nn_dimers
        if total_dimers <= 0:
            return -1, -1

        rand_val = np.random.rand() * total_dimers
        site = -1

        if rand_val < sn_dimers and len(undimered_s) > 0:
            site = np.random.choice(list(undimered_s))
            old_label = state.get_species_label(site)
            state.set_site_state(site, constants.SiteState.SOX2b_NANOGf if old_label == 'SOX2b' else constants.SiteState.NANOGb_NANOGf)
            new_label = state.get_species_label(site)
            state.species_counts[state.species_map["sox2_monomer_bound"]] -= 1
            state.species_counts[state.species_map["nanog_monomer_free"]] -= 1
            state.species_counts[state.species_map["nanog_sox2_dimer_single_bound"]] += 1

        elif rand_val < sn_dimers + ns_dimers and len(undimered_n) > 0:
            site = np.random.choice(list(undimered_n))
            old_label = state.get_species_label(site)
            state.set_site_state(site, constants.SiteState.NANOGb_SOX2f)
            new_label = state.get_species_label(site)
            state.species_counts[state.species_map["nanog_monomer_bound"]] -= 1
            state.species_counts[state.species_map["sox2_monomer_free"]] -= 1
            state.species_counts[state.species_map["nanog_sox2_dimer_single_bound"]] += 1

        elif len(undimered_n) > 0:
            site = np.random.choice(list(undimered_n))
            old_label = state.get_species_label(site)
            state.set_site_state(site, constants.SiteState.SOX2b_NANOGf if old_label == 'SOX2b' else constants.SiteState.NANOGb_NANOGf)
            new_label = state.get_species_label(site)
            state.species_counts[state.species_map["nanog_monomer_bound"]] -= 1
            state.species_counts[state.species_map["nanog_monomer_free"]] -= 1
            state.species_counts[state.species_map["nanog_nanog_dimer_single_bound"]] += 1

        if site != -1:
            context._update_site_times(
                current_t,
                site,
                old_label,
                new_label,
                reaction_index,
                publisher,
                time_reset_value=current_t,
                is_bound=True,
            )
            return site, -1
        return -1, -1

class SingleBoundDedimeriseStrategy(ReactionStrategy):
    def execute(self, state, current_t, reaction_index, context, publisher):
        dangling_sites = [
            s for s in range(state.total_sites)
            if state.chromatin_state[s] in (constants.SiteState.NANOGb_SOX2f, constants.SiteState.NANOGb_NANOGf, constants.SiteState.SOX2b_NANOGf)
        ]
        if not dangling_sites:
            return -1, -1

        site = np.random.choice(dangling_sites)
        bound_tf_type = state.get_base_tf(state.chromatin_state[site])
        partner_state = state.chromatin_partner_site[site]
        dangling_tf_type = 1 if state.chromatin_state[site] == constants.SiteState.NANOGb_SOX2f else 2

        old_label = state.get_species_label(site)
        state.set_site_state(site, constants.SiteState.SOX2b if state.get_base_tf(state.chromatin_state[site]) == 1 else constants.SiteState.NANOGb)
        new_label = state.get_species_label(site)

        if dangling_tf_type == 1:
            state.species_counts[state.species_map["sox2_monomer_free"]] += 1
        else:
            state.species_counts[state.species_map["nanog_monomer_free"]] += 1

        if context._is_heterodimer(bound_tf_type, dangling_tf_type):
            state.species_counts[state.species_map["nanog_sox2_dimer_single_bound"]] -= 1
            if bound_tf_type == 1:
                state.species_counts[state.species_map["sox2_monomer_bound"]] += 1
            else:
                state.species_counts[state.species_map["nanog_monomer_bound"]] += 1
        else:
            state.species_counts[state.species_map["nanog_nanog_dimer_single_bound"]] -= 1
            state.species_counts[state.species_map["nanog_monomer_bound"]] += 1

        context._update_site_times(
            current_t,
            site,
            old_label,
            new_label,
            reaction_index,
            publisher,
            paired_site=-1,
            time_reset_value=current_t,
            is_bound=True,
        )
        return site, -1

class ModelReactions:
    """Contains reaction logic methods to update ModelState attributes as dictated by the class ModelCall.

    Attributes:
        state (class): inherits simulation state information and important state update methods from ModelState.
    """

    def __init__(self, state: ModelState):
        self.state = state
        self.strategies = {
            2: TfBindingStrategy(),
            3: TfBindingStrategy(),
            6: TfUnbindingStrategy(),
            7: TfUnbindingStrategy(),
            10: SiteDimeriseStrategy(),
            11: BulkDimeriseStrategy(),
            12: SiteDedimeriseStrategy(),
            13: BulkDedimeriseStrategy(),
            14: SiteBulkDimeriseStrategy(),
            15: SingleBoundDedimeriseStrategy(),
        }
        self.log_only_strategy = LogOnlyStrategy()

    def execute(self, current_t, reaction_index, publisher=None):
        strategy = self.strategies.get(reaction_index, self.log_only_strategy)
        return strategy.execute(self.state, current_t, reaction_index, self, publisher)

    def _is_heterodimer(self, tf1_type, tf2_type):
        return (tf1_type == 1 and tf2_type == 2) or (tf1_type == 2 and tf2_type == 1)

    def _update_site_times(
        self,
        current_sim_time,
        site,
        old_species_label,
        new_species_label,
        reaction,
        publisher,
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

        if publisher:
            publisher.notify_site_state_changed(
                current_sim_time,
                site,
                old_species_label,
                new_species_label,
                reaction_name,
                duration,
                paired_site,
                is_bound,
            )
        self.state.chromatin_site_bind_times[site] = time_reset_value

    def _execute_tethered_binding_logic(
        self,
        current_t,
        reaction_index: int,
        dangling_type: int,
        dangling_state_markers: tuple,
        publisher,
    ):
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

        self.state.set_site_state(tethered_site, constants.SiteState.SOX2b_NANOGb if bound_tf_type == 1 else (constants.SiteState.NANOGb_SOX2b if dangling_type == 1 else constants.SiteState.NANOGb_NANOGb), target_vacant_site)
        self.state.set_site_state(target_vacant_site, constants.SiteState.SOX2b_NANOGb if dangling_type == 1 else (constants.SiteState.NANOGb_SOX2b if bound_tf_type == 1 else constants.SiteState.NANOGb_NANOGb), tethered_site)
        new_label = self.state.get_species_label(tethered_site)

        if self._is_heterodimer(bound_tf_type, dangling_type):
            self.state.species_counts[self.state.species_map["nanog_sox2_dimer_single_bound"]] -= 1
            self.state.species_counts[self.state.species_map["nanog_sox2_dimer_bound"]] += 1
        else:
            self.state.species_counts[self.state.species_map["nanog_nanog_dimer_single_bound"]] -= 1
            self.state.species_counts[self.state.species_map["nanog_nanog_dimer_bound"]] += 1

        self._update_site_times(
            current_t,
            tethered_site,
            old_tether_label,
            new_label,
            reaction_index,
            publisher,
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
            publisher,
            paired_site=tethered_site,
            time_reset_value=current_t,
            is_bound=True,
        )
        return tethered_site, target_vacant_site
