import numpy as np
from typing import List
from . import constants
from .config_default import SPECIES_NAMES, REACTION_NAMES
from .model_utils import TranscriptionFactor


def _add_to_tracker(arr, idx_arr, count, val):
    if idx_arr[val] == -1:
        arr[count] = val
        idx_arr[val] = count
        return count + 1
    return count

def _remove_from_tracker(arr, idx_arr, count, val):
    pos = idx_arr[val]
    if pos != -1:
        last_val = arr[count - 1]
        arr[pos] = last_val
        idx_arr[last_val] = pos
        idx_arr[val] = -1
        return count - 1
    return count

class ModelState:
    """Tracks the state of the simulation and includes commonly called methods."""

    def __init__(
        self,
        tfs: List[TranscriptionFactor],
        total_binding_sites: int,
        initial_species_states: dict,
        promoter_site: int = None,
    ):
        self.tfs = {tf.id: tf for tf in tfs}
        self.species_map = {name: i for i, name in enumerate(SPECIES_NAMES)}
        self.species_names = SPECIES_NAMES
        self.reaction_names = REACTION_NAMES
        self.total_sites = total_binding_sites
        self.promoter_site = (
            promoter_site
            if promoter_site is not None
            else int((total_binding_sites - 1) / 2)
        )

        self.species_counts = np.zeros(len(SPECIES_NAMES), dtype=np.int32)
        for name, idx in self.species_map.items():
            self.species_counts[idx] = initial_species_states.get(name, 0)
            
        self.is_activator = np.zeros(3, dtype=bool)
        for tf in tfs:
            self.is_activator[tf.id] = tf.is_activator

        # chromatin tracking arrays
        self.chromatin_state = np.zeros(self.total_sites, dtype=np.int8)
        
        
        self.chromatin_site_bind_times = np.full(self.total_sites, -1.0)

        # -1: monomer, -2: SOX2f:dimer, -3: NANOGf:dimer, >= 0: both bound
        self.chromatin_partner_site = np.full(self.total_sites, -1, dtype=np.int32)

        # Trackers (no more sets)
        self.vacant_sites = np.arange(self.total_sites, dtype=np.int32)
        self.vacant_idx = np.arange(self.total_sites, dtype=np.int32)
        self.vacant_count = self.total_sites

        self.bound_sites = np.zeros((3, self.total_sites), dtype=np.int32)
        self.bound_idx = np.full((3, self.total_sites), -1, dtype=np.int32)
        self.bound_count = np.zeros(3, dtype=np.int32)
        
        self.undimered_sites = np.zeros((3, self.total_sites), dtype=np.int32)
        self.undimered_idx = np.full((3, self.total_sites), -1, dtype=np.int32)
        self.undimered_count = np.zeros(3, dtype=np.int32)
        
        self.dimered_sites = np.zeros(self.total_sites, dtype=np.int32)
        self.dimered_idx = np.full(self.total_sites, -1, dtype=np.int32)
        self.dimered_count = 0
        
        self.dangling_counts = np.zeros(4, dtype=np.int32)

        # spatial distance matrices
        site_indicies = np.arange(self.total_sites)
        potential_dimer_partners = np.abs(
            site_indicies[:, None] - site_indicies[None, :]
        )
        self.weighted_potential_dimer_partners = np.exp(
            -potential_dimer_partners / 1
        )

        weight_matrix = np.ones((self.total_sites, self.total_sites))
        np.fill_diagonal(weight_matrix, 0.0)

        raw_weight_matrix_row_sum = np.max(weight_matrix.sum(axis=1))
        self.weighted_potential_dimer_partners = weight_matrix / raw_weight_matrix_row_sum

        self.total_dimer_weight_symmetric = 0.0
        self.total_tether_weight_s = 0.0
        self.total_tether_weight_n = 0.0

        # increment weights
        self.dimer_weight_matrix = np.zeros_like(self.weighted_potential_dimer_partners)
        
        self.tether_weight_s = np.zeros(self.total_sites)
        self.tether_weight_n = np.zeros(self.total_sites)
        self.weighted_potential_tether_partners = self.weighted_potential_dimer_partners
        self.tether_weight_matrix = np.zeros_like(self.weighted_potential_dimer_partners)

    def get_species_label(self, site):
        from . import constants
        return constants.STATE_STRINGS[self.chromatin_state[site]]

    def update_site_weights(self, site):
        """_summary_ Update spatial weight matrices to decide dimerisation and tethering partners.
        
        Args:
            site (int): index of the site to update.
        """
        from .constants import SiteState
        old_dimer_row_sum = np.sum(self.dimer_weight_matrix[site, :])
        new_dimer_row = np.zeros(self.total_sites)

        is_undimered = (self.chromatin_state == SiteState.SOX2b) | (self.chromatin_state == SiteState.NANOGb)

        if self.chromatin_state[site] in (SiteState.SOX2b, SiteState.NANOGb):
            tf_type = self.get_base_tf(self.chromatin_state[site])
            
            base_tfs = np.zeros(self.total_sites, dtype=int)
            for i in range(self.total_sites):
                base_tfs[i] = self.get_base_tf(self.chromatin_state[i])

            if tf_type == 1:
                valid_mask = (base_tfs == 2) & is_undimered
            elif tf_type == 2:
                valid_mask = ((base_tfs == 1) | (base_tfs == 2)) & is_undimered
            else:
                valid_mask = np.zeros(self.total_sites, dtype=bool)

            valid_mask[site] = False
            new_dimer_row[valid_mask] = self.weighted_potential_dimer_partners[site, valid_mask]

        self.dimer_weight_matrix[site, :] = new_dimer_row
        self.dimer_weight_matrix[:, site] = new_dimer_row
        self.total_dimer_weight_symmetric += (np.sum(new_dimer_row) - old_dimer_row_sum) * 2

        # recalculate tether weights efficiently
        is_vacant = self.chromatin_state == SiteState.EMPTY
        
        is_dangling_s = (self.chromatin_state == SiteState.NANOGb_SOX2f)
        is_dangling_n = (self.chromatin_state == SiteState.NANOGb_NANOGf) | (self.chromatin_state == SiteState.SOX2b_NANOGf)
        is_dangling = is_dangling_s | is_dangling_n

        # Update row for site (if site is dangling)
        if is_dangling[site]:
            self.tether_weight_matrix[site, :] = self.weighted_potential_tether_partners[site, :] * is_vacant
            self.tether_weight_matrix[site, site] = 0.0
        else:
            self.tether_weight_matrix[site, :] = 0.0
            
        # Update column for site (if site is vacant)
        if is_vacant[site]:
            self.tether_weight_matrix[:, site] = self.weighted_potential_tether_partners[:, site] * is_dangling
            self.tether_weight_matrix[site, site] = 0.0
        else:
            self.tether_weight_matrix[:, site] = 0.0

        # Recalculate global tether weights
        # Only dangling sites have positive row sums
        row_sums = np.sum(self.tether_weight_matrix, axis=1)
        
        self.tether_weight_s = row_sums * is_dangling_s
        self.tether_weight_n = row_sums * is_dangling_n

        self.total_tether_weight_s = np.sum(self.tether_weight_s)
        self.total_tether_weight_n = np.sum(self.tether_weight_n)
    def get_base_tf(self, state: int) -> int:
        """_summary_ Helper function returning the base bound tf of a given site state.

        Args:
            state (int): integer state representing chromatin site occupancy.

        Returns:
            int: 1 for SOX2, 2 for NANOG, 0 for empty.
        """
        from .constants import SiteState
        if state in (SiteState.SOX2b, SiteState.SOX2b_NANOGf, SiteState.SOX2b_NANOGb):
            return 1
        elif state in (SiteState.NANOGb, SiteState.NANOGb_SOX2f, SiteState.NANOGb_NANOGf, SiteState.NANOGb_NANOGb, SiteState.NANOGb_SOX2b):
            return 2
        return 0

    def set_site_state(self, site: int, new_state: int, new_partner: int = -1):
        """_summary_ Safely sets site state and manages all underlying index tracker arrays to maintain performance O(1).
        
        Args:
            site (int): chromatin site index.
            new_state (int): integer state to transition to.
            new_partner (int, optional): index of newly bound partner. Defaults to -1.
        """
        from .constants import SiteState
        old_state = self.chromatin_state[site]
        old_partner = self.chromatin_partner_site[site]
        if old_state == new_state and old_partner == new_partner:
            return

        self.chromatin_state[site] = new_state
        self.chromatin_partner_site[site] = new_partner

        from .model_state import _remove_from_tracker, _add_to_tracker

        # 1. Update vacant trackers
        if old_state == SiteState.EMPTY and new_state != SiteState.EMPTY:
            self.vacant_count = _remove_from_tracker(self.vacant_sites, self.vacant_idx, self.vacant_count, site)
        elif old_state != SiteState.EMPTY and new_state == SiteState.EMPTY:
            self.vacant_count = _add_to_tracker(self.vacant_sites, self.vacant_idx, self.vacant_count, site)

        # 2. Update bound trackers
        old_base_tf = self.get_base_tf(old_state)
        new_base_tf = self.get_base_tf(new_state)
        
        if old_base_tf != new_base_tf:
            if old_base_tf > 0:
                self.bound_count[old_base_tf] = _remove_from_tracker(self.bound_sites[old_base_tf], self.bound_idx[old_base_tf], self.bound_count[old_base_tf], site)
            if new_base_tf > 0:
                self.bound_count[new_base_tf] = _add_to_tracker(self.bound_sites[new_base_tf], self.bound_idx[new_base_tf], self.bound_count[new_base_tf], site)

        # 3. Update undimered trackers
        is_old_undimered = old_state in (SiteState.SOX2b, SiteState.NANOGb)
        is_new_undimered = new_state in (SiteState.SOX2b, SiteState.NANOGb)
        
        if is_old_undimered and not is_new_undimered:
            self.undimered_count[old_base_tf] = _remove_from_tracker(self.undimered_sites[old_base_tf], self.undimered_idx[old_base_tf], self.undimered_count[old_base_tf], site)
        elif not is_old_undimered and is_new_undimered:
            self.undimered_count[new_base_tf] = _add_to_tracker(self.undimered_sites[new_base_tf], self.undimered_idx[new_base_tf], self.undimered_count[new_base_tf], site)

        # 4. Update dimered trackers
        is_old_dimered = old_state in (SiteState.SOX2b_NANOGb, SiteState.NANOGb_NANOGb, SiteState.NANOGb_SOX2b)
        is_new_dimered = new_state in (SiteState.SOX2b_NANOGb, SiteState.NANOGb_NANOGb, SiteState.NANOGb_SOX2b)

        if is_old_dimered and not is_new_dimered:
            self.dimered_count = _remove_from_tracker(self.dimered_sites, self.dimered_idx, self.dimered_count, site)
        elif not is_old_dimered and is_new_dimered:
            self.dimered_count = _add_to_tracker(self.dimered_sites, self.dimered_idx, self.dimered_count, site)

        # 5. Update dangling counts
        # old dangling
        if old_state == SiteState.NANOGb_SOX2f:
            self.dangling_counts[1] -= 1
        elif old_state in (SiteState.SOX2b_NANOGf, SiteState.NANOGb_NANOGf):
            self.dangling_counts[2] -= 1
            
        # new dangling
        if new_state == SiteState.NANOGb_SOX2f:
            self.dangling_counts[1] += 1
        elif new_state in (SiteState.SOX2b_NANOGf, SiteState.NANOGb_NANOGf):
            self.dangling_counts[2] += 1

        self.update_site_weights(site)

    def is_promoter_active(self) -> bool:
        """_summary_ Checks if the designated promoter site is bound by an activating tf.

        Returns:
            bool: True if promoter is active, False otherwise.
        """
        from .constants import SiteState
        p_site = self.promoter_site
        p_state = self.chromatin_state[p_site]
        
        if p_state == SiteState.EMPTY:
            return False
            
        bound_tf_id = self.get_base_tf(p_state)
        if bound_tf_id > 0 and self.is_activator[bound_tf_id]:
            return True

        partner_site = self.chromatin_partner_site[p_site]
        if partner_site >= 0:
            partner_tf_id = self.get_base_tf(self.chromatin_state[partner_site])
            if partner_tf_id > 0 and self.is_activator[partner_tf_id]:
                return True

        return False
