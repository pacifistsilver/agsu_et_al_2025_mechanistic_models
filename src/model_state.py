import numpy as np
from typing import List
from dataclasses import dataclass
from .config_default import SPECIES_NAMES, REACTION_NAMES
from .model_utils import TranscriptionFactor


class ModelState:
    """Tracks the state of the simulation and includes commonly called methods.

    Attributes:
        rate_constants (dict): Reaction kinetic parameters (k_prod_s, k_bind_s, etc.).
        initial_species_states (dict): Initial molecule counts for the simulation.
        total_binding_sites (int): Total number of binding sites in the chromatin lattice.
        record_interval (float): Simulation time interval to record reactions.
        promoter_site (int): The index of the primary promoter site.
    """

    def __init__(
        self,
        tfs: List[TranscriptionFactor],
        total_binding_sites: int,
        initial_species_states: dict,
        promoter_site: int = None,
    ):
        self.tfs = {tf.id: tf for tf in tfs}
        self.tfs_by_dangling = {tf.dangling_id: tf for tf in tfs}
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

        # chromatin tracking arrays
        self.chromatin_lattice = np.zeros(self.total_sites, dtype=np.int8)
        self.chromatin_site_is_vacant = np.ones(self.total_sites, dtype=bool)
        self.chromatin_all_undimered_monomers = np.zeros(self.total_sites, dtype=bool)
        self.chromatin_site_bind_times = np.full(self.total_sites, -1.0)
        self.vacant_chromatin_sites = set(range(self.total_sites))

        # -1: monomer, -2: SOX2f:dimer, -3: NANOGf:dimer, >= 0: both bound
        self.chromatin_partner_state = np.full(self.total_sites, -1, dtype=np.int32)

        self.bound_sites = {tf.id: set() for tf in tfs}
        self.undimered_sites = {tf.id: set() for tf in tfs}
        self.dangling_counts = {tf.id: 0 for tf in tfs}

        self.dimered_dimer_sites = set()
        self.vacant_chromatin_sites = set(range(self.total_sites))

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
        self.total_tether_weight = 0.0

        # increment weights
        self.dimer_weight_matrix = np.zeros_like(self.weighted_potential_dimer_partners)
        self.bivalent_transition_matrix = np.zeros_like(self.weighted_potential_dimer_partners)

    def get_species_label(self, site):
        if site == -1 or self.chromatin_site_is_vacant[site]:
            return "EMPTY"

        tf_id = self.chromatin_lattice[site]
        if tf_id not in self.tfs:
            return "UNKNOWN"

        base_str = f"{self.tfs[tf_id].name}b"
        partner = self.chromatin_partner_state[site]

        if partner == -1:
            return base_str

        elif partner in self.tfs_by_dangling:
            dangling_tf = self.tfs_by_dangling[partner]
            return f"{base_str}:{dangling_tf.name}f"

        elif partner >= 0:
            partner_tf_id = self.chromatin_lattice[partner]
            partner_name = self.tfs[partner_tf_id].name

            names = sorted([self.tfs[tf_id].name, partner_name])
            return f"{names[0]}b:{names[1]}b"

        return "UNKNOWN"

    def update_site_weights(self, site):
        """Update spatial weight matrices to decide dimerisation partners."""
        # dimerisation logic
        old_dimer_row_sum = np.sum(self.dimer_weight_matrix[site, :])
        new_dimer_row = np.zeros(self.total_sites)

        if self.chromatin_all_undimered_monomers[site]:
            tf_type = self.chromatin_lattice[site]

            # identify valid partners based on TF type
            if tf_type == 1:
                valid_mask = (
                    self.chromatin_lattice == 2
                ) & self.chromatin_all_undimered_monomers
            elif tf_type == 2:
                valid_mask = (
                    (self.chromatin_lattice == 1) | (self.chromatin_lattice == 2)
                ) & self.chromatin_all_undimered_monomers
            else:
                valid_mask = np.zeros(self.total_sites, dtype=bool)

            valid_mask[site] = False
            new_dimer_row[valid_mask] = self.weighted_potential_dimer_partners[
                site, valid_mask
            ]

        # dimer updates
        self.dimer_weight_matrix[site, :] = new_dimer_row
        self.dimer_weight_matrix[:, site] = new_dimer_row
        self.total_dimer_weight_symmetric += (
            np.sum(new_dimer_row) - old_dimer_row_sum
        ) * 2

        # update tethering row
        new_tether_row = np.zeros(self.total_sites)

        if self.chromatin_partner_state[site] in (-2, -3):
            vacant_mask = self.chromatin_site_is_vacant
            new_tether_row[vacant_mask] = self.weighted_potential_dimer_partners[
                site, vacant_mask
            ]
            new_tether_row[site] = 0.0

        # update tethering col
        old_tether_col_sum = np.sum(self.bivalent_transition_matrix[:, site])
        new_tether_col = np.zeros(self.total_sites)

        if self.chromatin_site_is_vacant[site]:
            tether_mask = self.chromatin_partner_state < -1
            new_tether_col[tether_mask] = self.weighted_potential_dimer_partners[
                tether_mask, site
            ]
            new_tether_col[site] = 0.0

        # bivalent updates
        self.bivalent_transition_matrix[site, :] = new_tether_row
        self.bivalent_transition_matrix[:, site] = new_tether_col

        # global tethering weights update
        self.total_tether_weight += np.sum(new_tether_col) - old_tether_col_sum
        self.total_tether_weight_s = np.sum(
            self.bivalent_transition_matrix[self.chromatin_partner_state == -2, :]
        )
        self.total_tether_weight_n = np.sum(
            self.bivalent_transition_matrix[self.chromatin_partner_state == -3, :]
        )

    def set_site_state(
        self, site, is_vacant=None, tf_id=None, is_undimered=None, partner_state=None
    ):
        """Update status of a site on the chromatin array when a binding/unbinding/dimerisation reaction occurs.

        Args:
            site (int): index referring to position in chromatin array
            is_vacant (bool, optional): Update site occupancy status. Defaults to None.
            tf_type (int, optional): Form of TF. i.e. SOX2, NANOG. Defaults to None.
            is_undimered (bool, optional): Track dimer status. Defaults to None.
            dangling_tf_type (int, optional): For a dimerised molecule, this refers to the TF that is not bound to any site. i.e. it is bound to this TF but not bound to anything else. Defaults to None.
            dimer_partner (int, optional): index referring to the dimerised TF partner on the chromatin array. Defaults to None.
        """
        if tf_id is not None:
            old_tf_id = self.chromatin_lattice[site]
            if old_tf_id in self.bound_sites:
                self.bound_sites[old_tf_id].discard(site)

            self.chromatin_lattice[site] = tf_id

            if tf_id in self.bound_sites:
                self.bound_sites[tf_id].add(site)

        if is_vacant is not None:
            self.chromatin_site_is_vacant[site] = is_vacant
            if is_vacant:
                self.vacant_chromatin_sites.add(site)
            else:
                self.vacant_chromatin_sites.discard(site)

        if is_undimered is not None:
            self.chromatin_all_undimered_monomers[site] = is_undimered
            current_tf_id = self.chromatin_lattice[site]

            if current_tf_id in self.undimered_sites:
                if is_undimered:
                    self.undimered_sites[current_tf_id].add(site)
                else:
                    self.undimered_sites[current_tf_id].discard(site)

        if partner_state is not None:
            old_state = self.chromatin_partner_state[site]

            # Remove old state
            if old_state in self.tfs_by_dangling:
                dangling_tf = self.tfs_by_dangling[old_state]
                self.dangling_counts[dangling_tf.id] -= 1
            elif old_state >= 0:
                self.dimered_dimer_sites.discard(site)

            # Apply new state
            self.chromatin_partner_state[site] = partner_state

            if partner_state in self.tfs_by_dangling:
                dangling_tf = self.tfs_by_dangling[partner_state]
                self.dangling_counts[dangling_tf.id] += 1
            elif partner_state >= 0:
                self.dimered_dimer_sites.add(site)

        self.update_site_weights(site)

    def is_promoter_active(self) -> bool:
        """
        Evaluate if a TF is present at the promoter site,
        either directly bound or bridged via a dimer.
        """
        p_site = self.promoter_site

        # 1. If it's empty, it's definitely OFF
        if self.chromatin_site_is_vacant[p_site]:
            return False

        # 2. Check the directly bound molecule
        bound_tf_id = self.chromatin_lattice[p_site]
        if self.tfs[bound_tf_id].is_activator:
            return True

        # 3. Check the tethered partner (if it's a bivalent dimer)
        partner_site = self.chromatin_partner_state[p_site]
        if partner_site >= 0:
            partner_tf_id = self.chromatin_lattice[partner_site]
            if self.tfs[partner_tf_id].is_activator:
                return True

        return False
