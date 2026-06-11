import polars as pl
from .config_default import REACTION_NAMES, SPECIES_MAP
from .model_state import ModelState


class ModelLogger:
    """_summary_ Methods for recording reactions, and outputting DFs containing time course data for sites, residence times, and a list of all reactions that occur.

    Attributes:
        state (class): inherits ModelState class methods and attributes to record all reaction information.
    """

    def __init__(self, state: ModelState):
        self.state = state
        self.times, self.bulk_states, self.spatial_states = [], [], []
        self.partner_states = []
        self.reaction_history, self.residence_time_states = [], []

    def on_reaction_fired(
        self, current_t, reaction_index, primary_site=-1, secondary_site=-1
    ):
        reaction_name = (
            REACTION_NAMES[reaction_index]
            if reaction_index < len(REACTION_NAMES)
            else f"RXN_{reaction_index}"
        )
        self.reaction_history.append(
            (current_t, reaction_name, primary_site, secondary_site)
        )

    def on_snapshot(self, current_t, state):
        self.times.append(current_t)
        self.bulk_states.append(state.species_counts.copy())
        self.spatial_states.append(state.chromatin_state.copy())
        self.partner_states.append(state.chromatin_partner_site.copy())

    def on_site_state_changed(
        self,
        current_sim_time,
        site,
        old_species_label,
        new_species_label,
        reaction_name,
        duration,
        paired_site=-1,
        is_bound=True,
    ):
        self.residence_time_states.append(
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

    def generate_dataframes(self, final_t):
        """Outputs three dataframes containing simulation data.

        Return:
            polars dataframes df_states, df_dwell df_rxns.
        """
        from . import constants
        for i in range(self.state.total_sites):
            if (
                self.state.chromatin_state[i] != constants.SiteState.EMPTY
                and self.state.chromatin_site_bind_times[i] != -1.0
            ):

                partner_site = self.state.chromatin_partner_site[i]
                if partner_site >= 0:
                    if i > partner_site:
                        continue  # Skip duplicates for bivalent dimers
                    paired_site = partner_site
                else:
                    paired_site = -1

                species_label = self.state.get_species_label(i)

                self.residence_time_states.append(
                    [
                        final_t - self.state.chromatin_site_bind_times[i],
                        final_t,
                        i,
                        paired_site,
                        species_label,
                        "STILL_BOUND",
                        "END_OF_SIMULATION",
                        True,
                    ]
                )
        if self.bulk_states:
            import numpy as np
            bulk_matrix = np.stack(self.bulk_states)
            df_states = pl.DataFrame(
                {
                    "time": self.times,
                    **{
                        name: bulk_matrix[:, idx]
                        for name, idx in SPECIES_MAP.items()
                    },
                }
            )
        else:
            df_states = pl.DataFrame({"time": []})

        df_dwell = pl.DataFrame(
            self.residence_time_states,
            schema={
                "event_duration": pl.Float64,
                "current_sim_time": pl.Float64,
                "dwell_site": pl.Int64,
                "paired_site": pl.Int64,
                "old_species": pl.String,
                "new_species": pl.String,
                "reaction_name": pl.String,
                "is_bound": pl.Boolean,
            },
            orient="row",
        )

        df_rxns = pl.DataFrame(
            self.reaction_history,
            schema={
                "time": pl.Float64,
                "reaction_type": pl.String,
                "primary_site": pl.Int64,
                "secondary_site": pl.Int64,
            },
            orient="row",
        )
        return df_states, df_dwell, df_rxns

