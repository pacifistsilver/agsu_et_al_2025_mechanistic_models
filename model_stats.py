import os
import polars as pl
import pandas as pd
import numpy as np
from types import SimpleNamespace
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class Statistics():
    def __init__(self, lf_states: pl.LazyFrame, lf_dwell_times: pl.LazyFrame, lf_reactions: pl.LazyFrame, total_sim_time: int, total_binding_sites: int):
        self.total_sim_time = total_sim_time        
        self.df_states_log = lf_states
        self.dwell_times_log = lf_dwell_times
        self.df_reaction_log = lf_reactions
        self.total_binding_sites = total_binding_sites

    @classmethod
    def from_parquet_dir(cls, param_set_dir: str, total_sim_time: int, total_binding_sites: int = 10):
        """Instantiates the Statistics class by lazily scanning an entire directory of Parquet runs."""
        lf_states = pl.scan_parquet(os.path.join(param_set_dir, "states", "*.parquet"))
        lf_dwell = pl.scan_parquet(os.path.join(param_set_dir, "dwell_times", "*.parquet"))
        lf_rxns = pl.scan_parquet(os.path.join(param_set_dir, "rxns", "*.parquet"))
        
        return cls(lf_states, lf_dwell, lf_rxns, total_sim_time, total_binding_sites)

    def extract_per_run_features(self, param_set_id, rates, init_state, burn_in_fraction=0.2):
        mrna_stats = self._calc_mrna_features(burn_in_fraction)
        unbound_stats = self._calc_unbound_features()
        dwell_stats = self._calc_dwell_features()
        
        # Now uses the True Molecule Lifespan (Lagrangian Tracking)
        mfpt_stats = self._build_all_trajectories()

        # Join everything into one master table
        master = mrna_stats.join(unbound_stats, on="run_id", how="left") \
                          .join(dwell_stats, on="run_id", how="left") \
                          .join(mfpt_stats, on="run_id", how="left")

        # Inject metadata
        master = master.with_columns(pl.lit(param_set_id).alias("param_set_id"))
        for k, v in rates.items(): master = master.with_columns(pl.lit(v).alias(f"param_{k}"))
        for k, v in init_state.items(): master = master.with_columns(pl.lit(v).alias(f"init_{k}"))
            
        return master.collect().fill_null(0.0)

    def _calc_mrna_features(self, burn_in_fraction):
        burn_in = self.total_sim_time * burn_in_fraction
        return self.df_states_log.filter(pl.col("time") >= burn_in).group_by("run_id").agg(
            pl.col("mRNA").mean().alias("mean_mrna"),
            (pl.col("mRNA").var() / pl.col("mRNA").mean()).alias("mrna_fano"),
            (pl.col("mRNA").std() / pl.col("mRNA").mean()).alias("mrna_coefficient_of_variation")
        ).fill_nan(0.0)

    def _calc_unbound_features(self):
        return self.dwell_times_log.filter(pl.col("old_species") == "EMPTY").group_by("run_id").agg(
            pl.col("event_duration").mean().alias("mean_site_unbound_time")
        )

    def _calc_dwell_features(self):
        dwells_df = self.dwell_times_log.filter(pl.col("old_species") != "EMPTY").group_by("run_id", "old_species").agg(
            pl.col("event_duration").mean()
        ).collect()
        
        if dwells_df.is_empty(): return pl.LazyFrame(schema={"run_id": pl.Int32})
        
        pivot = dwells_df.pivot(index="run_id", on="old_species", values="event_duration", aggregate_function="first")
        return pivot.rename({c: f"mean_duration_{c}" for c in pivot.columns if c != "run_id"}).lazy()

    def _build_all_trajectories(self):
        """Converts the dwell log to pandas to chronologically track sliding molecules."""
        df = self.dwell_times_log.collect().to_pandas()
        
        # Ensure it is strictly chronological so we can step through time perfectly
        df = df.sort_values(by=["run_id", "current_sim_time"])
        
        all_completed_trajectories = []
        for run_id, run_data in df.groupby("run_id"):
            run_trajectories = self._track_run_trajectories(run_id, run_data)
            all_completed_trajectories.extend(run_trajectories)
            
        if not all_completed_trajectories:
            return pl.LazyFrame(schema={"run_id": pl.Int32})
            
        # Format the output exactly like you wanted for your master table
        mfpt_df = pl.DataFrame(all_completed_trajectories).group_by(["run_id", "starting_species"]).agg(
            pl.col("bound_lifespan_s").mean()
        )
        
        pivot = mfpt_df.pivot(index="run_id", on="starting_species", values="bound_lifespan_s", aggregate_function="first")
        rename_map = {c: f"MFPT_{c}" for c in pivot.columns if c != "run_id"}
        
        return pivot.rename(rename_map).lazy()

    def _track_run_trajectories(self, run_id, run_data):
        """Scans a single run to track individual molecules across sites using paired_site links."""
        active_lattice = {} # Maps site index to a shared Molecule dictionary
        completed = []
        molecule_counter = 0

        for row in run_data.itertuples():
            site = getattr(row, "dwell_site", -1)
            paired = getattr(row, "paired_site", -1)
            duration = getattr(row, "event_duration", 0.0)
            old_sp = getattr(row, "old_species", "UNKNOWN")
            new_sp = getattr(row, "new_species", "UNKNOWN")
            
            # --- 1. UPDATE CLOCK ---
            if site in active_lattice:
                # Add duration to this molecule's lifespan
                active_lattice[site]["bound_lifespan_s"] += duration

            # --- 2. ROUTE THE EVENT ---
            
            # CASE A: Brand New Arrival from Bulk (EMPTY -> Bound, no pair)
            if old_sp == "EMPTY" and paired == -1 and new_sp != "EMPTY":
                molecule_counter += 1
                active_lattice[site] = {
                    "molecule_id": molecule_counter,
                    "starting_species": new_sp,
                    "bound_lifespan_s": 0.0
                }
                
            # CASE B: The "Step" Forward (Tethered Binding)
            # A vacant site suddenly becomes bound and is paired to an existing site.
            elif old_sp == "EMPTY" and paired != -1 and new_sp != "EMPTY":
                if paired in active_lattice:
                    # Pass the EXACT SAME dictionary reference to the new foot's site.
                    # This links them so they share the exact same lifespan clock and ID.
                    active_lattice[site] = active_lattice[paired]

            # CASE C: Complete Unbinding to Bulk (Bound -> EMPTY, no pair)
            elif old_sp != "EMPTY" and new_sp == "EMPTY" and paired == -1:
                if site in active_lattice:
                    dying_molecule = active_lattice.pop(site)
                    completed.append({
                        "run_id": run_id,
                        "molecule_id": dying_molecule["molecule_id"],
                        "starting_species": dying_molecule["starting_species"],
                        "bound_lifespan_s": dying_molecule["bound_lifespan_s"]
                    })
                    
            # CASE D: One foot lifts up (The completion of the "Step")
            # Site goes EMPTY, but it was paired, meaning the molecule survives on the other site.
            elif old_sp != "EMPTY" and new_sp == "EMPTY" and paired != -1:
                if site in active_lattice:
                    # Just remove this foot from the lattice tracking.
                    # Because the dictionary is shared by reference, the other site 
                    # perfectly retains the molecule's ID and clock!
                    active_lattice.pop(site)

        return completed
        
        