"""StochPy-based Gillespie simulation of gene expression given spatiotemporal dynamics.
"""
import numpy as np
import polars as pl
import stochpy
import os
import sys

from .model_utils import TranscriptionFactor
from .config_default import SPECIES_NAMES, REACTION_NAMES, SPECIES_MAP
from .constants import STATE_STRINGS, SiteState
from .psc_generator import generate_psc


class ModelCall:
    def __init__(
        self,
        tfs,
        model_param: dict,
        model_var: dict,
        model_binding_sites: int,
        sim_max_time,
        record_interval: float = 1.0,
    ):
        self.rates = model_param
        self.max_time = sim_max_time
        self.num_sites = model_binding_sites
        self.record_interval = record_interval
        self.model_var = model_var

        # Ensure the .psc file exists
        psc_filename = "spatial_chromatin.psc"
        psc_path = os.path.join(os.getcwd(), psc_filename)
        
        generate_psc(self.num_sites, filename=psc_path)

        # Initialize StochPy SSA
        self.smod = stochpy.SSA()
        # Suppress extensive StochPy output
        # stochpy.core2.py uses standard print, but stochpy has Quiet()
        try:
            stochpy.Quiet()
        except:
            pass

        self.smod.Model(model_file=psc_filename, dir=os.getcwd())

    def run_trajectory(self):
        # 1. Apply rate parameters
        for param_name, param_val in self.rates.items():
            self.smod.ChangeParameter(param_name, param_val)

        # 2. Run stochastic simulation
        # Use Direct method which is robust for many reactions
        self.smod.DoStochSim(method='Direct', mode='time', end=self.max_time, trajectories=1)

        # 3. Process output data
        # We don't need GetTrajectoryData(1) as StochPy stores the last run in data_stochsim
        times = self.smod.data_stochsim.time
        species_matrix = self.smod.data_stochsim.species
        labels = self.smod.data_stochsim.species_labels
        
        # Build dictionary for fast column access
        col_idx = {label: i for i, label in enumerate(labels)}

        # Helper to safely sum columns
        def sum_cols(prefix, suffixes, labels_list):
            total = np.zeros(len(times))
            for suff in suffixes:
                name = prefix + suff
                if name in col_idx:
                    total += species_matrix[:, col_idx[name]]
            return total

        # --- Reconstruct df_states ---
        df_states_dict = {"time": times}
        
        S_free = species_matrix[:, col_idx["S_free"]] if "S_free" in col_idx else np.zeros(len(times))
        N_free = species_matrix[:, col_idx["N_free"]] if "N_free" in col_idx else np.zeros(len(times))
        SN_free = species_matrix[:, col_idx["SN_free"]] if "SN_free" in col_idx else np.zeros(len(times))
        NN_free = species_matrix[:, col_idx["NN_free"]] if "NN_free" in col_idx else np.zeros(len(times))
        mRNA = species_matrix[:, col_idx["mRNA"]] if "mRNA" in col_idx else np.zeros(len(times))

        S_bound_total = np.zeros(len(times))
        N_bound_total = np.zeros(len(times))
        SN_single_bound_total = np.zeros(len(times))
        NN_single_bound_total = np.zeros(len(times))
        
        for i in range(self.num_sites):
            if f"S_{i}_S" in col_idx: S_bound_total += species_matrix[:, col_idx[f"S_{i}_S"]]
            if f"S_{i}_N" in col_idx: N_bound_total += species_matrix[:, col_idx[f"S_{i}_N"]]
            if f"S_{i}_SNf" in col_idx: SN_single_bound_total += species_matrix[:, col_idx[f"S_{i}_SNf"]]
            if f"S_{i}_NSf" in col_idx: SN_single_bound_total += species_matrix[:, col_idx[f"S_{i}_NSf"]]
            if f"S_{i}_NNf" in col_idx: NN_single_bound_total += species_matrix[:, col_idx[f"S_{i}_NNf"]]

        SN_dimer_bound = np.zeros(len(times))
        NN_dimer_bound = np.zeros(len(times))
        
        for i in range(self.num_sites):
            for j in range(i+1, self.num_sites):
                if f"T_{i}_{j}_SN" in col_idx: SN_dimer_bound += species_matrix[:, col_idx[f"T_{i}_{j}_SN"]]
                if f"T_{i}_{j}_NS" in col_idx: SN_dimer_bound += species_matrix[:, col_idx[f"T_{i}_{j}_NS"]]
                if f"T_{i}_{j}_NN" in col_idx: NN_dimer_bound += species_matrix[:, col_idx[f"T_{i}_{j}_NN"]]

        df_states_dict["sox2_monomer_free"] = S_free
        df_states_dict["nanog_monomer_free"] = N_free
        df_states_dict["nanog_sox2_dimer_free"] = SN_free
        df_states_dict["nanog_nanog_dimer_free"] = NN_free
        df_states_dict["sox2_monomer_bound"] = S_bound_total
        df_states_dict["nanog_monomer_bound"] = N_bound_total
        df_states_dict["nanog_sox2_dimer_bound"] = SN_dimer_bound
        df_states_dict["nanog_nanog_dimer_bound"] = NN_dimer_bound
        df_states_dict["nanog_sox2_dimer_single_bound"] = SN_single_bound_total
        df_states_dict["nanog_nanog_dimer_single_bound"] = NN_single_bound_total
        df_states_dict["mRNA"] = mRNA

        # Ensure correct column order
        final_state_cols = {"time": times}
        for name in SPECIES_NAMES:
            final_state_cols[name] = df_states_dict.get(name, np.zeros(len(times)))
        df_states = pl.DataFrame(final_state_cols)

        # --- Reconstruct df_dwell ---
        # State inference
        site_states = np.full((len(times), self.num_sites), "EMPTY", dtype=object)
        site_partners = np.full((len(times), self.num_sites), -1, dtype=int)

        # Monomers and dangling
        for i in range(self.num_sites):
            idx_s = col_idx.get(f"S_{i}_S", -1)
            idx_n = col_idx.get(f"S_{i}_N", -1)
            idx_snf = col_idx.get(f"S_{i}_SNf", -1)
            idx_nsf = col_idx.get(f"S_{i}_NSf", -1)
            idx_nnf = col_idx.get(f"S_{i}_NNf", -1)
            
            if idx_s != -1: site_states[species_matrix[:, idx_s] == 1, i] = "SOX2b"
            if idx_n != -1: site_states[species_matrix[:, idx_n] == 1, i] = "NANOGb"
            if idx_snf != -1: site_states[species_matrix[:, idx_snf] == 1, i] = "SOX2b:NANOGf"
            if idx_nsf != -1: site_states[species_matrix[:, idx_nsf] == 1, i] = "NANOGb:SOX2f"
            if idx_nnf != -1: site_states[species_matrix[:, idx_nnf] == 1, i] = "NANOGb:NANOGf"

        # Tethers
        for i in range(self.num_sites):
            for j in range(i+1, self.num_sites):
                idx_sn = col_idx.get(f"T_{i}_{j}_SN", -1)
                idx_ns = col_idx.get(f"T_{i}_{j}_NS", -1)
                idx_nn = col_idx.get(f"T_{i}_{j}_NN", -1)
                
                if idx_sn != -1:
                    mask = species_matrix[:, idx_sn] == 1
                    site_states[mask, i] = "SOX2b:NANOGb"
                    site_partners[mask, i] = j
                    site_states[mask, j] = "NANOGb:SOX2b"
                    site_partners[mask, j] = i
                    
                if idx_ns != -1:
                    mask = species_matrix[:, idx_ns] == 1
                    site_states[mask, i] = "NANOGb:SOX2b"
                    site_partners[mask, i] = j
                    site_states[mask, j] = "SOX2b:NANOGb"
                    site_partners[mask, j] = i
                    
                if idx_nn != -1:
                    mask = species_matrix[:, idx_nn] == 1
                    site_states[mask, i] = "NANOGb:NANOGb"
                    site_partners[mask, i] = j
                    site_states[mask, j] = "NANOGb:NANOGb"
                    site_partners[mask, j] = i

        residence_events = []
        # Calculate dwell events
        for i in range(self.num_sites):
            prev_state = site_states[0, i]
            prev_partner = site_partners[0, i]
            start_t = times[0]
            
            for t_idx in range(1, len(times)):
                curr_state = site_states[t_idx, i]
                curr_partner = site_partners[t_idx, i]
                
                if curr_state != prev_state or curr_partner != prev_partner:
                    if prev_state != "EMPTY":
                        duration = times[t_idx] - start_t
                        residence_events.append([
                            duration,
                            times[t_idx],
                            i,
                            prev_partner,
                            prev_state,
                            curr_state,
                            "STOCHPY_RXN",  # Exact rxn name not tracked explicitly
                            False if curr_state == "EMPTY" else True
                        ])
                    start_t = times[t_idx]
                    prev_state = curr_state
                    prev_partner = curr_partner
                    
            # Handle end of simulation
            if prev_state != "EMPTY":
                residence_events.append([
                    times[-1] - start_t,
                    times[-1],
                    i,
                    prev_partner,
                    prev_state,
                    "STILL_BOUND",
                    "END_OF_SIMULATION",
                    True
                ])

        df_dwell = pl.DataFrame(
            residence_events,
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

        # Provide a dummy df_rxns since StochPy doesn't trivially export the exact reaction history strings natively
        df_rxns = pl.DataFrame(
            [],
            schema={
                "time": pl.Float64,
                "reaction_type": pl.String,
                "primary_site": pl.Int64,
                "secondary_site": pl.Int64,
            },
            orient="row",
        )

        return df_states, df_dwell, df_rxns
