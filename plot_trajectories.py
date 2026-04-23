import itertools

import numpy as np
import polars as pl
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from main import PartitionFunction, ModelCall

class ModelPlot:
    """Retrieve model outputs and partition function outputs and handle plotting of said data.
    
    Inherits class attributes from "model" scope. i.e. takes the attributes from ModelCall
    
    Attributes:
        spatial_states: Lattice structure of chromatin. 
        n_binding_sites: The total number of binding sites.
        sox2_simulation_variables: Starting population of free/bound/mRNA
        bulk_states: Tracks the amount of free/bound/mrna
        times: Time interval to record data. 
        dwell_time_df: Dataframe containing all dwell_times in a trajectory. i.e. time spent in sites.
        simulation_reaction_history_df: Dataframe containing reactions that occur in a trajectory.

    """
    def __init__(self, model):
        self.spatial_states = model.spatial_states 
        self.n_binding_sites = model.n_binding_sites
        self.sox2_simulation_variables = model.sox2_simulation_variables
        self.bulk_states = model.bulk_states
        self.times = model.times
        self.dwell_time_df = model.sim_site_dwell_times_df
        self.simulation_reaction_history_df = model.sim_reaction_history_df

    # Retrieve model data
    @staticmethod
    def generate_thermodynamic_expectation_rho(S_vals, N_vals, K1_vals, c=1.0, alpha=1.0):
        """Retrieve expected transcription rate from partition function given parameters.
        
        Args:
            S_vals:
                Array of bulk TF values to iterate over.
            N_vals:
                Array of total binding sites in chromatin.
            K1_vals:
                Array of binding affinities K1.
            c:
                Dimensionless binding weight represented as a float variable.
            alpha:
                Float variable representing transcription rate.
        
        Returns
            Dataframe instance. 
            For example:
            {
                "S": 1,
                "N": 1,
                "K": 1,
                "thermo_maxmial_rate": 1.0,
                ...
            }
        Raises:
            None
        """
        results = []
        for S, N, K1 in itertools.product(S_vals, N_vals, K1_vals):
            rate_maximal = PartitionFunction.return_maximal_rho(int(N), K1, S, c, alpha)
            rate_linear = PartitionFunction.return_nonmaximal_rho(int(N), K1, S, c, alpha, mode="linear")
            rate_saturating = PartitionFunction.return_nonmaximal_rho(int(N), K1, S, c, alpha,K_alpha=1, mode="saturating")
            results.append({"S": S, "N": int(N), "K1": K1, "thermo_maximal_rate": rate_maximal, "thermo_linear_rate": rate_linear, "thermo_saturating_rate": rate_saturating})
            
        return pd.DataFrame(results)

    def get_effective_transcription_rate(self) -> float:
        if not self.bulk_states:
            return 0.0
            
        k_prod_m = self.sox2_simulation_variables[2] 
        
        # Extract the number of bound SOX2 molecules at each time step (index 1 of bulk_states)
        bound_counts = [state[1] for state in self.bulk_states]
        average_bound = sum(bound_counts) / len(bound_counts)
        
        return average_bound * k_prod_m
    
    def return_mRNA_data(self) -> float:
        if not self.bulk_states:
            return 0.0
        mrna_counts = [state[2] for state in self.bulk_states]
        return (sum(mrna_counts), len(mrna_counts))

    def get_average_dwell_time(self) -> float:
        if self.dwell_time_df.is_empty():
            return 0.0
        return self.dwell_time_df.select(pl.col("dwell_time").mean()).item()

    # Save model outputs to file
    def save_residence_time_log(self, filename: str = "time-log.txt"):
        residence_time_states = (
            self.dwell_time_df.group_by("dwell_site")
            .agg(pl.col("dwell_time").sum())
            .sort(by="dwell_time", descending=True)
        )
        np.savetxt(filename, residence_time_states, fmt="%g", header="dwell_site/dwell_time", delimiter=", ")
        
    def save_reaction_history_log(self, filename: str = "reaction_history.csv"):
        # Cast and save without permanently mutating the instance's dataframe
        df_out = self.simulation_reaction_history_df.cast({"site_target": pl.Int64, "site_paired_with": pl.Int64})
        df_out.write_csv(filename)
    
    # plot trajectories
    def plot_nucleosome_occupancy_history(self, filename: str = "nuc_history.png"):
        sns.set_theme(style="ticks")
        custom_cmap = ListedColormap(["#e2e8f0", "#3b82f6"])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.spatial_states, cmap=custom_cmap, cbar=False, yticklabels="auto")
        
        plt.title("Sox2 Nucleosome Occupancy Over Time", pad=15)
        plt.xlabel(f"Nucleosome Index (0 to {self.n_binding_sites - 1})")
        plt.ylabel("Time (Simulation Steps)")
        
        legend_elements = [Patch(facecolor="#e2e8f0", label="Unbound"), Patch(facecolor="#3b82f6", label="Sox2 Bound")]
        plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.25, 1))
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def plot_trajectory_and_noise(self, burn_in_fraction: float = 0.2) -> dict:
        """Plots key variables over time from the initialised trajectory, and calculates the Fano factor and CV of the transcription rate.
        
        
        """
        if not self.bulk_states:
            print("No trajectory data found. Ensure the model has been run before plotting.")
            return {"fano": 0.0, "cv": 0.0, "mean_rate": 0.0}

        times_array = np.array(self.times)
        sox2_free = np.array([state[0] for state in self.bulk_states])
        sox2_bound = np.array([state[1] for state in self.bulk_states])
        mrna = np.array([state[2] for state in self.bulk_states])
        
        k_prod_m = self.sox2_simulation_variables[2]
        production_rate = sox2_bound * k_prod_m
        
        burn_in_idx = int(len(times_array) * burn_in_fraction)
        steady_state_rate = production_rate[burn_in_idx:]
        
        if len(steady_state_rate) == 0:
            return {"fano": 0.0, "cv": 0.0, "mean_rate": 0.0}
            
        mean_rate = np.mean(steady_state_rate)
        var_rate = np.var(steady_state_rate)
        
        fano = var_rate / mean_rate if mean_rate > 0 else 0.0
        cv = np.sqrt(var_rate) / mean_rate if mean_rate > 0 else 0.0
        
        # 4. Plotting
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(
            f'Trajectory Dynamics\nProduction Rate Noise - Fano: {fano:.3f} | CV: {cv:.3f}', 
            fontsize=14, y=0.95
        )
        
        # mRNA
        axes[0].step(times_array, mrna, where='post', color='#9b59b6', linewidth=1.5)
        axes[0].set_ylabel('mRNA Count')
        axes[0].grid(True, alpha=0.3)
        
        # Free SOX2
        axes[1].step(times_array, sox2_free, where='post', color='#3498db', linewidth=1.5)
        axes[1].set_ylabel('Free SOX2')
        axes[1].grid(True, alpha=0.3)
        
        # Bound SOX2
        axes[2].step(times_array, sox2_bound, where='post', color='#e67e22', linewidth=1.5)
        axes[2].set_ylabel('Bound SOX2')
        axes[2].grid(True, alpha=0.3)
        
        # mRNA Production Rate
        axes[3].step(times_array, production_rate, where='post', color='#e74c3c', linewidth=1.5)
        axes[3].set_ylabel('Prod. Rate\n(mRNA / time)')
        axes[3].set_xlabel('Time (Simulation Steps)')
        axes[3].grid(True, alpha=0.3)
        
        # Cleanup spines
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
        return {"fano": fano, "cv": cv, "mean_rate": mean_rate}
    
def run_1d_simulation_slice(args):
    """Runs a single stochastic simulation to extract mean AND standard deviation."""
    S, N, K1 = args
    
    model_param = {"sox2_free": int(S), "sox2_bound": 0, "mrna_count": 0}
    model_var = {
        "k_prod_s": 0.0, "k_deg_s": 0.0, 
        "k_prod_m": 1.0, "k_deg_m": 0.1, 
        "k_bind": K1, "k_unbind": 1.0, 
        "k_hop": 1.0, "extra": 1.0
    }
    
    sim_max_time = 100  
    model = ModelCall(
        model_param=model_param, 
        model_var=model_var, 
        model_binding_sites=int(N), 
        sim_max_time=sim_max_time, 
        record_interval=1.0, 
        track_history=False
    )
    
    param_df, _, _ = model.run_trajectory()
    burn_in_cutoff = sim_max_time * 0.25
    steady_state_df = param_df.filter(pl.col("time") > burn_in_cutoff)
    if len(steady_state_df) > 0:
        bound_array = steady_state_df["sox2_bound"].to_numpy()
        
        # Rate = bound SOX2 * k_prod_m
        sim_mean = np.mean(bound_array) * model_var["k_prod_m"]
        sim_std = np.std(bound_array) * model_var["k_prod_m"]
    else:
        sim_mean, sim_std = 0.0, 0.0
        
    return {"S": S, "N": int(N), "K1": K1, "sim_mean": sim_mean, "sim_std": sim_std}

def plot_1d_cross_section_grid_tufte():
    # 1. Define the Grid Parameters
    target_Ns = [1, 5, 10]               
    target_K1s = [0.1, 0.5, 1.0]         
    S_vals = np.linspace(1.0, 100.0, 30) 
    tasks = [(S, N, K1) for N in target_Ns for K1 in target_K1s for S in S_vals]
                
    print(f"Running {len(tasks)} stochastic simulations for the 3x3 grid...")
    sim_results = []
    
    with ProcessPoolExecutor() as executor:
        for result in executor.map(run_1d_simulation_slice, tasks):
            sim_results.append(result)
            
    sim_df = pd.DataFrame(sim_results)
    
    print("Calculating theoretical thermodynamic rates...")
    thermo_df = ModelPlot.generate_thermodynamic_expectation_rho(S_vals, target_Ns, target_K1s)
    merged_df = pd.merge(sim_df, thermo_df, on=["S", "N", "K1"])
    
    plt.rcParams['font.family'] = 'sans-serif'
    fig, axes = plt.subplots(nrows=len(target_K1s), ncols=len(target_Ns), figsize=(12, 10), sharex=True, sharey='row')
    fig.suptitle('Rate of Transcription vs. Free SOX2', fontsize=16, fontweight='light', y=0.98)
    
    for row_idx, K1 in enumerate(target_K1s):
        for col_idx, N in enumerate(target_Ns):
            ax = axes[row_idx, col_idx]
            subset = merged_df[(np.isclose(merged_df['K1'], K1)) & (merged_df['N'] == N)]
            lower_error = np.minimum(subset["sim_mean"], subset["sim_std"])
            upper_error = subset["sim_std"]
            asymmetric_error = [lower_error, upper_error]
            
            ax.errorbar(subset["S"], subset["sim_mean"], yerr=asymmetric_error, 
                         fmt='o', color='#2b6bb0', markersize=4, capsize=0, elinewidth=1, alpha=0.8,
                         label="Simulation (Mean ± 1 SD)" if (row_idx==0 and col_idx==0) else "")
            
            ax.plot(subset["S"], subset["thermo_linear_rate"], 
                     label="Rho Expected (Linear)" if (row_idx==0 and col_idx==0) else "", 
                     color='black', linewidth=1.5, zorder=2)
            ax.plot(subset["S"], subset["thermo_saturating_rate"], 
                     label="Rho Expected (Saturating)" if (row_idx==0 and col_idx==0) else "", 
                     color='#888888', linestyle='--', linewidth=1.2, zorder=1)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cccccc')
            ax.spines['bottom'].set_color('#cccccc')
            ax.tick_params(axis='both', colors='#555555', length=4, width=0.5)
            ax.set_title(f'Sites = {N} | Ka = {K1} | Kd = 1.0', fontsize=11, color='#333333', loc='left')
            ax.yaxis.grid(True, linestyle='-', color='#eeeeee', linewidth=0.7)
            ax.xaxis.grid(False)
            if row_idx == len(target_K1s) - 1:
                ax.set_xlabel('Bulk SOX2', fontsize=11, color='#333333', labelpad=8)
            if col_idx == 0:
                ax.set_ylabel('k1', fontsize=11, color='#333333', labelpad=8)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace=0.15, hspace=0.3) 
    plt.show()
    
if __name__ == "__main__":
    plot_1d_cross_section_grid_tufte()