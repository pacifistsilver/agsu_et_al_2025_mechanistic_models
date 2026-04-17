import numpy as np
import seaborn as sns
import polars as pl
import matplotlib.pyplot as plt
import math
import pandas as pd
import concurrent.futures
import os
import itertools
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

class ThermodynamicsCalculator:
    """Handles the statistical thermodynamic formulas for the Sox2 cluster model."""
    
    @staticmethod
    def calculate_Zn(n: int, N: int, K1: float, S: float, c: float) -> float:
        """Calculates the statistical weight of exactly n bound molecules."""
        binding_term = math.comb(N, n) * ((K1 * S)**n)
        
        interaction_sum = sum(
            math.comb(N - n, l) * math.comb(n, l) * math.factorial(l) * (c**l)
            for l in range(min(n, N - n) + 1)
        )
        return binding_term * interaction_sum

    @staticmethod
    def calculate_total_Z(N: int, K1: float, S: float, c: float) -> float:
        """Calculates the total partition function by summing all Zn."""
        return sum(ThermodynamicsCalculator.calculate_Zn(n, N, K1, S, c) for n in range(N + 1))
    
    @staticmethod
    def return_maximal_rho(N: int, K1: float, S: float, c: float, alpha: float) -> float:
        zed = ThermodynamicsCalculator.calculate_total_Z(N, K1, S, c)
        return alpha * (1 - (zed**-1)) if zed else 0.0
    
    @staticmethod
    def return_nonmaximal_rho(N: int, K1: float, S: float, c: float, alpha: float, K_alpha: float = None, mode: str = "constant") -> float:
        numerator_sum = 0.0
        Z_total = ThermodynamicsCalculator.calculate_total_Z(N, K1, S, c)
        
        # Start at 1, as n=0 will result in an alpha_n of 0 anyway
        for n in range(1, N + 1):
            Zn = ThermodynamicsCalculator.calculate_Zn(n, N, K1, S, c)
            
            if mode == "constant":
                alpha_n = alpha
            elif mode == "linear":
                alpha_n = alpha * n
            elif mode == "saturating" and K_alpha is not None:
                alpha_n = (alpha * K_alpha * n) / (K_alpha * n + 1)
            else:
                alpha_n = 0.0
                
            numerator_sum += alpha_n * Zn
            
        return numerator_sum / Z_total if Z_total else 0.0

class SimulationResultHandler:
    def __init__(self, model):
        self.spatial_states = model.spatial_states
        self.n_binding_sites = model.n_binding_sites
        self.sox2_simulation_variables = model.sox2_simulation_variables
        self.bulk_states = model.bulk_states
        self.times = model.times
        self.dwell_time_df = model.simulation_site_dwell_time_df
        self.simulation_reaction_history_df = model.simulation_reaction_history_df
    
    
    @staticmethod
    def generate_thermodynamic_expectation_rho(S_vals, N_vals, K1_vals, c=1.0, alpha=1.0):
        results = []
        for S, N, K1 in itertools.product(S_vals, N_vals, K1_vals):
            # Make sure to cast N to int, as math.comb requires integers
            rate_maximal = ThermodynamicsCalculator.return_maximal_rho(int(N), K1, S, c, alpha)
            rate_linear = ThermodynamicsCalculator.return_nonmaximal_rho(int(N), K1, S, c, alpha, mode="linear")
            rate_saturating = ThermodynamicsCalculator.return_nonmaximal_rho(int(N), K1, S, c, alpha,K_alpha=1, mode="saturating")
            results.append({"S": S, "N": int(N), "K1": K1, "thermo_maximal_rate": rate_maximal, "thermo_linear_rate": rate_linear, "thermo_saturating_rate": rate_saturating})
            
        return pd.DataFrame(results)

    
    # --- METRIC EXTRACTION ---
    
    def get_effective_transcription_rate(self) -> float:
        """Calculates the average empirical transcription rate from the simulation."""
        if not self.bulk_states:
            return 0.0
            
        # k_prod_m is the 3rd variable in the variables list (index 2)
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

    # --- FILE I/O ---
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

    @staticmethod
    def save_sweep_results(sweep_results: list[dict], filename: str = "sweep_results.csv"):
        if not sweep_results:
            print("CRITICAL: No results to save!")
            return
        pl.DataFrame(sweep_results).write_csv(filename)
        print(f"Successfully saved {len(sweep_results)} trajectories to {filename}")

    @staticmethod
    def load_sweep_results(filename: str = "sweep_results.csv") -> list[dict]:
        if not os.path.exists(filename):
            print(f"CRITICAL: The file {filename} does not exist.")
            return []
        results_list = pl.read_csv(filename).to_dicts()
        print(f"Successfully loaded {len(results_list)} trajectories from {filename}")
        return results_list

    # --- PLOTTING ---
    
    def plot_mrna_trajectory(self, filename: str = "mrna_trajectory.png"):
        """Plots the mRNA count over time for a single simulation."""
        df = self.simulation_parameter_df.to_pandas()
        
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x="time", y="mrna", color="#e74c3c", drawstyle="steps-post", linewidth=1.5)
        
        plt.title("mRNA Trajectory Over Time", fontsize=14, pad=15)
        plt.ylabel("mRNA Count")
        plt.xlabel("Time (s)")
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    @staticmethod
    def plot_3d_transcription_scatter(S_vals, N_vals, K1_vals, c=1.0, alpha=1.0, filename="3d_transcription_heatmap.png"):
        """Plots a 3D scatter plot where the color of the point represents the transcription rate."""
        df = SimulationResultHandler.generate_thermodynamic_expectation_rho(S_vals, N_vals, K1_vals, c, alpha)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points, coloring them by the Transcription Rate
        sc = ax.scatter(
            df['S'], 
            df['N'], 
            df['K1'], 
            c=df['Transcription_Rate'], 
            cmap='magma', 
            marker='o',
            s=60,      # Size of the bubbles
            alpha=0.8  # Slight transparency to see overlapping data
        )
        
        ax.set_title("3D Phase Space of Transcription Rate", pad=20, fontsize=14)
        ax.set_xlabel("Free SOX2 (S)")
        ax.set_ylabel("Total Binding Sites (N)")
        ax.set_zlabel("Affinity ($K_1$)")
        
        # Add a colorbar to act as the 'heatmap' legend
        cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label("Theoretical Transcription Rate", rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def plot_transcription_propensity_heatmap(S_vals, N_vals, K1_vals, c=1.0, alpha=1.0, filename="transcription_heatmap.png"):
        df = SimulationResultHandler.generate_transcription_landscape_data(S_vals, N_vals, K1_vals, c, alpha)

        # Create a FacetGrid, creating a new subplot for every value of K1
        g = sns.FacetGrid(df, col="K1", col_wrap=3, height=4)
        
        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')
            # Pivot the subset of data for this specific K1 value into a 2D matrix
            d = data.pivot(index=args[1], columns=args[0], values=args[2])
            d = d.sort_index(ascending=False) # So smaller N is at the bottom
            sns.heatmap(d, cmap="magma", cbar=True, **kwargs)

        g.map_dataframe(draw_heatmap, 'S', 'N', 'Transcription_Rate')
        g.set_axis_labels("Free SOX2 (S)", "Binding Sites (N)")
        g.set_titles(col_template="Affinity (K1) = {col_name:.2f}")
        g.figure.suptitle("Transcription Rate across S, N, and K1", y=1.05, fontsize=16)
        
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


    def plot_dwell_time_distribution(self, filename: str = "dwell_time_dist.png"):
        if self.dwell_time_df.is_empty():
            print("No dwell time data to plot.")
            return

        df = self.dwell_time_df.to_pandas()
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x="dwell_time", bins=30, kde=True, color="#9b59b6", edgecolor="white")
        
        plt.title("Distribution of Sox2 Residence Times", fontsize=14, pad=10)
        plt.xlabel("Dwell Time (seconds)")
        plt.ylabel("Frequency")
        
        mean_dwell = df["dwell_time"].mean()
        plt.axvline(mean_dwell, color='red', linestyle='--', label=f'Mean: {mean_dwell:.2f}s')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    def plot_bound_state_probability(self, filename: str = "state_probability.png"):
        if not self.bulk_states:
            return

        bound_counts = [state[1] for state in self.bulk_states]
        plt.figure(figsize=(8, 5))
        sns.histplot(x=bound_counts, discrete=True, stat="probability", color="#2ecc71")
        
        plt.title("Microstate Probability Distribution", fontsize=14, pad=10)
        plt.xlabel("Number of Bound Sox2 Molecules ($n$)")
        plt.ylabel("Probability ($P(n)$)")
        plt.xlim(0, self.n_binding_sites)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

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

    def plot_propensity_history(self, filename: str = "transcription_propensity.png"):
        k_prod_m = self.sox2_simulation_variables[2]
        bound_counts = np.array([state[1] for state in self.bulk_states])
        transcription_propensities = bound_counts * k_prod_m

        df = pd.DataFrame({"Time (s)": self.times, "Propensity": transcription_propensities})

        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x="Time (s)", y="Propensity", color="#3b82f6", drawstyle="steps-post", linewidth=1.5)
        
        plt.title("Dynamics of Transcription Propensity", fontsize=14, pad=15)
        plt.ylabel("Propensity (Reaction Rate $a_j$)")
        plt.xlabel("Time (s)")
        
        avg_prop = np.mean(transcription_propensities)
        plt.axhline(avg_prop, color="red", linestyle="--", alpha=0.6, label=f"Mean: {avg_prop:.2f}")
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_parameter_sweep(sweep_results: list[dict], filename: str = "all_parameters_vs_dwell_time.png", mrna_filename: str = "mrna_vs_dwell_time.png"):
        if not sweep_results:
            print("CRITICAL: No results were generated! Check your simulation logic.")
            return

        sweep_df = pd.DataFrame(sweep_results)
        sns.set_theme(context="paper", style="darkgrid", palette="pastel", font="sans-serif")

        # --- PLOT 1 ---
        g1 = sns.relplot(data=sweep_df, x="parameter_value", y="average_dwell_time", col="swept_parameter", col_wrap=4, kind="line", marker="o", errorbar="sd", linewidth=2, height=3, aspect=1.2)
        g1.figure.suptitle("One-At-A-Time Parameter Sweep vs. Average Dwell Time", fontsize=16, y=1.05)
        g1.set_axis_labels("Parameter Value", "Average Dwell Time (s)")
        g1.set_titles("{col_name}") 
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        
        # --- PLOT 2 ---
        g2 = sns.displot(data=sweep_df, x="average_dwell_time", y="average_mrna", col="swept_parameter", col_wrap=4, kind="kde", fill=True, cmap="mako", thresh=0.05, height=3.5, aspect=1.2)
        g2.figure.suptitle("Density of mRNA Output vs. Dwell Time", fontsize=16, y=1.05)
        g2.set_axis_labels("Average Dwell Time (s)", "Average mRNA Count")
        g2.set_titles("{col_name}")        
        plt.savefig(mrna_filename, dpi=300, bbox_inches="tight") # Fixed bug here
        plt.close()

        # --- PLOT 3 ---
        g3 = sns.relplot(data=sweep_df, x="parameter_value", y="average_mrna", col="swept_parameter", col_wrap=4, kind="line", marker="o", errorbar="sd", linewidth=2, height=3, aspect=1.2, color="#e74c3c")
        g3.figure.suptitle("Parameter Sweep vs. Average mRNA Output", fontsize=16, y=1.05)
        g3.set_axis_labels("Parameter Value", "Average mRNA Count")
        g3.set_titles("{col_name}") 
        plt.savefig("parameters_vs_mrna.png", dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_2d_heatmap(sweep_results: list[dict], param_x: str, param_y: str, metric: str = "average_mrna", filename: str = "2d_heatmap.png"):
        df = pd.DataFrame(sweep_results)
        pivot_table = df.pivot_table(values=metric, index=param_y, columns=param_x, aggfunc='mean')
        pivot_table = pivot_table.sort_index(ascending=False)

        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, cmap="magma", cbar_kws={'label': metric.replace("_", " ").title()})
        
        plt.title(f"{metric.replace('_', ' ').title()} Phase Diagram", fontsize=14, pad=15)
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        
    @staticmethod
    def plot_multidim_sweep_heatmap(sweep_results: list[dict], metric: str = "average_mrna", filename: str = "multidim_heatmap.png"):
        if not sweep_results:
            print("CRITICAL: No sweep results to plot!")
            return
            
        df = pd.DataFrame(sweep_results)
        
        # Average the metric across the multiple duplicate trajectories (trajectory_id)
        df_agg = df.groupby(["sox2_free", "n_binding_sites", "affinity"])[metric].mean().reset_index()

        # Create a FacetGrid, creating a new subplot for every value of Affinity (K1)
        g = sns.FacetGrid(df_agg, col="affinity", col_wrap=3, height=4.5)
        
        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')
            # Pivot the subset of data for this specific affinity value into a 2D matrix
            d = data.pivot(index=args[1], columns=args[0], values=args[2])
            d = d.sort_index(ascending=False) # So smaller N is at the bottom
            sns.heatmap(d, cmap="magma", cbar=True, cbar_kws={'label': metric.replace("_", " ").title()}, **kwargs)

        # args mapping: x="sox2_free", y="n_binding_sites", values=metric
        g.map_dataframe(draw_heatmap, "sox2_free", "n_binding_sites", metric)
        g.set_axis_labels("Free SOX2 (S)", "Binding Sites (N)")
        g.set_titles(col_template="Affinity ($K_1$) = {col_name}")
        g.figure.suptitle(f"{metric.replace('_', ' ').title()} Phase Space", y=1.05, fontsize=16)
        
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Heatmap successfully saved to {filename}")

class ClusterModel:
    def __init__(self, model_param: dict, model_var: dict, model_binding_sites: int, sim_max_time: int, record_interval: float = 1.0, track_history: bool = True):
        self.t_max: int = sim_max_time
        self.record_interval = record_interval
        self.sox2_model_parameters: dict = model_param
        self.sox2_model_variables: dict = model_var
        self.n_binding_sites: int = model_binding_sites
        self.track_history = track_history
        sns.set_theme(context="paper", style="darkgrid", palette="pastel", font="sans-serif")

    def _initialize_state(self):
        """Initializes all arrays, constants, and trackers for the simulation."""
        self.sox2_simulation_parameters = [self.sox2_model_parameters[key] for key in self.sox2_model_parameters]
        self.sox2_simulation_variables = [self.sox2_model_variables[key] for key in self.sox2_model_variables]

        self.t = 0.0
        self.next_record_time = 0.0
        
        # Spatial Tracking Arrays
        self.is_free = np.ones(self.n_binding_sites, dtype=bool)
        self.is_unpaired_bound = np.zeros(self.n_binding_sites, dtype=bool)
        self.binding_sites_array = np.zeros(self.n_binding_sites, dtype=np.int8)
        self.parameter_states = np.array(self.sox2_simulation_parameters, dtype=np.int32)
        self.bridged_to = np.full(self.n_binding_sites, -1, dtype=np.int32)
        # Kernel & Hopping Weights
        self.kernel_matrix = np.ones((self.n_binding_sites, self.n_binding_sites), dtype=float)
        np.fill_diagonal(self.kernel_matrix, 0.0)
        self.hop_weights = np.sum(self.kernel_matrix, axis=1)

        # Reaction Constants
        self.stoichiometry_matrix = np.array([
            [1, -1, -1, 1, 0, 0, 0, 0],
            [0, 1, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 0, 0],
        ], dtype=np.int32)

        self.reaction_names = {
            0: "prod_s", 1: "bind", 2: "deg_s", 3: "unbind",
            4: "prod_m", 5: "deg_m", 6: "pair"
        }

        # Data Trackers
        self.times, self.bulk_states, self.spatial_states, self.reaction_history, self.residence_time_states = [], [], [], [], []
        self.site_bind_times = np.full(self.n_binding_sites, -1.0)

    def _calculate_propensities(self):
        """Calculates reaction propensities based on the current state."""
        k_prod_s, k_deg_s, k_prod_m, k_deg_m, k_bind, k_unbind, k_hop, _ = self.sox2_simulation_variables
        sox2_free, sox2_bound, mrna_count = self.parameter_states
        
        unbound_sites = np.sum(self.is_free)
        total_hop_weight = np.sum(self.hop_weights[self.is_unpaired_bound])

        propensities = np.array([
            k_prod_s,  
            k_bind * sox2_free * unbound_sites,  
            k_deg_s * sox2_free,  
            k_unbind * sox2_bound,  
            k_prod_m * sox2_bound,  
            k_deg_m * mrna_count,  
            k_hop * total_hop_weight,  
        ])
        return propensities, np.sum(propensities)

    def _execute_spatial_reaction(self, reaction_index):
        """Processes state changes for spatially dependent reactions (bind, unbind, pair)."""
        site_target, site_paired_with = -1, -1

        if reaction_index == 1:  # bind
            free_indices = np.where(self.is_free)[0]
            site_target = np.random.choice(free_indices)
            
            self.is_free[site_target] = False
            self.is_unpaired_bound[site_target] = True
            self.binding_sites_array[site_target] = 1
            self.hop_weights -= self.kernel_matrix[:, site_target]
            
            self.site_bind_times[site_target] = self.t
            site_paired_with = 0 

        elif reaction_index == 3:  # unbind
            bound_indices = np.where(~self.is_free)[0]
            if len(bound_indices) > 0:
                chosen_site = np.random.choice(bound_indices)
                duration = self.t - self.site_bind_times[chosen_site]
                self.residence_time_states.append([chosen_site, duration])
                self.site_bind_times[chosen_site] = -1.0 
                
                self.is_free[chosen_site] = True
                self.binding_sites_array[chosen_site] = 0
                self.hop_weights += self.kernel_matrix[:, chosen_site]
                
                if self.bridged_to[chosen_site] != -1:
                    paired_site = self.bridged_to[chosen_site]
                    self.bridged_to[chosen_site] = -1
                    self.bridged_to[paired_site] = -1
                    self.is_unpaired_bound[paired_site] = True 
                    
                    self.parameter_states[0] -= 1  
                    self.parameter_states[1] += 1  
                else:
                    self.is_unpaired_bound[chosen_site] = False
                    
                site_target = chosen_site
                site_paired_with = 0 

        elif reaction_index == 6:  # pair
            unpaired_indices = np.where(self.is_unpaired_bound)[0]
            unpaired_weights = self.hop_weights[unpaired_indices]
            
            dwell_site = np.random.choice(unpaired_indices, p=(unpaired_weights / np.sum(unpaired_weights)))
            dest_weights = self.kernel_matrix[dwell_site, :] * self.is_free
            next_site = np.random.choice(self.n_binding_sites, p=(dest_weights / np.sum(dest_weights)))

            site_target, site_paired_with = dwell_site, next_site
            self.is_free[next_site] = False
            self.binding_sites_array[next_site] = 1
            self.hop_weights -= self.kernel_matrix[:, next_site]
            self.is_unpaired_bound[dwell_site] = False 
            
            self.site_bind_times[next_site] = self.t
            self.bridged_to[dwell_site] = next_site
            self.bridged_to[next_site] = dwell_site

        if self.track_history:
            self.reaction_history.append((self.t, self.reaction_names[reaction_index], site_target, site_paired_with))

    def _record_snapshot(self, t_next_reaction):
        """Records bulk and spatial state histories at set intervals."""
        while self.next_record_time <= self.t + t_next_reaction and self.next_record_time <= self.t_max:
            self.times.append(self.next_record_time)
            self.bulk_states.append(self.parameter_states.copy())
            self.spatial_states.append(self.binding_sites_array.copy())
            self.next_record_time += self.record_interval

    def _generate_dataframes(self):
        """Wraps up final states and generates Polars DataFrames."""
        for i in range(self.n_binding_sites):
            if not self.is_free[i] and self.site_bind_times[i] != -1.0:
                self.residence_time_states.append([i, self.t - self.site_bind_times[i]])

        self.simulation_parameter_df = pl.DataFrame({
            "time": self.times,
            "sox2_free": [s[0] for s in self.bulk_states],
            "sox2_bound": [s[1] for s in self.bulk_states],
            "mrna": [s[2] for s in self.bulk_states],
        })
        
        self.simulation_site_dwell_time_df = pl.DataFrame(
            {"dwell_site": [s[0] for s in self.residence_time_states], "dwell_time": [s[1] for s in self.residence_time_states]},
            schema={"dwell_site": pl.Int64, "dwell_time": pl.Float64},
        )

        self.simulation_reaction_history_df = pl.DataFrame(
            self.reaction_history, 
            schema=["time", "reaction_type", "site_target", "site_paired_with"]
        )            
        return self.simulation_parameter_df, self.simulation_site_dwell_time_df, self.simulation_reaction_history_df

    def run_sox2_sim(self):
        """Main execution loop for the stochastic simulation."""
        self._initialize_state()

        while self.t < self.t_max:
            propensities, total_prop = self._calculate_propensities()
            if total_prop == 0:
                break
            
            r1, r2 = np.random.random(2)
            t_next_reaction = (1.0 / total_prop) * np.log(1.0 / r1)
            
            self._record_snapshot(t_next_reaction)
            self.t += t_next_reaction

            cumulative_props = np.cumsum(propensities)
            reaction_index = np.searchsorted(cumulative_props, r2 * total_prop)
            self.parameter_states += self.stoichiometry_matrix[:, reaction_index]

            self._execute_spatial_reaction(reaction_index)

        return self._generate_dataframes()



if __name__ == "__main__":
    output_folder = "simulation_outputs"
    os.makedirs(output_folder, exist_ok=True)
    
    # Base simulation properties
    sox2_model_variables_dict = {
        "k_prod_s": 0,
        "k_deg_s": 0,
        "k_prod_m": 0.01,
        "k_deg_m": 0.005,
        "k_bind": 1,
        "k_unbind": 0.05,
        "k_hop": 1,
        "lambda_hop": 1,
    }
    
    sox2_model_parameters_dict = {"sox2_free": 500, "sox2_bound": 0, "mrna": 0}

    # --- SWEEP CONFIGURATION ---
    # Define ranges to test across our 3 target dimensions: S, N, K1
    S_values = [1, 5, 10, 100]           # Free SOX2 Concentrations
    N_values = [1, 5, 10, 100]         # Number of Binding Sites
    K1_values = [0.1, 0.5, 1.0, 10.0]      # Affinity (k_bind / k_unbind)
    
    sim_time = 1000
    n_trajectories_per_condition = 3
    output_filename = os.path.join(output_folder, "multidim_sweep_results.csv")

