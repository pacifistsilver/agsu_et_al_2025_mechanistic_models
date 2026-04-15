import numpy as np
import seaborn as sns
import polars as pl
import matplotlib.pyplot as plt
import math
import pandas as pd
import concurrent.futures
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

class SimulationResultHandler:
    def __init__(self, model):
        """
        Initialise the handler with data from a completed ClusterModel simulation.
        """
        self.spatial_states = model.spatial_states
        self.n_binding_sites = model.n_binding_sites
        self.sox2_simulation_variables = model.sox2_simulation_variables
        self.bulk_states = model.bulk_states
        self.times = model.times
        self.dwell_time_df = model.simulation_site_dwell_time_df
        self.simulation_reaction_history_df = model.simulation_reaction_history_df
    
    def get_average_mrna(self):
        """Calculates and returns the time-averaged mRNA count."""
        if not self.bulk_states:
            return 0.0
        # The mRNA count is the 3rd element (index 2) in the bulk state array [sox2_free, sox2_bound, mrna]
        mrna_counts = [state[2] for state in self.bulk_states]
        return sum(mrna_counts) / len(mrna_counts)

    def plot_dwell_time_distribution(self, filename="dwell_time_dist.png"):
        """Plots a histogram and KDE of all individual dwell times."""
        if self.dwell_time_df.is_empty():
            print("No dwell time data to plot.")
            return


        plt.figure(figsize=(8, 5))
        # Convert to pandas for plotting
        df = self.dwell_time_df.to_pandas()

        sns.histplot(
            data=df, 
            x="dwell_time", 
            bins=30, 
            kde=True, 
            color="#9b59b6",
            edgecolor="white"
        )
        
        plt.title("Distribution of Sox2 Residence Times", fontsize=14, pad=10)
        plt.xlabel("Dwell Time (seconds)")
        plt.ylabel("Frequency")
        
        # Add a line for the mean
        mean_dwell = df["dwell_time"].mean()
        plt.axvline(mean_dwell, color='red', linestyle='--', label=f'Mean: {mean_dwell:.2f}s')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    def plot_bound_state_probability(self, filename="state_probability.png"):
        """Plots the fraction of time the cluster spends with 'n' molecules bound."""
        if not self.bulk_states:
            return

        # Extract the number of bound Sox2 molecules (index 1) at every snapshot
        bound_counts = [state[1] for state in self.bulk_states]

        plt.figure(figsize=(8, 5))
        sns.histplot(
            x=bound_counts, 
            discrete=True, 
            stat="probability", # Crucial: converts raw counts to fraction of total time
            color="#2ecc71"
        )
        
        plt.title("Microstate Probability Distribution", fontsize=14, pad=10)
        plt.xlabel("Number of Bound Sox2 Molecules ($n$)")
        plt.ylabel("Probability ($P(n)$)")
        plt.xlim(0, self.n_binding_sites)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    @staticmethod
    def plot_parameter_sweep(sweep_results, filename="all_parameters_vs_dwell_time.png", mrna_filename="mrna_vs_dwell_time.png"):
        """Plots the results of a 1D parameter sweep."""
        if not sweep_results:
            print("CRITICAL: No results were generated! Check your simulation logic.")
            return

        import pandas as pd
        sweep_df = pd.DataFrame(sweep_results)

        sns.set_theme(context="paper", style="darkgrid", palette="pastel", font="sans-serif")

        # --- PLOT 1: Parameter Value vs Dwell Time ---
        g1 = sns.relplot(
            data=sweep_df,
            x="parameter_value",
            y="average_dwell_time",
            col="swept_parameter", 
            col_wrap=4,            
            kind="line",
            marker="o",
            errorbar="sd",         
            linewidth=2,
            height=3,
            aspect=1.2
        )
        g1.figure.suptitle("One-At-A-Time Parameter Sweep vs. Average Dwell Time", fontsize=16, y=1.05)
        g1.set_axis_labels("Parameter Value", "Average Dwell Time (s)")
        g1.set_titles("{col_name}") 
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        
        # --- PLOT 2: Dwell Time vs Average mRNA ---
        g2 = sns.displot(
            data=sweep_df,
            x="average_dwell_time",
            y="average_mrna",
            col="swept_parameter", 
            col_wrap=4,
            kind="kde",       # 2D Kernel Density Estimation
            fill=True,        # Fills the contours with solid color gradients
            cmap="mako",      # A great sequential colormap for density
            thresh=0.05,      # Trims the ultra-low density outer noise
            height=3.5,
            aspect=1.2
        )
        
        g2.figure.suptitle("Density of mRNA Output vs. Dwell Time", fontsize=16, y=1.05)
        g2.set_axis_labels("Average Dwell Time (s)", "Average mRNA Count")
        g2.set_titles("{col_name}")        
        plt.savefig("mrna_filename", dpi=300, bbox_inches="tight")
        
        g3 = sns.relplot(
            data=sweep_df,
            x="parameter_value",
            y="average_mrna",
            col="swept_parameter", 
            col_wrap=4,            
            kind="line",
            marker="o",
            errorbar="sd",  # Shows the standard deviation across your 100 trajectories       
            linewidth=2,
            height=3,
            aspect=1.2,
            color="#e74c3c" # Giving mRNA a distinct red color
        )
        g3.figure.suptitle("Parameter Sweep vs. Average mRNA Output", fontsize=16, y=1.05)
        g3.set_axis_labels("Parameter Value", "Average mRNA Count")
        g3.set_titles("{col_name}") 
        plt.savefig("parameters_vs_mrna.png", dpi=300, bbox_inches="tight")
        plt.show()
    @staticmethod
    def plot_2d_heatmap(sweep_results, param_x, param_y, metric="average_mrna", filename="2d_heatmap.png"):
        """Plots a 2D heatmap showing how two parameters interact to drive a metric."""

        df = pd.DataFrame(sweep_results)
        
        # Pivot the data so param_y is on the rows, param_x is on the columns, and the metric is the value
        pivot_table = df.pivot_table(values=metric, index=param_y, columns=param_x, aggfunc='mean')
        
        # Sort the index so the y-axis goes from low (bottom) to high (top)
        pivot_table = pivot_table.sort_index(ascending=False)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            pivot_table, 
            cmap="magma", 
            cbar_kws={'label': metric.replace("_", " ").title()}
        )
        
        plt.title(f"{metric.replace('_', ' ').title()} Phase Diagram", fontsize=14, pad=15)
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
    def plot_nucleosome_occupancy_history(self, filename="nuc_history.png"):
        """Plots the occupancy heatmap of Sox2 binding sites over time."""
        sns.set_theme(style="ticks")
        custom_cmap = ListedColormap(["#e2e8f0", "#3b82f6"])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.spatial_states, 
            cmap=custom_cmap, 
            cbar=False, 
            yticklabels="auto"
        )
        
        plt.title("Sox2 Nucleosome Occupancy Over Time", pad=15)
        plt.xlabel(f"Nucleosome Index (0 to {self.n_binding_sites - 1})")
        plt.ylabel("Time (Simulation Steps)")
        
        legend_elements = [
            Patch(facecolor="#e2e8f0", label="Unbound"),
            Patch(facecolor="#3b82f6", label="Sox2 Bound"),
        ]
        plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.25, 1))
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_propensity_history(self, filename="transcription_propensity.png"):
        """Calculates and plots the transcription propensity dynamics."""
        # Extract k_prod_m (3rd index in variables)
        k_prod_m = self.sox2_simulation_variables[2]
        
        # Calculate propensity: sox2_bound (index 1) * k_prod_m
        bulk_array = np.array(self.bulk_states)
        transcription_propensities = bulk_array[:, 1] * k_prod_m

        df = pl.DataFrame({
            "Time (s)": self.times, 
            "Propensity": transcription_propensities
        })

        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=df,
            x="Time (s)",
            y="Propensity",
            color="#3b82f6",
            drawstyle="steps-post",
            linewidth=1.5,
        )
        
        plt.title("Dynamics of Transcription Propensity", fontsize=14, pad=15)
        plt.ylabel("Propensity (Reaction Rate $a_j$)")
        plt.xlabel("Time (s)")
        
        avg_prop = np.mean(transcription_propensities)
        plt.axhline(avg_prop, color="red", linestyle="--", alpha=0.6, label=f"Mean: {avg_prop:.2f}")
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_residence_time_log(self, filename="time-log.txt"):
        """Groups and saves the total dwell time per binding site."""
        residence_time_states = (
            self.dwell_time_df.group_by("dwell_site")
            .agg(pl.col("dwell_time").sum())
            .sort(by="dwell_time", descending=True)
        )
        
        np.savetxt(
            filename,
            residence_time_states,
            fmt="%g",
            header="dwell_site/dwell_time",
            delimiter=", ",
        )
        
    def save_reaction_history_log(self, filename="reaction_history.txt"):
            self.simulation_reaction_history_df = self.simulation_reaction_history_df.cast({"site_target": pl.Int64, "site_paired_with": pl.Int64})
            self.simulation_reaction_history_df.write_csv(filename)
                   
    def get_average_dwell_time(self):
        """Calculates and returns the average dwell time for the trajectory."""
        if self.dwell_time_df.is_empty():
            return 0.0
        return self.dwell_time_df.select(pl.col("dwell_time").mean()).item()

    @staticmethod
    def save_sweep_results(sweep_results, filename="sweep_results.csv"):
        """Saves the aggregated sweep results to a CSV file."""
        if not sweep_results:
            print("CRITICAL: No results to save!")
            return
        
        # Convert the list of dictionaries to a Polars DataFrame and save
        df = pl.DataFrame(sweep_results)
        df.write_csv(filename)
        print(f"Successfully saved {len(sweep_results)} trajectories to {filename}")

    @staticmethod
    def load_sweep_results(filename="sweep_results.csv"):
        """Reads sweep results from a CSV and returns them as a list of dictionaries."""
        import os
        if not os.path.exists(filename):
            print(f"CRITICAL: The file {filename} does not exist.")
            return []
        
        # Read the CSV and convert it back to the exact list-of-dicts format
        df = pl.read_csv(filename)
        results_list = df.to_dicts()
        print(f"Successfully loaded {len(results_list)} trajectories from {filename}")
        
        return results_list

class ClusterModel:
    def __init__(self, model_param: dict, model_var: dict, model_binding_sites: int, sim_max_time: int, record_interval: float = 1.0, track_history: bool = True):
        self.t_max: int = sim_max_time
        self.record_interval = record_interval
        self.sox2_model_parameters: dict = model_param
        self.sox2_model_variables: dict = model_var
        self.n_binding_sites: int = model_binding_sites
        self.track_history = track_history
        sns.set_theme(
            context="paper", style="darkgrid", palette="pastel", font="sans-serif"
        )

    def run_sox2_sim(self, output_filename="reaction_history.csv"):
        self.sox2_simulation_parameters = [
            self.sox2_model_parameters[key] for key in self.sox2_model_parameters
        ]
        self.sox2_simulation_variables = [
            self.sox2_model_variables[key] for key in self.sox2_model_variables
        ]

        k_prod_s, k_deg_s, k_prod_m, k_deg_m, k_bind, k_unbind, k_hop, lambda_hop = (
            self.sox2_simulation_variables
        )

        t = 0.0
        # track the states with O(1)/O(N) boolean masks instead of np.where scanning
        is_free = np.ones(self.n_binding_sites, dtype=bool)
        is_unpaired_bound = np.zeros(self.n_binding_sites, dtype=bool)
        
        binding_sites_array = np.zeros(self.n_binding_sites, dtype=np.int8)
        parameter_states = np.array(self.sox2_simulation_parameters, dtype=np.int32)
        bridged_to = np.full(self.n_binding_sites, -1, dtype=np.int32)
        indices = np.arange(self.n_binding_sites)
        
        kernel_matrix = np.ones((self.n_binding_sites, self.n_binding_sites), dtype=float)
        np.fill_diagonal(kernel_matrix, 0.0)

        # precompute local hop weights (O(N) updates instead of full matrix rebuild)
        # hop_weights[i] represents the sum of hopping probabilities from site i to all currently free sites.
        hop_weights = np.sum(kernel_matrix, axis=1)

        stoichiometry_matrix = np.array(
            [
                [1, -1, -1, 1, 0, 0, 0, 0],
                [0, 1, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, 0, 0],
            ],
            dtype=np.int32,
        )

        reaction_names = {
            0: "prod_s", 1: "bind", 2: "deg_s", 3: "unbind",
            4: "prod_m", 5: "deg_m", 6: "pair"
        }

        self.times = []
        self.bulk_states = []
        self.spatial_states = []
        self.reaction_history = [] 
        self.residence_time_states = []
        site_bind_times = np.full(self.n_binding_sites, -1.0)
        
        next_record_time = 0.0 

        while t < self.t_max:
            sox2_free, sox2_bound, mrna_count = parameter_states
            
            unbound_nucleosomes = np.sum(is_free)
            # fetch valid hopping weights dynamically without matrix math
            total_hop_weight = np.sum(hop_weights[is_unpaired_bound])

            propensities = np.array([
                k_prod_s,  
                k_bind * sox2_free * unbound_nucleosomes,  
                k_deg_s * sox2_free,  
                k_unbind * sox2_bound,  
                k_prod_m * sox2_bound,  
                k_deg_m * mrna_count,  
                k_hop * total_hop_weight,  
            ])
            total_prop = np.sum(propensities)
            if total_prop == 0:
                break
            
            r1, r2 = np.random.random(2)
            t_next_reaction = (1.0 / total_prop) * np.log(1.0 / r1)
            
            #snapshot states at set intervals, NOT every micro-step
            while next_record_time <= t + t_next_reaction and next_record_time <= self.t_max:
                self.times.append(next_record_time)
                self.bulk_states.append(parameter_states.copy())
                self.spatial_states.append(binding_sites_array.copy())
                next_record_time += self.record_interval

            t += t_next_reaction

            cumulative_props = np.cumsum(propensities)
            reaction_index = np.searchsorted(cumulative_props, r2 * total_prop)
            parameter_states += stoichiometry_matrix[:, reaction_index]

            site_target = -1
            site_paired_with = -1

            if reaction_index == 1:  # bind
                free_indices = np.where(is_free)[0]
                site_target = np.random.choice(free_indices)
                
                # Update masks & arrays
                is_free[site_target] = False
                is_unpaired_bound[site_target] = True
                binding_sites_array[site_target] = 1
                
                # Global hop weights update: target site is no longer a valid destination for ANY molecule
                hop_weights -= kernel_matrix[:, site_target]
                
                site_bind_times[site_target] = t
                site_paired_with = 0 

            elif reaction_index == 3:  # unbind
                bound_indices = np.where(~is_free)[0]
                
                if len(bound_indices) > 0:
                    chosen_site = np.random.choice(bound_indices)
                    
                    duration = t - site_bind_times[chosen_site]
                    self.residence_time_states.append([chosen_site, duration])
                    site_bind_times[chosen_site] = -1.0 
                    
                    is_free[chosen_site] = True
                    binding_sites_array[chosen_site] = 0
                    
                    # Global hop weights update: chosen site is now a valid destination again
                    hop_weights += kernel_matrix[:, chosen_site]
                    
                    if bridged_to[chosen_site] != -1:
                        paired_site = bridged_to[chosen_site]
                        bridged_to[chosen_site] = -1
                        bridged_to[paired_site] = -1
                        is_unpaired_bound[paired_site] = True # the remaining molecule is now unpaired
                        
                        # lowkey jank but thats okay here
                        parameter_states[0] -= 1  
                        parameter_states[1] += 1  
                    else:
                        is_unpaired_bound[chosen_site] = False
                        
                    site_target = chosen_site
                    site_paired_with = 0 

            elif reaction_index == 6:  # pair
                # tw-step selection (O(N))
                unpaired_indices = np.where(is_unpaired_bound)[0]
                unpaired_weights = hop_weights[unpaired_indices]
                
                # pick the source molecule based on its individual hop capability
                dwell_site = np.random.choice(unpaired_indices, p=(unpaired_weights / np.sum(unpaired_weights)))
                # pick destination relative only to the source molecule
                dest_weights = kernel_matrix[dwell_site, :] * is_free
                next_site = np.random.choice(self.n_binding_sites, p=(dest_weights / np.sum(dest_weights)))

                site_target = dwell_site
                site_paired_with = next_site

                is_free[next_site] = False
                binding_sites_array[next_site] = 1
                
                # global hop weights update: next_site is taken
                hop_weights -= kernel_matrix[:, next_site]
                is_unpaired_bound[dwell_site] = False 
                
                site_bind_times[next_site] = t
                bridged_to[dwell_site] = next_site
                bridged_to[next_site] = dwell_site

            if self.track_history:
                self.reaction_history.append((t, reaction_names[reaction_index], site_target, site_paired_with))

        # record any sites still bound at the end
        for i in range(self.n_binding_sites):
            if not is_free[i] and site_bind_times[i] != -1.0:
                duration = t - site_bind_times[i]
                self.residence_time_states.append([i, duration])

        self.simulation_parameter_df = pl.DataFrame(
            {
                "time": self.times,
                "sox2_free": [state[0] for state in self.bulk_states],
                "sox2_bound": [state[1] for state in self.bulk_states],
                "mrna": [state[2] for state in self.bulk_states],
            }
        )
        
        dwell_sites = [state[0] for state in self.residence_time_states]
        dwell_times = [state[1] for state in self.residence_time_states]
        self.simulation_site_dwell_time_df = pl.DataFrame(
            {"dwell_site": dwell_sites, "dwell_time": dwell_times},
            schema={"dwell_site": pl.Int64, "dwell_time": pl.Float64},
        )

        self.simulation_reaction_history_df = pl.DataFrame(
            self.reaction_history, 
            schema=["time", "reaction_type", "site_target", "site_paired_with"]
        )            
        return self.simulation_parameter_df, self.simulation_site_dwell_time_df, self.simulation_reaction_history_df

    """Handles the statistical thermodynamic formulas for the Sox2 cluster model."""
    
def run_single_trajectory(param_name, val, trajectory_id, base_params, base_vars, sites, max_time):
    """Runs a single simulation and returns the extracted metric."""
    current_vars = base_vars.copy()
    current_vars[param_name] = val
    
    model_call = ClusterModel(
        base_params.copy(),
        current_vars,
        model_binding_sites=sites,
        sim_max_time=max_time,
        track_history=True 
    )
    sim_params, sim_dwell_times, sim_reactions = model_call.run_sox2_sim()
    
    handler = SimulationResultHandler(model_call)
    
    # Extract BOTH metrics
    avg_dwell = handler.get_average_dwell_time()
    avg_mrna = handler.get_average_mrna() 
    
    return {
        "swept_parameter": param_name,
        "parameter_value": val,
        "trajectory_id": trajectory_id,
        "average_dwell_time": avg_dwell,
        "average_mrna": avg_mrna # Pass this new metric back to the main loop
    }
## Parameter and Variables
sox2_model_variables_dict = {
    "k_prod_s": 1,
    "k_deg_s": 0.05,
    "k_prod_m": 1,
    "k_deg_m": 0.05,
    "k_bind": 1,
    "k_unbind": 0.05,
    "k_hop": 1,
    "lambda_hop": 1,
}
sox2_model_parameters_dict = {"sox2_free": 1, "sox2_bound": 0, "mrna": 0}

## Model Calling
if __name__ == "__main__":
    import concurrent.futures
    import os

    # --- TOGGLE THIS TO SWITCH BETWEEN COMPUTING AND JUST PLOTTING ---
    RUN_NEW_SWEEP = False 
    RESULTS_FILE = "sweep_results.csv"
    
    if RUN_NEW_SWEEP:
        sox2_model_variables_dict = {
            "k_prod_s": 1,
            "k_deg_s": 0.05,
            "k_prod_m": 1,
            "k_deg_m": 0.05,
            "k_bind": 1,
            "k_unbind": 0.05,
            "k_hop": 1,
            "lambda_hop": 1,
        }
        sox2_model_parameters_dict = {"sox2_free": 1, "sox2_bound": 0, "mrna": 0}

        k_values = np.linspace(0.1, 1.0, 10)
        n_trajectories_per_value = 100
        sim_time = 500
        sweep_results = []

        print("Building task list...")
        futures = [] 
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            for param_name in sox2_model_variables_dict.keys():
                for val in k_values:
                    for i in range(n_trajectories_per_value):
                        future = executor.submit(
                            run_single_trajectory,
                            param_name, val, i,
                            sox2_model_parameters_dict,
                            sox2_model_variables_dict,
                            5, sim_time
                        )
                        futures.append(future)
            
            print(f"Executing {len(futures)} parallel simulations. Please wait...")
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    sweep_results.append(result)
                    print(f"Adding {result}")
                except Exception as e:
                    print(f"A simulation failed with error: {e}")

        print("All simulations complete.")
        
        # Save the results to disk immediately after finishing!
        SimulationResultHandler.save_sweep_results(sweep_results, filename=RESULTS_FILE)

    else:
        # Bypass the simulations and just load the data from your CSV
        print("Bypassing simulation compute. Loading saved data...")
        sweep_results = SimulationResultHandler.load_sweep_results(filename=RESULTS_FILE)


    # Finally, plot the results (whether freshly computed or loaded from disk)
    if sweep_results:
        #SimulationResultHandler.plot_parameter_sweep(sweep_results)
        SimulationResultHandler.plot_2d_heatmap(filename="my_state_probs.png")