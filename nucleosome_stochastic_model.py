import numpy as np
import seaborn as sns
import polars as pl
import matplotlib.pyplot as plt
import math
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

class ClusterModel:
    def __init__(
        self,
        model_param: dict,
        model_var: dict,
        model_binding_sites: int,
        sim_max_time: int,
    ):
        self.t_max: int = sim_max_time
        self.sox2_model_parameters: dict = model_param
        self.sox2_model_variables: dict = model_var
        self.n_binding_sites: int = model_binding_sites
        sns.set_theme(
            context="paper", style="darkgrid", palette="pastel", font="sans-serif"
        )

    def plot_nucleosome_occupancy_history(self):
        sns.set_theme(style="ticks")
        custom_cmap = ListedColormap(["#e2e8f0", "#3b82f6"])
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            self.spatial_states, cmap=custom_cmap, cbar=False, yticklabels="auto"
        )
        plt.title("Sox2 Nucleosome Occupancy Over Time", pad=15)
        plt.xlabel("Nucleosome Index (0 to 49)")
        plt.ylabel("Time (Simulation Steps )")
        legend_elements = [
            Patch(facecolor="#e2e8f0", label="Unbound"),
            Patch(facecolor="#3b82f6", label="Sox2 Bound"),
        ]
        plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.25, 1))
        plt.tight_layout()
        plt.savefig("nuc_history.png")

    def plot_mrna_history(self):
        # converts 2d array of numpy array calls into 2d array without numpy formatting.
        # e.g. original format is array([  0,   1, 106], dtype=int32), array([  0,   1, 105], dtype=int32) ...
        # new format is [[  1   0   0]
        #               [  0   1   0]
        #               [  0   1   0]
        #               ...
        self.bulk_states = np.array(self.bulk_states)
        df = pl.DataFrame(
            {"Time (s)": self.times, "mRNA Molecules": self.bulk_states[:, 2]}
        )
        plt.figure(figsize=(10, 5))
        ax = sns.lineplot(
            data=df,
            x="Time (s)",
            y="mRNA Molecules",
            color="#10b981",
            drawstyle="steps-post",
            linewidth=1.5,
        )
        plt.title(
            "Stochastic mRNA Production Driven by Sox2 Binding", fontsize=14, pad=15
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig("seaborn_mrna_plot.png")

    def return_residence_time(self):
        # for each site int64 calculate the sum of the dwell time
        self.residence_time_states = self.simulation_site_dwell_time_df.group_by(
            "dwell_site"
        ).agg(pl.col("dwell_time").sum())
        np.savetxt(
            "time-log.txt",
            self.residence_time_states.sort(by="dwell_time", descending=True),
            fmt="%g",
            header="dwell_site/dwell_time",
            delimiter=", ",
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
            binding_sites_array = np.zeros(self.n_binding_sites, dtype=np.int8)
            parameter_states = np.array(self.sox2_simulation_parameters, dtype=np.int32)
            bridged_to = np.full(self.n_binding_sites, -1, dtype=np.int8)
            indices = np.arange(self.n_binding_sites)
            
            dist_matrix = np.abs(indices[:, None] - indices[None, :])
            kernel_matrix = np.exp(-dist_matrix / lambda_hop)
            np.fill_diagonal(kernel_matrix, 0.0)

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

            self.times = [0.0]
            self.bulk_states = [parameter_states.copy()]
            self.spatial_states = [binding_sites_array.copy()]
            
            self.reaction_history = [] 
            self.residence_time_states = []
            
            # NEW: Array to track the exact timestamp when each site becomes occupied
            site_bind_times = np.full(self.n_binding_sites, -1.0)

            while t < self.t_max:
                sox2_free, sox2_bound, mrna_count = parameter_states
                
                unpaired_bound_vec = ((binding_sites_array == 1) & (bridged_to == -1)).astype(float)
                empty_vec = (binding_sites_array == 0).astype(float)
                unbound_nucleosomes = np.sum(binding_sites_array == 0)
                
                P_matrix = kernel_matrix * unpaired_bound_vec[:, None] * empty_vec[None, :]
                total_hop_weight = np.sum(P_matrix)

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
                t += t_next_reaction
                cumulative_props = np.cumsum(propensities)
                reaction_index = np.searchsorted(cumulative_props, r2 * total_prop)
                parameter_states += stoichiometry_matrix[:, reaction_index]

                site_1 = -1
                site_2 = -1

                if reaction_index == 1:  # bind
                    free_sites = np.where(binding_sites_array == 0)[0]
                    site_1 = np.random.choice(free_sites)
                    binding_sites_array[site_1] = 1
                    
                    # RECORD BIND TIME
                    site_bind_times[site_1] = t

                elif reaction_index == 3:  # unbind
                    bound_sites = np.where(binding_sites_array == 1)[0]
                    
                    if len(bound_sites) > 0:
                        chosen_site = np.random.choice(bound_sites)
                        
                        # CALCULATE AND RECORD TRUE DWELL TIME
                        duration = t - site_bind_times[chosen_site]
                        self.residence_time_states.append([chosen_site, duration])
                        site_bind_times[chosen_site] = -1.0 # reset timer
                        
                        binding_sites_array[chosen_site] = 0
                        
                        if bridged_to[chosen_site] != -1:
                            paired_site = bridged_to[chosen_site]
                            bridged_to[chosen_site] = -1
                            bridged_to[paired_site] = -1
                            
                            # Correct the Gillespie mass balance
                            parameter_states[0] -= 1  
                            parameter_states[1] += 1  
                            
                        site_1 = chosen_site

                elif reaction_index == 6:  # pair
                    flat_probs = P_matrix.ravel() / total_hop_weight
                    chosen_flat_idx = np.random.choice(
                        self.n_binding_sites * self.n_binding_sites, p=flat_probs
                    )
                    dwell_site, next_site = np.unravel_index(
                        chosen_flat_idx, (self.n_binding_sites, self.n_binding_sites)
                    )

                    site_1 = dwell_site
                    site_2 = next_site

                    binding_sites_array[next_site] = 1
                    
                    # RECORD BIND TIME FOR THE NEWLY BRIDGED SITE
                    # (dwell_site remains bound, so its bind time continues uninterrupted)
                    site_bind_times[next_site] = t
                    
                    bridged_to[dwell_site] = next_site
                    bridged_to[next_site] = dwell_site

                self.reaction_history.append((t, reaction_names[reaction_index], site_1, site_2))

                self.times.append(t)
                self.bulk_states.append(parameter_states.copy())
                self.spatial_states.append(binding_sites_array.copy())

            # NEW: End-of-simulation sweep
            # Record the dwell time of any sites that are STILL bound when t reaches t_max
            for i in range(self.n_binding_sites):
                if binding_sites_array[i] == 1 and site_bind_times[i] != -1.0:
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
                schema=["time", "reaction_type", "site_1", "site_2"]
            )
            self.simulation_reaction_history_df = self.simulation_reaction_history_df.cast({"site_1": pl.Int64, "site_2": pl.Int64})
            self.simulation_reaction_history_df.write_csv(output_filename)
            
            return self.simulation_parameter_df, self.simulation_site_dwell_time_df

## Parameter and Variables
sox2_model_variables_dict = {
    "k_prod_s": 0,
    "k_deg_s": 0,
    "k_prod_m": 1,
    "k_deg_m": 0,
    "k_bind": 1,
    "k_unbind": 0.01,
    "k_hop": 1,
    "lambda_hop": 1,
}
sox2_model_parameters_dict = {"sox2_free": 1, "sox2_bound": 0, "mrna": 0}

## Model Calling
model_call = ClusterModel(
    sox2_model_parameters_dict,
    sox2_model_variables_dict,
    model_binding_sites=50,
    sim_max_time=500,
)
param_df, dwell_df = model_call.run_sox2_sim()
model_call.plot_mrna_history()
model_call.plot_nucleosome_occupancy_history()
model_call.return_residence_time()

