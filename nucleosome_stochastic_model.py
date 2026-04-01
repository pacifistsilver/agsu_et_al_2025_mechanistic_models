import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

class ClusterModel():
    def __init__(self, model_param, model_var, model_binding_sites, sim_max_time):
        self.t_max: int = sim_max_time
        self.sox2_model_parameters: dict = model_param
        self.sox2_model_variables: dict = model_var
        self.n_binding_sites: int = model_binding_sites
        sns.set_theme(context="paper", style="darkgrid", palette="pastel", font="sans-serif")
        
    def plot_nucleosome_occupancy_history(self):
        sns.set_theme(style="ticks")
        custom_cmap = ListedColormap(['#e2e8f0', '#3b82f6'])
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(self.spatial_states, 
                        cmap=custom_cmap, 
                        cbar=False, 
                        yticklabels="auto")
        plt.title("Sox2 Nucleosome Occupancy Over Time", pad=15)
        plt.xlabel("Nucleosome Index (0 to 49)")
        plt.ylabel("Time (Simulation Steps )")
        legend_elements = [Patch(facecolor='#e2e8f0', label='Unbound'),
                        Patch(facecolor='#3b82f6', label='Sox2 Bound')]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
        plt.tight_layout()
        plt.savefig('nuc_history.png')
        
    def plot_mrna_history(self):
        # converts 2d array of numpy array calls into 2d array without numpy formatting.
        # e.g. original format is array([  0,   1, 106], dtype=int32), array([  0,   1, 105], dtype=int32) ... 
        # new format is [[  1   0   0]
        #               [  0   1   0]
        #               [  0   1   0]
        #               ...
        self.bulk_states = np.array(self.bulk_states)  
        df = pd.DataFrame({
            'Time (s)': self.times, 
            'mRNA Molecules': self.bulk_states[:, 2]
        })
        plt.figure(figsize=(10, 5))
        ax = sns.lineplot(
            data=df, 
            x='Time (s)', 
            y='mRNA Molecules', 
            color='#10b981', 
            drawstyle='steps-post',  
            linewidth=1.5
        )
        plt.title("Stochastic mRNA Production Driven by Sox2 Binding", fontsize=14, pad=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig('seaborn_mrna_plot.png')
        
    # return residence time for each site i.e. how long is each index bound for?
    # t1. calculate time spent bound generally   
    # t2. calculate time spent bound for specific states
    def return_residence_time(self):
        # take time spent in the state subtract the next state
        # taking the last state, subtract ts from tmax time
        # note: all valuse in scientific notation e.g. 1 is 1.0x10**1
        np.savetxt("time-log.txt", self.residence_time_states)
        

    def run_sox2_sim(self):
        k_prod_s, k_deg_s, k_prod_m, k_deg_m, k_bind, k_unbind, k_hop, lambda_hop = self.sox2_model_variables
        t = 0.0
        binding_sites_array = np.zeros(self.n_binding_sites, dtype=np.int8) # 0 = free, 1 = bound
        state = np.array(self.sox2_model_parameters, dtype=np.int32) # track bulk molecule counts
        bridged_to = np.full(self.n_binding_sites, -1, dtype=np.int8) # track bridged sites
        indices = np.arange(self.n_binding_sites)
        # initialise 2d matrix where rwo and columns are there corresponding nucleosome number
        # lambda -> hopping distance
        dist_matrix = np.abs(indices[:, None] - indices[None, :]) 
        # probability of hopping to surrounding sites described as an exponential decay curve.
        kernel_matrix = np.exp(-dist_matrix / lambda_hop)
        np.fill_diagonal(kernel_matrix, 0.0)

        # Rows: [sox2_free, sox2_bound, mrna_count]
        # Cols: [prod_s, bind, deg_s, unbind, prod_m, deg_m, slide_left, slide_right]
        stoichiometry_matrix = np.array([
            [ 1, -1, -1,  1,  0,  0,  0,  0], 
            [ 0,  1,  0, -1,  0,  0,  0,  0], 
            [ 0,  0,  0,  0,  1, -1,  0,  0]  
        ], dtype=np.int32)
        
        self.times = [0.0]
        self.bulk_states = [state.copy()]                  
        self.spatial_states = [binding_sites_array.copy()]    
        self.residence_time_states = []
        
        # issue: need to update bridged_to array at the end, ensure sox2 isnt accidently created when unbound is called, and more
        while t < self.t_max:
            sox2_free, sox2_bound, mrna_count = state
            bound_vec = (binding_sites_array == 1).astype(float)
            empty_vec = (binding_sites_array == 0).astype(float)
            unbound_nucleosomes = np.sum(binding_sites_array == 0)
            bound_nucleosomes = np.sum(binding_sites_array == 1)
            # p_matrix contains all valid weights of a molecule moving from i to j nucleosome
            # removes all other points where j is bound or i is empty. 
            P_matrix = kernel_matrix * bound_vec[:, None] * empty_vec[None, :]  
            total_hop_weight = np.sum(P_matrix) # summing all valid weights.
                        
            ## gillespie logic
            propensities = np.array([
                k_prod_s,                                 # 0: prod_s
                k_bind * sox2_free * unbound_nucleosomes, # 1: bind
                k_deg_s * sox2_free,                      # 2: deg_s
                k_unbind * sox2_bound,                    # 3: unbind
                k_prod_m * sox2_bound,                    # 4: prod_m
                k_deg_m * mrna_count,                     # 5: deg_m
                k_hop * total_hop_weight                  # 6: hop
                ])            
            total_prop = np.sum(propensities)
            if total_prop == 0:
                break
            r1, r2 = np.random.random(2)
            tau = (1.0 / total_prop) * np.log(1.0 / r1)
            t += tau
            cumulative_props = np.cumsum(propensities)
            reaction_index = np.searchsorted(cumulative_props, r2 * total_prop)
            state += stoichiometry_matrix[:, reaction_index]
            
            ## reaction logic
            if reaction_index == 1: # bind reaction
                free_sites = np.where(binding_sites_array == 0)[0]
                binding_sites_array[np.random.choice(free_sites)] = 1
                
            elif reaction_index == 3: # unbind reaction
                bound_sites = np.where(binding_sites_array == 1)[0]
                binding_sites_array[np.random.choice(bound_sites)] = 0
                
            elif reaction_index == 6: # pair reaction
                        # flatten p_matrix into 1d array
                        flat_probs = P_matrix.ravel() / total_hop_weight
                        # select specific transition based on probability
                        chosen_flat_idx = np.random.choice(self.n_binding_sites * self.n_binding_sites, p=flat_probs)
                        # convert said index into a 2d array
                        source_idx, target_idx = np.unravel_index(chosen_flat_idx, (self.n_binding_sites, self.n_binding_sites))
                        
                        # need to add a state where it is hopped to another site, but also it also remains at the original site as well.
                        # so effectively 1 sox2 can occupy two sites at once. 
                        # separate sox2 array?
                        # track where sox2 is bound and that it is the same molecule. 
                        # sox2 when bound to two sites cannot hop to another site.
                        # or.. add a rule where you track where and when a hop has occured 
                        # you can feed this into the list of valid transitions
                        binding_sites_array[source_idx] = 0
                        binding_sites_array[target_idx] = 1
                        self.residence_time_states.append([source_idx, target_idx])
            
            self.times.append(t)
            self.bulk_states.append(state.copy())
            self.spatial_states.append(binding_sites_array.copy())
          
        return self.times, np.array(self.bulk_states), np.array(self.spatial_states)
sox2_model_variables = {
    "k_prod_s": 0,
    "k_deg_s": 0,
    "k_prod_m": 1,
    "k_deg_m": 0.01,
    "k_bind": 1,
    "k_unbind": 0.01,
    "k_hop": 1,
    "lambda_hop": 1
}
sox2_model_parameters = {
    "sox2_free": 1,
    "sox2_bound": 0,
    "mrna": 0
}
sox2_simulation_parameters = [sox2_model_parameters[key] for key in sox2_model_parameters]
sox2_simulation_variables = [sox2_model_variables[key] for key in sox2_model_variables]

## model calling
model_call = ClusterModel(sox2_simulation_parameters, sox2_simulation_variables, model_binding_sites=50, sim_max_time=500)
ClusterModel.run_sox2_sim(model_call)
ClusterModel.plot_mrna_history(model_call)
ClusterModel.plot_nucleosome_occupancy_history(model_call)
ClusterModel.return_residence_time(model_call)