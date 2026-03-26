import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# sox2 binding/unbinding to nucleosome
# probably done at some k association rate
# free sox2 -> bound sox2
# nucleosome -> nucleosome x sox2
# transcription only occurs when nucleosome is bound to sox2
# sox2 transfering to adjacent cluster

# task 1.1: add some spatial term? > random chance of free sox2 diffusing into binding the cluster
# task 1.2: 1d sliding term? -> same as task 2 lowkey X
# some probability of sliding to the next nuclesome X
# task 2: intercluster transfer? X
# sox2 initiating clustering? <- don't know  how to implement this
def plot_nucleosome_occupancy_history(occupancy_history):
    custom_cmap = ListedColormap(['#e2e8f0', '#3b82f6'])
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(occupancy_history, 
                    cmap=custom_cmap, 
                    cbar=False, 
                    yticklabels=True)
    plt.title("Sox2 Spatial Binding Over Time", fontsize=14, pad=15)
    plt.xlabel("Nucleosome Index (0 to 49)", fontsize=12)
    plt.ylabel("Time (Simulation Steps )", fontsize=12)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e2e8f0', label='Unbound'),
                    Patch(facecolor='#3b82f6', label='Sox2 Bound')]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.show()

def simulate_sox2_with_sliding(state_parameters, state_variables, t_max):
    t = 0.0
    
    # Unpack parameters (Added k_slide at the end)
    k_prod_s, k_deg_s, k_prod_m, k_deg_m, k_bind, k_unbind, k_slide = state_variables
    
    # 50 nucleosomes: 0 = Unbound, 1 = Bound
    nucleosome_array = np.zeros(50, dtype=np.int8)
    state = np.array(state_parameters, dtype=np.int32)
    
    # STOICHIOMETRY MATRIX
    # Rows: [sox2_free, sox2_bound, mrna_count]
    # Cols: [prod_s, bind, deg_s, unbind, prod_m, deg_m, slide_left, slide_right]
    stoichiometry_matrix = np.array([
        [ 1, -1, -1,  1,  0,  0,  0,  0], 
        [ 0,  1,  0, -1,  0,  0,  0,  0], 
        [ 0,  0,  0,  0,  1, -1,  0,  0]  
    ], dtype=np.int32)
    
    times = [0.0]
    bulk_states = [state.copy()]                  
    spatial_states = [nucleosome_array.copy()]    

    while t < t_max:
        sox2_free, sox2_bound, mrna_count = state
        
        # Calculate Bulk Propensities
        unbound_nucleosomes = np.sum(nucleosome_array == 0)
        bound_nucleosomes = np.sum(nucleosome_array == 1)
        # can_slide_right[i] is True if site i is Bound (1) AND site i+1 is Unbound (0)
        can_slide_right = (nucleosome_array[:-1] == 1) & (nucleosome_array[1:] == 0)
        # can_slide_left[i] is True if site i+1 is Bound (1) AND site i is Unbound (0)
        can_slide_left = (nucleosome_array[1:] == 1) & (nucleosome_array[:-1] == 0)
        hop_probability = (1/z) * e**-(i: j/ lambda)
        
        num_slide_right = np.sum(can_slide_right)
        num_slide_left = np.sum(can_slide_left)
        propensities = np.array([
            k_prod_s,                                 # 0: prod_s
            k_bind * sox2_free * unbound_nucleosomes, # 1: bind
            k_deg_s * sox2_free,                      # 2: deg_s
            k_unbind * bound_nucleosomes,             # 3: unbind
            k_prod_m * sox2_bound,                    # 4: prod_m
            k_deg_m * mrna_count,                     # 5: deg_m
            k_slide * num_slide_left,                 # 6: slide_left
            k_slide * num_slide_right,                 # 7: slide_right
            k_unbind * 
        ])
        
        total_prop = np.sum(propensities)
        if total_prop == 0:
            break
            
        # Gillespie Core
        r1, r2 = np.random.random(2)
        tau = (1.0 / total_prop) * np.log(1.0 / r1)
        t += tau
        
        cumulative_props = np.cumsum(propensities)
        reaction_index = np.searchsorted(cumulative_props, r2 * total_prop)
        state += stoichiometry_matrix[:, reaction_index]
        
        # Spatial Array Updates
        if reaction_index == 1: # Bind
            available = np.where(nucleosome_array == 0)[0]
            nucleosome_array[np.random.choice(available)] = 1
            
        elif reaction_index == 3: # Unbind
            bound = np.where(nucleosome_array == 1)[0]
            nucleosome_array[np.random.choice(bound)] = 0
            
        elif reaction_index == 6: # Slide Left
            # Find an index 'i' where site i+1 is Bound and site i is Unbound
            valid_indices = np.where(can_slide_left)[0]
            target = np.random.choice(valid_indices)
            nucleosome_array[target + 1] = 0 # Protein leaves
            nucleosome_array[target] = 1     # Protein arrives
            
        elif reaction_index == 7: # Slide Right
            # Find an index 'i' where site i is Bound and site i+1 is Unbound
            valid_indices = np.where(can_slide_right)[0]
            target = np.random.choice(valid_indices)
            nucleosome_array[target] = 0     # Protein leaves
            nucleosome_array[target + 1] = 1 # Protein arrives
        
        # Record history
        times.append(t)
        bulk_states.append(state.copy())
        spatial_states.append(nucleosome_array.copy())
        
    return times, np.array(bulk_states), np.array(spatial_states)

t_max = 500.0      
sox2_model_parameters = [1, 0, 0]
sox2_model_variables = [0, 0.0, 1, 0.01, 1, 0.1, 1]
#k_prod_s, k_deg_s, k_prod_m, k_deg_m, k_bind, k_unbind, k_slide

times, bulk_states, spatial_states = simulate_sox2_with_sliding(sox2_model_parameters, sox2_model_variables, t_max)
times_array = np.array(times)
bulk_history = np.array(bulk_states)     
spatial_history = np.array(spatial_states) 
plot_nucleosome_occupancy_history(spatial_history)