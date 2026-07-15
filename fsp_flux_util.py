import numpy as np
import pylab
import os

def plot_probability_flux(model, p, domain_states, proj_x_func, proj_x_delta_func, proj_y_func, proj_y_delta_func, title="Probability Flux Vector Field"):
    """
    Computes and plots the probability flux vector field.
    
    Parameters:
    - model: The CME model dictionary
    - p: 1D array of probabilities for each state in the domain
    - domain_states: 2D array (num_species, N) of states in the domain
    - proj_x_func: Function to map domain_states -> X coordinates (1D array)
    - proj_x_delta_func: Function to map a transition vector -> Delta X
    - proj_y_func: Function to map domain_states -> Y coordinates (1D array)
    - proj_y_delta_func: Function to map a transition vector -> Delta Y
    """
    
    X_coords = proj_x_func(domain_states).astype(int)
    Y_coords = proj_y_func(domain_states).astype(int)
    
    max_X = np.max(X_coords) if len(X_coords) > 0 else 0
    max_Y = np.max(Y_coords) if len(Y_coords) > 0 else 0
    
    V_x = np.zeros((max_X + 1, max_Y + 1))
    V_y = np.zeros((max_X + 1, max_Y + 1))
    
    transitions = model['transitions']
    propensities = model['propensities']
    
    # Convert distribution dictionary to a 1D numpy array aligned with domain_states
    N = domain_states.shape[1]
    p_array = np.zeros(N)
    for j in range(N):
        state_tuple = tuple(domain_states[:, j])
        p_array[j] = p.get(state_tuple, 0.0)
    
    for r, (trans, prop_fn) in enumerate(zip(transitions, propensities)):
        # Calculate propensities for all states by unpacking the rows
        try:
            A_r = prop_fn(*domain_states)
        except TypeError:
            # If it requires time 't'
            A_r = prop_fn(*domain_states, 0.0) 
            
        # A_r is now exactly 1D (N,)
        
        # Flux out of each state
        F_r = p_array * A_r
        
        dX = proj_x_delta_func(trans)
        dY = proj_y_delta_func(trans)
        
        if dX == 0 and dY == 0:
            continue
            
        # Accumulate flux on the grid using numpy operations for speed
        for j in range(N):
            x = X_coords[j]
            y = Y_coords[j]
            V_x[x, y] += F_r[j] * dX
            V_y[x, y] += F_r[j] * dY
            
    # Plotting
    X_grid, Y_grid = np.meshgrid(np.arange(max_X + 1), np.arange(max_Y + 1), indexing='ij')
    
    pylab.figure(figsize=(10, 8))
    
    # We can also plot a marginal probability heatmap as background
    P_marginal = np.zeros((max_X + 1, max_Y + 1))
    for j in range(N):
        x = X_coords[j]
        y = Y_coords[j]
        P_marginal[x, y] += p_array[j]
        
    # Mask zeros to not plot them in imshow
    P_marginal_plot = np.ma.masked_where(P_marginal == 0, P_marginal)
    pylab.imshow(P_marginal_plot.T, origin='lower', cmap='Blues', aspect='auto', alpha=0.8)
    pylab.colorbar(label='Probability')
    
    # Plot quiver
    # Mask zero vectors for cleaner plot
    magnitude = np.sqrt(V_x**2 + V_y**2)
    mask = magnitude > (np.max(magnitude) * 1e-4) # Filter out floating point noise
    
    if np.any(mask):
        pylab.quiver(X_grid[mask], Y_grid[mask], V_x[mask], V_y[mask], 
                     color='red', angles='xy', scale=None, width=0.003)
    
    pylab.xlabel("Total Bound TFs")
    pylab.ylabel("mRNA Count")
    pylab.title(title)
    pylab.tight_layout()
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    pylab.savefig(os.path.join(plots_dir, "flux_vector_field.png"), dpi=300)
    print("Saved plot to " + os.path.join(plots_dir, "flux_vector_field.png"))
