from main import SimulationResultHandler, ClusterModel
import itertools
import numpy as np
import polars as pl
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
def plot_all_k1_heatmaps():
    # 1. Define parameter ranges 
    S_vals = np.linspace(1.0, 100.0, 20)      # Free SOX2: 1.0 to 100.0
    K1_vals = np.linspace(0.1, 1.0, 10)       # Affinity: 10 distinct values
    N_vals = np.arange(1, 21)                 # Binding sites: 1 to 20
    
    # 2. Generate Data
    print("Generating data...")
    df = SimulationResultHandler.generate_thermodynamic_expectation_rho(S_vals, N_vals, K1_vals)
    
    rates = ['thermo_maximal_rate', 'thermo_linear_rate', 'thermo_saturating_rate']
    rate_titles = ['Maximal Rate', 'Linear Rate', 'Saturating Rate']
    
    # Calculate grid dimensions based on the number of K1 values
    n_k1 = len(K1_vals)
    ncols = 5 # Set to 5 columns max per row
    nrows = int(np.ceil(n_k1 / ncols))
    
    # Loop through each rate and create a separate figure
    for rate, title in zip(rates, rate_titles):
        
        # Create a dynamically sized grid (e.g., 2x5 for 10 values)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        fig.suptitle(f'{title} (N vs S) across ALL Affinities (K1)', fontsize=20, y=1.02)
        
        # Flatten the axes array to easily iterate over it in a 1D loop
        axes_flat = axes.flatten()
        
        for idx, target_K1 in enumerate(K1_vals):
            ax = axes_flat[idx]
            
            # Slice the dataframe for the current K1 value
            df_slice = df[np.isclose(df['K1'], target_K1)]
            
            # Pivot: S on Y-axis (index), N on X-axis (columns)
            pivot_df = df_slice.pivot(index='S', columns='N', values=rate)
            pivot_df.index = np.round(pivot_df.index, 1)
            
            # Show colorbar only on the rightmost column of each row
            show_cbar = (idx % ncols == ncols - 1) or (idx == n_k1 - 1)
            
            # Plot the heatmap
            sns.heatmap(pivot_df, cmap='viridis', ax=ax, cbar=show_cbar, 
                        cbar_kws={'label': 'Rate'} if show_cbar else None)
            
            # Formatting titles and axes
            ax.set_title(f'K1 ≈ {target_K1:.2f}', fontsize=12)
            
            # Label Y-axes only for the leftmost plots in the grid
            ax.set_ylabel('Free SOX2 (S)' if idx % ncols == 0 else '')
            # Label X-axes only for the bottom row plots in the grid
            ax.set_xlabel('Binding Sites (N)' if idx >= n_k1 - ncols else '')
            
            # Invert Y-axis so larger S values appear at the top
            ax.invert_yaxis() 
        
        # If your K1_vals aren't perfectly divisible by 5, hide the empty subplots
        for idx in range(n_k1, len(axes_flat)):
            fig.delaxes(axes_flat[idx])
            
        plt.tight_layout()
        plt.show()

def run_single_simulation(args):
    """Runs a single instance of the ClusterModel and returns the steady-state rate."""
    S, N, K1 = args
    
    # 1. Map parameters to the simulation inputs
    # Assuming S represents the initial amount of free SOX2
    model_param = {
        "sox2_free": int(S), 
        "sox2_bound": 0, 
        "mrna_count": 0
    }
    
    # Assuming K1 is represented by k_bind/k_unbind. We fix unbind to 1.0 and scale bind.
    # Set k_prod_m to 1.0 to get a normalized production rate.
    model_var = {
        "k_prod_s": 0.0, 
        "k_deg_s": 0.0, 
        "k_prod_m": 1.0, 
        "k_deg_m": 0.1, 
        "k_bind": K1, 
        "k_unbind": 0.1, 
        "k_hop": 0.0, 
        "extra": 0.0
    }
    
    # 2. Initialize and run model (using a shorter t_max for grid scanning)
    sim_max_time = 200
    model = ClusterModel(
        model_param=model_param, 
        model_var=model_var, 
        model_binding_sites=int(N), 
        sim_max_time=sim_max_time, 
        record_interval=1.0, 
        track_history=False # Turn off heavy tracking for mass grid execution
    )
    
    param_df, _, _ = model.run_sox2_sim()
    
    # 3. Calculate the average simulated rate (ignoring the first 25% of time as burn-in)
    burn_in_cutoff = sim_max_time * 0.25
    steady_state_df = param_df.filter(pl.col("time") > burn_in_cutoff)
    
    if len(steady_state_df) > 0:
        # Rate = average bound SOX2 * k_prod_m
        simulated_rate = steady_state_df["sox2_bound"].mean() * model_var["k_prod_m"]
    else:
        simulated_rate = 0.0
        
    return {"S": S, "N": int(N), "K1": K1, "simulated_rate": simulated_rate}

def generate_simulation_grid(S_vals, N_vals, K1_vals):
    """Generates the grid of parameters and runs them in parallel."""
    tasks = list(itertools.product(S_vals, N_vals, K1_vals))
    results = []
    
    print(f"Starting {len(tasks)} simulations in parallel...")
    
    # Use max workers available on your CPU
    with ProcessPoolExecutor() as executor:
        for result in executor.map(run_single_simulation, tasks):
            results.append(result)
            
    print("Simulations complete!")
    return pl.DataFrame(results).to_pandas()

def plot_simulated_k1_heatmaps():
    # 1. Define parameter ranges 
    # NOTE: You may want to reduce the resolution (e.g., to 10) if the simulation takes too long
    S_vals = np.arange(1, 50, 50)      # Free SOX2: 1.0 to 100.0
    K1_vals = np.linspace(0.1, 1.0, 10)       # Affinity: 10 distinct values
    N_vals = np.arange(1, 10)                 # Binding sites: 1 to 20
    
    # 2. Run Parallel Simulations
    df = generate_simulation_grid(S_vals, N_vals, K1_vals)
    
    # 3. Calculate grid dimensions based on the number of K1 values
    n_k1 = len(K1_vals)
    ncols = 5 # Set to 5 columns max per row
    nrows = int(np.ceil(n_k1 / ncols))
    
    # Create the figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    fig.suptitle('Simulated mRNA Production Rate (N vs S) across Affinities (K1)', fontsize=20, y=1.02)
    
    axes_flat = axes.flatten()
    
    for idx, target_K1 in enumerate(K1_vals):
        ax = axes_flat[idx]
        
        # Slice the dataframe for the current K1 value
        df_slice = df[np.isclose(df['K1'], target_K1)]
        
        # Pivot: S on Y-axis (index), N on X-axis (columns)
        pivot_df = df_slice.pivot(index='S', columns='N', values='simulated_rate')
        pivot_df.index = np.round(pivot_df.index, 1)
        
        # Show colorbar only on the rightmost column of each row
        show_cbar = (idx % ncols == ncols - 1) or (idx == n_k1 - 1)
        
        # Plot the heatmap
        sns.heatmap(pivot_df, cmap='magma', ax=ax, cbar=show_cbar, 
                    cbar_kws={'label': 'Simulated Rate'} if show_cbar else None)
        
        # Formatting titles and axes
        ax.set_title(f'K1 ≈ {target_K1:.2f}', fontsize=12)
        
        # Label Y-axes only for the leftmost plots in the grid
        ax.set_ylabel('Initial Free SOX2 (S)' if idx % ncols == 0 else '')
        # Label X-axes only for the bottom row plots in the grid
        ax.set_xlabel('Binding Sites (N)' if idx >= n_k1 - ncols else '')
        
        # Invert Y-axis so larger S values appear at the top
        ax.invert_yaxis() 
    
    # Hide empty subplots if K1_vals length is not a multiple of ncols
    for idx in range(n_k1, len(axes_flat)):
        fig.delaxes(axes_flat[idx])
        
    plt.tight_layout()
    plt.show()
    
def run_1d_simulation_slice(args):
    """Runs a single stochastic simulation to extract mean AND standard deviation."""
    S, N, K1 = args
    
    model_param = {"sox2_free": int(S), "sox2_bound": 0, "mrna_count": 0}
    model_var = {
        "k_prod_s": 0.0, "k_deg_s": 0.0, 
        "k_prod_m": 1.0, "k_deg_m": 0.1, 
        "k_bind": K1, "k_unbind": 1.0, 
        "k_hop": 0.0, "extra": 0.0
    }
    
    # Use a longer simulation time to get smooth, reliable standard deviations
    sim_max_time = 500  
    model = ClusterModel(
        model_param=model_param, 
        model_var=model_var, 
        model_binding_sites=int(N), 
        sim_max_time=sim_max_time, 
        record_interval=1.0, 
        track_history=False
    )
    
    param_df, _, _ = model.run_sox2_sim()
    
    # Discard the first 25% of the simulation to allow the system to reach steady-state
    burn_in_cutoff = sim_max_time * 0.25
    steady_state_df = param_df.filter(pl.col("time") > burn_in_cutoff)
    
    if len(steady_state_df) > 0:
        # Convert to numpy for easy math
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
    
    # 2. Setup Tasks
    tasks = [(S, N, K1) for N in target_Ns for K1 in target_K1s for S in S_vals]
                
    print(f"Running {len(tasks)} stochastic simulations for the 3x3 grid...")
    sim_results = []
    
    with ProcessPoolExecutor() as executor:
        for result in executor.map(run_1d_simulation_slice, tasks):
            sim_results.append(result)
            
    sim_df = pd.DataFrame(sim_results)
    
    # 3. Get Theoretical Data
    print("Calculating theoretical thermodynamic rates...")
    thermo_df = SimulationResultHandler.generate_thermodynamic_expectation_rho(S_vals, target_Ns, target_K1s)
    
    merged_df = pd.merge(sim_df, thermo_df, on=["S", "N", "K1"])
    
    # 4. Tufte Styling Setup
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, axes = plt.subplots(nrows=len(target_K1s), ncols=len(target_Ns), figsize=(12, 10), sharex=True, sharey='row')
    fig.suptitle('Steady-State mRNA Production Rate vs. Free SOX2', fontsize=16, fontweight='light', y=0.98)
    
    for row_idx, K1 in enumerate(target_K1s):
        for col_idx, N in enumerate(target_Ns):
            ax = axes[row_idx, col_idx]
            subset = merged_df[(np.isclose(merged_df['K1'], K1)) & (merged_df['N'] == N)]
            
            # --- PREVENT NEGATIVE ERROR BARS ---
            # Lower error cannot exceed the mean itself (clips at 0)
            lower_error = np.minimum(subset["sim_mean"], subset["sim_std"])
            upper_error = subset["sim_std"]
            asymmetric_error = [lower_error, upper_error]
            
            # --- PLOTTING ---
            # Plot Stochastic Simulation points (Tufte prefers high contrast, simple geometries)
            ax.errorbar(subset["S"], subset["sim_mean"], yerr=asymmetric_error, 
                         fmt='o', color='#2b6bb0', markersize=4, capsize=0, elinewidth=1, alpha=0.8,
                         label="Simulation (Mean ± 1 SD)" if (row_idx==0 and col_idx==0) else "")
            
            # Plot Theoretical Lines (Elegant thin lines)
            ax.plot(subset["S"], subset["thermo_linear_rate"], 
                     label="Rho Expected (Linear)" if (row_idx==0 and col_idx==0) else "", 
                     color='black', linewidth=1.5, zorder=2)
            ax.plot(subset["S"], subset["thermo_saturating_rate"], 
                     label="Rho Expected (Saturating)" if (row_idx==0 and col_idx==0) else "", 
                     color='#888888', linestyle='--', linewidth=1.2, zorder=1)
            
            # --- TUFTE STYLING (Erase non-data ink) ---
            # Remove top and right spines completely
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Make the remaining axes lines light gray
            ax.spines['left'].set_color('#cccccc')
            ax.spines['bottom'].set_color('#cccccc')
            
            # Soften tick parameters
            ax.tick_params(axis='both', colors='#555555', length=4, width=0.5)
            
            # Subplot titles acting as labels instead of heavy boxes
            ax.set_title(f'Sites = {N} | Ka = {K1} | Kd = 1.0', fontsize=11, color='#333333', loc='left')
            
            # Minimalist horizontal grid lines only
            ax.yaxis.grid(True, linestyle='-', color='#eeeeee', linewidth=0.7)
            ax.xaxis.grid(False)
            
            # Axis labels only on the outer edges to maximize data space
            if row_idx == len(target_K1s) - 1:
                ax.set_xlabel('Bulk SOX2', fontsize=11, color='#333333', labelpad=8)
            if col_idx == 0:
                ax.set_ylabel('k1', fontsize=11, color='#333333', labelpad=8)
                
            # Place a clean legend without a box
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace=0.15, hspace=0.3) 
    plt.show()

if __name__ == "__main__":
    plot_1d_cross_section_grid_tufte()