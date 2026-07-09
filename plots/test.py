import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def demonstrate_burst_kinetics(n_bursts, v, mean_off, mean_on):
    """
    Demonstrates the math of Sections 1.1 and 1.2 by sampling 
    directly from the underlying probability distributions.
    """
    # 1. Sample the underlying dwell times (assuming exponential distributions for CTM)
    # This represents the stochastic "Timer"
    t_off_samples = np.random.exponential(scale=mean_off, size=n_bursts)
    t_on_samples = np.random.exponential(scale=mean_on, size=n_bursts)
    
    # --- Section 1.1: Burst Frequency ---
    # Cycle time is the sum of the random variables (Convolution in probability)
    cycle_times = t_off_samples + t_on_samples
    
    # Theoretical vs Simulated Mean Cycle Time
    theoretical_cycle_time = mean_off + mean_on
    simulated_cycle_time = np.mean(cycle_times)
    
    theoretical_freq = 1 / theoretical_cycle_time
    simulated_freq = 1 / simulated_cycle_time
    
    # --- Section 1.2: Burst Size ---
    # The "Printer": We draw from a Poisson distribution where the mean is v * tau
    # Crucially, tau is our array of stochastically sampled ON times
    burst_sizes = np.random.poisson(lam=v * t_on_samples)
    
    theoretical_burst_size = v * mean_on
    simulated_burst_size = np.mean(burst_sizes)
    
    # Compile results for visualization
    df = pd.DataFrame({
        'Cycle Time': cycle_times,
        'Burst Size': burst_sizes
    })
    
    print("--- Section 1.1: Burst Frequency ---")
    print(f"Theoretical Frequency: {theoretical_freq:.4f}")
    print(f"Simulated Frequency:   {simulated_freq:.4f}\n")
    
    print("--- Section 1.2: Burst Size ---")
    print(f"Theoretical Mean Burst Size: {theoretical_burst_size:.2f}")
    print(f"Simulated Mean Burst Size:   {simulated_burst_size:.2f}\n")
    
    # Visualization
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.histplot(df['Cycle Time'], kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Distribution of Cycle Times')
    axes[0].axvline(simulated_cycle_time, color='red', linestyle='--', label=f'Mean: {simulated_cycle_time:.2f}')
    axes[0].legend()
    
    sns.histplot(df['Burst Size'], discrete=True, ax=axes[1], color='green')
    axes[1].set_title('Distribution of Burst Sizes')
    axes[1].axvline(simulated_burst_size, color='red', linestyle='--', label=f'Mean: {simulated_burst_size:.2f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# Run the demonstration with sample parameters
demonstrate_burst_kinetics(n_bursts=10000, v=15.0, mean_off=20.0, mean_on=5.0)