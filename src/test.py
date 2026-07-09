import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def simulate_heterodimer_bursts(alpha_s, beta_s, alpha_n, beta_n, k_y, n_bursts):
    """
    Simulates the heterodimer model phenomenologically.
    It calculates the exact time spent meandering in the active states (tau_on),
    then draws the burst size from a Poisson distribution at the end.
    """
    burst_records = []
    
    # Pre-calculate OFF state exit rate
    rate_leave_off = alpha_s + alpha_n
    
    for _ in range(n_bursts):
        # ==========================================
        # 1. THE OFF STATE
        # ==========================================
        # Draw the time spent in (0,0)
        tau_off = np.random.exponential(1.0 / rate_leave_off)
        
        # Determine which state initiates the burst
        prob_s_binds_first = alpha_s / rate_leave_off
        current_state = 1 if np.random.rand() < prob_s_binds_first else 2
        
        # ==========================================
        # 2. THE ON STATE (The Meandering Timer)
        # ==========================================
        tau_on = 0.0
        
        # Loop continues until the system falls back to state 0 (0,0)
        while current_state != 0:
            if current_state == 1: # State (1,0): Sox2 Bound
                rate_leave = beta_s + alpha_n
                tau_on += np.random.exponential(1.0 / rate_leave)
                # Next state: Unbind to (0,0) or Bind Nanog to (1,1)
                current_state = 0 if np.random.rand() < (beta_s / rate_leave) else 3
                
            elif current_state == 2: # State (0,1): Nanog Bound
                rate_leave = beta_n + alpha_s
                tau_on += np.random.exponential(1.0 / rate_leave)
                # Next state: Unbind to (0,0) or Bind Sox2 to (1,1)
                current_state = 0 if np.random.rand() < (beta_n / rate_leave) else 3
                
            elif current_state == 3: # State (1,1): Both Bound
                rate_leave = beta_s + beta_n
                tau_on += np.random.exponential(1.0 / rate_leave)
                # Next state: Sox2 unbinds to (0,1) or Nanog unbinds to (1,0)
                current_state = 2 if np.random.rand() < (beta_s / rate_leave) else 1

        # ==========================================
        # 3. PHENOMENOLOGICAL BURST DRAW
        # ==========================================
        # Draw the total burst size based on the total time spent ON
        burst_size = np.random.poisson(1/k_y * tau_on)
        
        # Calculate macroscopic cycle metrics
        cycle_time = tau_off + tau_on
        burst_frequency = 1.0 / cycle_time
        
        burst_records.append({
            'Burst Size': burst_size,
            'Burst Frequency': burst_frequency,
            'Tau ON': tau_on,
            'Tau OFF': tau_off
        })

    return pd.DataFrame(burst_records)

# ==========================================
# Execution and Plotting
# ==========================================

# 1. Define Model Parameters
alpha_s, beta_s = 0.5, 0.06   # Sox2 rates
alpha_n, beta_n = 0.2, 0.2  # Nanog rates
k_y = 1.0                   # Transcription rate 
n_bursts = 10000             # Number of bursts to simulate

# 2. Run Simulation
df = simulate_heterodimer_bursts(alpha_s, beta_s, alpha_n, beta_n, k_y, n_bursts)

# 3. Plotting with Seaborn
sns.set_theme(style="whitegrid", context="talk")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Burst Size Distribution
sns.histplot(data=df, x='Burst Size', discrete=True, color='#9b59b6', ax=axes[0])
mean_burst = df['Burst Size'].mean()
axes[0].axvline(mean_burst, color='black', linestyle='--', linewidth=2.5, label=f'Simulated Mean: {mean_burst:.1f}')
axes[0].set_title('Heterodimer Burst Size Distribution')
axes[0].set_xlabel('mRNA Transcripts per Burst')
axes[0].set_ylabel('Count')
axes[0].legend()

# Plot 2: Burst Frequency Distribution
sns.histplot(data=df, x='Burst Frequency', kde=True, bins=40, color='#e67e22', ax=axes[1])
mean_freq = df['Burst Frequency'].mean()
axes[1].axvline(mean_freq, color='black', linestyle='--', linewidth=2.5, label=f'Simulated Mean: {mean_freq:.3f}')
axes[1].set_title('Heterodimer Burst Frequency')
axes[1].set_xlabel('Frequency ($1 / Cycle Time$)')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
plt.show()