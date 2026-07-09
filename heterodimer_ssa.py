import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath("./src"))
import gillespie as gil
from ssa_params import heterodimer_params

params, initial_state, stoichiometry = heterodimer_params


def propensity_fn(state, p_params):
    n00, n10, n01, n11, y = state
    alpha_s = p_params["alpha_s"]
    beta_s = p_params["beta_s"]
    alpha_n = p_params["alpha_n"]
    beta_n = p_params["beta_n"]
    k_y = p_params["k_y"]
    gamma_y = p_params["gamma_y"]

    prop_bind_S_empty = alpha_s  * n00
    prop_bind_N_empty = alpha_n  * n00
    prop_unbind_S_n10 = beta_s * n10
    prop_unbind_N_n01 = beta_n * n01

    # Secondary bindings to form n11
    prop_bind_N_n10 = alpha_n * n10 
    prop_bind_S_n01 = alpha_s * n01 

    # Unbindings from n11
    prop_unbind_N_n11 = beta_n * n11
    prop_unbind_S_n11 = beta_s * n11

    # Transcription and degradation
    prop_transcription = k_y * (n10 + n01 + n11)
    prop_degradation = gamma_y * y

    return [
        prop_bind_S_empty,
        prop_bind_N_empty,
        prop_unbind_S_n10,
        prop_unbind_N_n01,
        prop_bind_N_n10,
        prop_bind_S_n01,
        prop_unbind_N_n11,
        prop_unbind_S_n11,
        prop_transcription,
        prop_degradation,
    ]


if __name__ == "__main__":
    t_max = 100
    print("Running Gillespie simulation for n10, n01, n11 model...")
    times, states = gil.gillespie(
        initial_state,
        stoichiometry,
        propensity_fn,
        t_max=t_max,
        parameters=params,
        max_steps=50000,
    )
    
    promoter_idx = [1, 2, 3]
    active_start = 1
    active_end = 200
    on_times, off_times = gil.extract_on_off(times, states, promoter_idx, active_start, active_end)
    print(f"Mean ON time: {on_times.mean()}")
    print(f"Mean OFF time: {off_times.mean()}")

    mrna = states[:, 4]
    mean_mrna = np.mean(mrna)
    fano_mrna = np.var(mrna) / mean_mrna if mean_mrna > 0 else 0
    print(f"Mean mRNA: {mean_mrna:.2f}")
    print(f"Fano factor: {fano_mrna:.2f}")

    
    print("Simulation complete!")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))

    ax1.step(
        times, states[:, 3], label="($n_{11}$)", color="purple", alpha=0.7
    )
    ax1.set_ylabel("Bound TFs")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.step(times, states[:, 4], label="mRNA ($y$)", color="red")
    ax2.set_ylabel("mRNA Count")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Histogram for mRNA
    mrna_max = int(np.max(states[:, 4]))
    bins = np.arange(mrna_max + 2) - 0.5
    ax3.hist(states[:, 4], bins=bins, density=True, color="gray", edgecolor="black", alpha=0.7)
    ax3.set_xlabel("mRNA Count")
    ax3.set_ylabel("Probability Density")
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    # Save the plot
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "cme_gillespie_heterodimer_simulation.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")
