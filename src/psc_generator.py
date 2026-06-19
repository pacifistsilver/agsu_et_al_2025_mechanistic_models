import numpy as np
import os

def generate_psc(num_sites, promoter_site=None, filename="spatial_chromatin.psc"):
    if promoter_site is None:
        promoter_site = int((num_sites - 1) / 2)
        
    site_indices = np.arange(num_sites)
    potential_dimer_partners = np.abs(site_indices[:, None] - site_indices[None, :])
    weighted_potential_dimer_partners = np.exp(-potential_dimer_partners / 1)
    weight_matrix = np.ones((num_sites, num_sites))
    np.fill_diagonal(weight_matrix, 0.0)
    
    row_sums = weight_matrix.sum(axis=1)
    raw_weight_matrix_row_sum = np.max(row_sums) if len(row_sums) > 0 else 1.0
    W = weight_matrix / raw_weight_matrix_row_sum
    
    lines = ["# Spatial Chromatin Model Dynamically Generated for StochPy"]
    lines.append("")
    lines.append("# --- Reactions ---")
    lines.append("")
    
    rxn_idx = 1
    
    lines.append(f"R{rxn_idx}:")
    lines.append("    $pool > S_free")
    lines.append("    k_s_in")
    rxn_idx += 1
    
    lines.append(f"R{rxn_idx}:")
    lines.append("    $pool > N_free")
    lines.append("    k_n_in")
    rxn_idx += 1
    
    lines.append(f"R{rxn_idx}:")
    lines.append("    S_free > $pool")
    lines.append("    k_s_out * S_free")
    rxn_idx += 1
    
    lines.append(f"R{rxn_idx}:")
    lines.append("    N_free > $pool")
    lines.append("    k_n_out * N_free")
    rxn_idx += 1
    
    # 2. Bulk Dimerisation
    lines.append(f"R{rxn_idx}:")
    lines.append("    S_free + N_free > SN_free")
    lines.append("    k_dimerise * S_free * N_free")
    rxn_idx += 1
    
    lines.append(f"R{rxn_idx}:")
    lines.append("    SN_free > S_free + N_free")
    lines.append("    k_dissociate * SN_free")
    rxn_idx += 1
    
    lines.append(f"R{rxn_idx}:")
    lines.append("    N_free + N_free > NN_free")
    lines.append("    k_dimerise * 0.5 * N_free * (N_free - 1)")
    rxn_idx += 1
    
    lines.append(f"R{rxn_idx}:")
    lines.append("    NN_free > N_free + N_free")
    lines.append("    k_dissociate * NN_free")
    rxn_idx += 1
    
    # Site specific reactions
    for i in range(num_sites):
        # Monomer Binding / Unbinding
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_E + S_free > S_{i}_S")
        lines.append(f"    k_bind_s * S_{i}_E * S_free")
        rxn_idx += 1
        
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_S > S_{i}_E + S_free")
        lines.append(f"    k_unbind_s * S_{i}_S")
        rxn_idx += 1
        
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_E + N_free > S_{i}_N")
        lines.append(f"    k_bind_n * S_{i}_E * N_free")
        rxn_idx += 1
        
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_N > S_{i}_E + N_free")
        lines.append(f"    k_unbind_n * S_{i}_N")
        rxn_idx += 1
        
        # Dimer Binding / Unbinding (from Bulk)
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_E + SN_free > S_{i}_SNf")
        lines.append(f"    k_bind_s * S_{i}_E * SN_free")
        rxn_idx += 1
        
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_SNf > S_{i}_E + SN_free")
        lines.append(f"    k_unbind_s * S_{i}_SNf")
        rxn_idx += 1
        
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_E + SN_free > S_{i}_NSf")
        lines.append(f"    k_bind_n * S_{i}_E * SN_free")
        rxn_idx += 1
        
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_NSf > S_{i}_E + SN_free")
        lines.append(f"    k_unbind_n * S_{i}_NSf")
        rxn_idx += 1
        
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_E + NN_free > S_{i}_NNf")
        lines.append(f"    2.0 * k_bind_n * S_{i}_E * NN_free")
        rxn_idx += 1
        
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_NNf > S_{i}_E + NN_free")
        lines.append(f"    k_unbind_n * S_{i}_NNf")
        rxn_idx += 1
        
        # On-Site Dimerisation (Dangling arm formation and dissociation)
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_S + N_free > S_{i}_SNf")
        lines.append(f"    k_dimerise * S_{i}_S * N_free")
        rxn_idx += 1
        
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_SNf > S_{i}_S + N_free")
        lines.append(f"    k_dissociate * S_{i}_SNf")
        rxn_idx += 1
        
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_N + S_free > S_{i}_NSf")
        lines.append(f"    k_dimerise * S_{i}_N * S_free")
        rxn_idx += 1
        
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_NSf > S_{i}_N + S_free")
        lines.append(f"    k_dissociate * S_{i}_NSf")
        rxn_idx += 1
        
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_N + N_free > S_{i}_NNf")
        lines.append(f"    k_dimerise * S_{i}_N * N_free")
        rxn_idx += 1
        
        lines.append(f"R{rxn_idx}:")
        lines.append(f"    S_{i}_NNf > S_{i}_N + N_free")
        lines.append(f"    k_dissociate * S_{i}_NNf")
        rxn_idx += 1

    # Spatial Tethering
    for i in range(num_sites):
        for j in range(num_sites):
            if i == j:
                continue
                
            w = W[i, j]
            
            # If we enforce i < j for the tether state names to avoid duplication
            # Tether names: T_{min}_{max}_{Tf_at_min}{Tf_at_max}
            t_min = min(i, j)
            t_max = max(i, j)
            
            # Tethering Monomer + Monomer
            if i < j:
                # S_i_S + S_j_N -> T_i_j_SN
                lines.append(f"R{rxn_idx}:\n    S_{i}_S + S_{j}_N > T_{i}_{j}_SN\n    k_dimerise * {w} * S_{i}_S * S_{j}_N")
                rxn_idx += 1
                lines.append(f"R{rxn_idx}:\n    T_{i}_{j}_SN > S_{i}_S + S_{j}_N\n    k_dissociate * T_{i}_{j}_SN")
                rxn_idx += 1
                
                # S_i_N + S_j_S -> T_i_j_NS
                lines.append(f"R{rxn_idx}:\n    S_{i}_N + S_{j}_S > T_{i}_{j}_NS\n    k_dimerise * {w} * S_{i}_N * S_{j}_S")
                rxn_idx += 1
                lines.append(f"R{rxn_idx}:\n    T_{i}_{j}_NS > S_{i}_N + S_{j}_S\n    k_dissociate * T_{i}_{j}_NS")
                rxn_idx += 1
                
                # S_i_N + S_j_N -> T_i_j_NN
                lines.append(f"R{rxn_idx}:\n    S_{i}_N + S_{j}_N > T_{i}_{j}_NN\n    k_dimerise * {w} * S_{i}_N * S_{j}_N * 2.0")
                rxn_idx += 1
                lines.append(f"R{rxn_idx}:\n    T_{i}_{j}_NN > S_{i}_N + S_{j}_N\n    k_dissociate * T_{i}_{j}_NN")
                rxn_idx += 1
            
            # Tethering Dangling + Vacant
            # i has the dangling arm, j is vacant
            # SOX2b_NANOGf at i + EMPTY at j
            if i < j:
                t_state = f"T_{i}_{j}_SN"
            else:
                t_state = f"T_{j}_{i}_NS"
            lines.append(f"R{rxn_idx}:\n    S_{i}_SNf + S_{j}_E > {t_state}\n    k_bind_n * {w} * S_{i}_SNf * S_{j}_E")
            rxn_idx += 1
            # Note: The untethering of this specific formation (falling off DNA) was handled below. 
            # Wait, no. The forward reaction is Dangling + Vacant > Tether. 
            # The reverse is Tether > Dangling + Vacant. 
            # The Tether > Dangling + Vacant represents the newly bound foot unbinding.
            if i < j:
                # T_i_j_SN loses foot j (NANOG) -> S_i_SNf + S_j_E
                lines.append(f"R{rxn_idx}:\n    {t_state} > S_{i}_SNf + S_{j}_E\n    k_unbind_n * {t_state}")
                rxn_idx += 1
            else:
                # T_j_i_NS loses foot i (NANOG) -> S_i_SNf + S_j_E
                lines.append(f"R{rxn_idx}:\n    {t_state} > S_{i}_SNf + S_{j}_E\n    k_unbind_n * {t_state}")
                rxn_idx += 1
            
            # NANOGb_SOX2f at i + EMPTY at j
            if i < j:
                t_state = f"T_{i}_{j}_NS"
            else:
                t_state = f"T_{j}_{i}_SN"
            lines.append(f"R{rxn_idx}:\n    S_{i}_NSf + S_{j}_E > {t_state}\n    k_bind_s * {w} * S_{i}_NSf * S_{j}_E")
            rxn_idx += 1
            if i < j:
                # T_i_j_NS loses foot j (SOX2) -> S_i_NSf + S_j_E
                lines.append(f"R{rxn_idx}:\n    {t_state} > S_{i}_NSf + S_{j}_E\n    k_unbind_s * {t_state}")
                rxn_idx += 1
            else:
                # T_j_i_SN loses foot i (SOX2) -> S_i_NSf + S_j_E
                lines.append(f"R{rxn_idx}:\n    {t_state} > S_{i}_NSf + S_{j}_E\n    k_unbind_s * {t_state}")
                rxn_idx += 1
                
            # NANOGb_NANOGf at i + EMPTY at j
            if i < j:
                t_state = f"T_{i}_{j}_NN"
            else:
                t_state = f"T_{j}_{i}_NN"
            lines.append(f"R{rxn_idx}:\n    S_{i}_NNf + S_{j}_E > {t_state}\n    k_bind_n * {w} * S_{i}_NNf * S_{j}_E")
            rxn_idx += 1
            if i < j:
                lines.append(f"R{rxn_idx}:\n    {t_state} > S_{i}_NNf + S_{j}_E\n    k_unbind_n * {t_state}")
                rxn_idx += 1
            else:
                lines.append(f"R{rxn_idx}:\n    {t_state} > S_{i}_NNf + S_{j}_E\n    k_unbind_n * {t_state}")
                rxn_idx += 1

    # Transcription
    active_states = []
    active_states.extend([f"S_{promoter_site}_S", f"S_{promoter_site}_N", f"S_{promoter_site}_SNf", f"S_{promoter_site}_NSf", f"S_{promoter_site}_NNf"])
    for i in range(num_sites):
        if i < promoter_site:
            active_states.extend([f"T_{i}_{promoter_site}_SN", f"T_{i}_{promoter_site}_NS", f"T_{i}_{promoter_site}_NN"])
        elif i > promoter_site:
            active_states.extend([f"T_{promoter_site}_{i}_SN", f"T_{promoter_site}_{i}_NS", f"T_{promoter_site}_{i}_NN"])
            
    promoter_active_sum = " + ".join(active_states)
    
    lines.append(f"R{rxn_idx}:")
    lines.append("    $pool > mRNA")
    lines.append(f"    k_prod_m * ({promoter_active_sum})")
    rxn_idx += 1
    
    lines.append(f"R{rxn_idx}:")
    lines.append("    mRNA > $pool")
    lines.append("    k_deg_m * mRNA")
    rxn_idx += 1

    lines.append("")
    lines.append("# --- Initial Conditions ---")
    lines.append("S_free = 0")
    lines.append("N_free = 0")
    lines.append("SN_free = 0")
    lines.append("NN_free = 0")
    lines.append("mRNA = 0")
    for i in range(num_sites):
        lines.append(f"S_{i}_E = 1")
        lines.append(f"S_{i}_S = 0")
        lines.append(f"S_{i}_N = 0")
        lines.append(f"S_{i}_SNf = 0")
        lines.append(f"S_{i}_NSf = 0")
        lines.append(f"S_{i}_NNf = 0")
        
    for i in range(num_sites):
        for j in range(i+1, num_sites):
            lines.append(f"T_{i}_{j}_SN = 0")
            lines.append(f"T_{i}_{j}_NS = 0")
            lines.append(f"T_{i}_{j}_NN = 0")

    lines.append("")
    lines.append("# --- Parameters ---")
    lines.append("k_s_in = 0.0")
    lines.append("k_n_in = 0.0")
    lines.append("k_s_out = 0.0")
    lines.append("k_n_out = 0.0")
    lines.append("k_bind_s = 0.0")
    lines.append("k_unbind_s = 0.0")
    lines.append("k_bind_n = 0.0")
    lines.append("k_unbind_n = 0.0")
    lines.append("k_dimerise = 0.0")
    lines.append("k_dissociate = 0.0")
    lines.append("k_prod_m = 0.0")
    lines.append("k_deg_m = 0.0")

    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"Generated {filename} with {rxn_idx - 1} reactions.")

if __name__ == "__main__":
    generate_psc(10)
