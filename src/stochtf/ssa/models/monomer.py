"""Monomer model: SOX2 and NANOG bind independently, neither dimerises.

State vector is ``[nf, sf, nb, sb, y]`` -- free and bound NANOG, free and bound
SOX2, and mRNA.
"""

from stochtf.ssa.params import monomer_params

params, initial_state, stoichiometry = monomer_params

#: Indices of the state vector that count as promoter occupancy.
PROMOTER_IDX = [2, 3]
#: Index of the mRNA species.
MRNA_IDX = 4


def propensity_fn(state, p_params):
    nf, sf, nb, sb, y = state
    alpha_s = p_params["alpha_s"]
    beta_s = p_params["beta_s"]
    alpha_n = p_params["alpha_n"]
    beta_n = p_params["beta_n"]
    k_y = p_params["k_y"]
    gamma_y = p_params["gamma_y"]

    prop_bind_s = alpha_s * sf
    prop_unbind_s = beta_s * sb
    prop_bind_n = alpha_n * nf
    prop_unbind_n = beta_n * nb

    prop_transcription = k_y * (nb + sb)
    prop_degradation = gamma_y * y

    return [
        prop_bind_s,
        prop_unbind_s,
        prop_bind_n,
        prop_unbind_n,
        prop_transcription,
        prop_degradation,
    ]


MODEL = (params, initial_state, stoichiometry, propensity_fn, PROMOTER_IDX, MRNA_IDX)
