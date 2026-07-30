"""Heterodimer model: distinct SOX2 and NANOG sites on the same promoter.

State vector is ``[n00, n10, n01, n11, y]`` -- the four promoter occupancy
compartments and mRNA. Site s flips at (alpha_s, beta_s), site n at
(alpha_n, beta_n); mRNA is made whenever at least one site is bound.
"""

from stochtf.ssa.params import heterodimer_params

params, initial_state, stoichiometry = heterodimer_params

#: Indices of the state vector that count as promoter occupancy.
PROMOTER_IDX = [1, 2, 3]
#: Index of the mRNA species.
MRNA_IDX = 4


def propensity_fn(state, p_params):
    n00, n10, n01, n11, y = state
    alpha_s = p_params["alpha_s"]
    beta_s = p_params["beta_s"]
    alpha_n = p_params["alpha_n"]
    beta_n = p_params["beta_n"]
    k_y = p_params["k_y"]
    gamma_y = p_params["gamma_y"]

    prop_bind_S_empty = alpha_s * n00
    prop_bind_N_empty = alpha_n * n00
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


MODEL = (params, initial_state, stoichiometry, propensity_fn, PROMOTER_IDX, MRNA_IDX)
