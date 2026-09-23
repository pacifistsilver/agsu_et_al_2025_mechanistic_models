from stochtf.ssa.params import homodimer_params

params, initial_state, stoichiometry = homodimer_params

PROMOTER_IDX = [1, 2, 3]
MRNA_IDX = 4

def propensity_fn(state, p_params):
    n00, n10, n01, n11, y = state
    alpha_n = p_params["alpha_n"]
    beta_n = p_params["beta_n"]
    k_y = p_params["k_y"]
    gamma_y = p_params["gamma_y"]

    prop_bind_n01 = alpha_n * n00
    prop_bind_n10 = alpha_n * n00
    prop_unbind_n01 = beta_n * n01
    prop_unbind_n10 = beta_n * n10
    prop_bind_N_n01_n11 = alpha_n * n01
    prop_bind_N_n10_n11 = alpha_n * n10
    prop_unbind_N_n11 = beta_n * n11

    prop_transcription = k_y * (n10 + n01 + n11)
    prop_degradation = gamma_y * y

    return [
        prop_bind_n10,
        prop_bind_n01,
        prop_unbind_n10,
        prop_unbind_n01,
        prop_bind_N_n10_n11,
        prop_bind_N_n01_n11,
        prop_unbind_N_n11,
        prop_unbind_N_n11,
        prop_transcription,
        prop_degradation,
    ]


MODEL = (params, initial_state, stoichiometry, propensity_fn, PROMOTER_IDX, MRNA_IDX)
