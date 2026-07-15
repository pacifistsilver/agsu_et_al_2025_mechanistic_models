"""
Parameters for stochastic simulations for monomer, heterodimer, and homodimer models.
"""

import gillespie as gil

heterodimer_params = [
    {
        "alpha_s": 0.5, #0.5
        "beta_s": 0.06, # 0.06
        "alpha_n": 0.3, # 0.3
        "beta_n": 0.2, # 0.2
        "k_y": 1,
        "gamma_y": 2,
        "mean_burst_size": 5,
    },
    [1, 0, 0, 0, 0],  # [n00, n10, n01, n11, y]
    [
        [-1, 1, 0, 0, 0],  
        [-1, 0, 1, 0, 0],  
        [1, -1, 0, 0, 0],  
        [1, 0, -1, 0, 0],  
        [0, -1, 0, 1, 0],  
        [0, 0, -1, 1, 0],  
        [0, 1, 0, -1, 0],  
        [0, 0, 1, -1, 0],  
        gil.burst_stoichiometry,  
        [0, 0, 0, 0, -1],  
    ],
]
monomer_params = [
    {
        "alpha_s": 0.5,
        "beta_s": 0.06,
        "alpha_n": 0.3,
        "beta_n": 0.2,
        "k_y": 0.1,
        "gamma_y": 0.5,
        "mean_burst_size": 5,
    },
    [1, 9, 0, 0, 0], # nanogfree, sox2free, nanogbound, sox2bound, y
    [[0, -1, 0, 1, 0], [0, 1, 0, -1, 0], [-1, 0, 1, 0, 0], [1, 0, -1, 0, 0], gil.burst_stoichiometry, [0, 0, 0, 0, -1]],
]


homodimer_params = [
    {
        "alpha_n": 0.3,
        "beta_n": 0.2,
        "k_y": 1,
        "gamma_y": 2,
        "mean_burst_size": 5,
    },
    [1, 0, 0, 0, 0],  # [n00, n10, n01, n11, y]
    [
        [-1, 1, 0, 0, 0],  # n00 -> n10
        [-1, 0, 1, 0, 0],  # n00 -> n01
        [1, -1, 0, 0, 0],  # n10 -> n00
        [1, 0, -1, 0, 0],  # n01 -> n00
        [0, -1, 0, 1, 0],  # n10 -> n11
        [0, 0, -1, 1, 0],  # n01 -> n11
        [0, 1, 0, -1, 0],  # n11 -> n10
        [0, 0, 1, -1, 0],  # n11 -> n01
        gil.burst_stoichiometry,  # 0 -> y
        [0, 0, 0, 0, -1],  # y -> 0
    ],
]