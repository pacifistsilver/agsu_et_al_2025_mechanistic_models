"""Stochastic models of NANOG/SOX2 promoter dynamics.

Code for Agsu et al., doi:10.64898/2025.12.03.691924.

analytical  closed forms and FSP for two-site promoters and dimer drivers
ssa         Gillespie sims of the monomer/homodimer/heterodimer models
cme         master equation solver (finite state projection)
inference   fitting those models to allele-resolved counts
"""

__version__ = "1.0.0"
