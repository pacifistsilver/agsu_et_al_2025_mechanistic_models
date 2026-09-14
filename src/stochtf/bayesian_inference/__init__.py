"""Fitting the promoter models to allele-resolved count data.

models
    SMC over the two competing topologies: the heterodimer (two independent
    sites) and the monomer (one site the two factors compete for). Proposals are
    scored against the exact stationary distribution by default, or through a
    simulator and a discrepancy if you ask for the ABC route.

likelihood
    The exact stationary likelihood, built on the distribution from
    analytical/pgf.py. Covers the independent-site gates only.
"""
