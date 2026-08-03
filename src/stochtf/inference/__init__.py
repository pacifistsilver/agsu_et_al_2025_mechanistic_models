"""Exact-likelihood parameter inference against allele-resolved count data.

The stationary distribution is computed exactly (stochtf.analytical.pgf), so the
full-distribution likelihood replaces the former ABC simulator and summary
statistic. Sampling is still SMC.
"""
