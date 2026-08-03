"""Stochastic modelling of NANOG/SOX2 dynamics in gene expression noise.

Code accompanying Agsu et al., "Protein-protein interactions drive differences
in the spatiotemporal dynamics of transcription factors NANOG and SOX2 in naive
pluripotent cells", doi:10.64898/2025.12.03.691924.

Subpackages
-----------
analytical  closed-form and FSP results for two-site promoters and dimer drivers
ssa         Gillespie simulation of the monomer / homodimer / heterodimer models
cme         Chemical Master Equation solver (finite state projection)
inference   exact-likelihood parameter inference against allele-resolved counts
"""

__version__ = "1.0.0"
