"""Parameter inference against allele-resolved count data.

``models``
    ABC-SMC over the two competing promoter topologies: the heterodimer (two
    independent sites) and the monomer (one site the two factors compete for).
    Parameters are scored by simulating stationary counts and comparing them to
    the data, with no likelihood evaluated.

``likelihood``
    The exact stationary likelihood, from the distribution computed in
    ``stochtf.analytical.pgf``. Kept as the reference the ABC posterior can be
    checked against; it covers the independent-site gates only.

``identifiability``
    What stationary counts can determine at all, from the Fisher information --
    worth reading before interpreting any of the marginals.
"""
