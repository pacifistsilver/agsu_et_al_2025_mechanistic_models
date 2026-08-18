"""Rate constants shared by the model and by the analysis of its fits.

Kept out of :mod:`stochtf.inference.models`, which is where they are applied,
so that reading a fit back -- a report, a figure -- does not have to import the
whole PyMC stack to learn two numbers.
"""

#: Promoter dissociation rates, in units of the mRNA degradation rate gamma.
#: These are the pair single-molecule tracking measures directly, which is why
#: the inference layer pins them rather than inferring them: stationary counts
#: cannot determine four switching rates at once. See
#: :mod:`stochtf.inference.identifiability`.
BETA_S = 0.04
BETA_N = 0.26
