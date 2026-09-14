"""Rate constants shared by the model and by the analysis of its fits.

Kept out of :mod:`stochtf.inference.models`, which is where they are applied,
so that reading a fit back -- a report, a figure -- does not have to import the
whole PyMC stack to learn two numbers.
"""

BETA_S = 0.04
BETA_N = 0.26
