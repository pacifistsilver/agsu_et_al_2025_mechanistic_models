"""Dissociation rates, shared by the fitting code and by anything reading fits back.

These live here rather than in inference.models so that a report or a figure
doesn't have to import the whole PyMC stack just to get two numbers.
"""

# In units of gamma, the mRNA degradation rate. Single-molecule tracking measures
# this pair directly, which is why we pin them instead of fitting them: stationary
# counts can't determine four switching rates at once. See inference/identifiability.py.
BETA_S = 0.04
BETA_N = 0.26
