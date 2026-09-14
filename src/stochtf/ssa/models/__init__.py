"""Propensities and parameter sets for the three promoter models.

Each module gives a propensity_fn(state, params) and a MODEL tuple of
(params, initial_state, stoichiometry, propensity_fn, promoter_idx, mrna_idx),
which is all run_ssa needs to drive any of them the same way.
"""

from stochtf.ssa.models import heterodimer, homodimer, monomer

MODELS = {
    "monomer": monomer.MODEL,
    "homodimer": homodimer.MODEL,
    "heterodimer": heterodimer.MODEL,
}

__all__ = ["MODELS", "monomer", "homodimer", "heterodimer"]
