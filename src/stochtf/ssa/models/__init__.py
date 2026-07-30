"""Propensity functions and parameter sets for the three promoter models.

Each module exposes ``propensity_fn(state, params)`` and a ``MODEL`` tuple of
(params, initial_state, stoichiometry, propensity_fn, promoter_idx, mrna_idx) so
that ``scripts/run_ssa.py`` can drive any of them uniformly.
"""

from stochtf.ssa.models import heterodimer, homodimer, monomer

MODELS = {
    "monomer": monomer.MODEL,
    "homodimer": homodimer.MODEL,
    "heterodimer": heterodimer.MODEL,
}

__all__ = ["MODELS", "monomer", "homodimer", "heterodimer"]
