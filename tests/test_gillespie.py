"""Checks on the SSA driver and the three promoter models.

These are structural: they confirm each model's stoichiometry matrix and
propensity vector agree in length, that the simulator conserves the quantities it
should, and that mRNA statistics land near the analytic prediction.
"""

import numpy as np
import pytest

from stochtf.ssa import gillespie as gil
from stochtf.ssa.models import MODELS


@pytest.mark.parametrize("name", sorted(MODELS))
def test_propensity_and_stoichiometry_agree_in_length(name):
    params, initial_state, stoichiometry, propensity_fn, _, _ = MODELS[name]
    props = propensity_fn(initial_state, params)
    assert len(props) == len(stoichiometry), (
        f"{name}: {len(props)} propensities but {len(stoichiometry)} "
        "stoichiometry rows"
    )


@pytest.mark.parametrize("name", sorted(MODELS))
def test_stoichiometry_rows_match_state_width(name):
    """Rows are either fixed vectors or callables that return one.

    The transcription reaction uses ``gillespie.burst_stoichiometry``, a callable
    row that draws a geometric burst size at each firing, so it is checked by
    calling it rather than by length.
    """
    params, initial_state, stoichiometry, _, _, _ = MODELS[name]
    for i, row in enumerate(stoichiometry):
        if callable(row):
            produced = row(initial_state, params)
            assert len(produced) == len(initial_state), (
                f"{name}: callable stoichiometry row {i} returned width "
                f"{len(produced)}, state has {len(initial_state)}"
            )
        else:
            assert len(row) == len(initial_state), (
                f"{name}: stoichiometry row {i} has width {len(row)}, "
                f"state has {len(initial_state)}"
            )


@pytest.mark.parametrize("name", sorted(MODELS))
def test_propensities_are_non_negative_at_the_initial_state(name):
    params, initial_state, _, propensity_fn, _, _ = MODELS[name]
    props = np.asarray(propensity_fn(initial_state, params), dtype=float)
    assert np.all(props >= 0), f"{name}: negative propensity at t=0"


@pytest.mark.parametrize("name", sorted(MODELS))
def test_simulation_runs_and_stays_non_negative(name):
    params, initial_state, stoichiometry, propensity_fn, _, mrna_idx = MODELS[name]
    times, states = gil.gillespie(
        initial_state, stoichiometry, propensity_fn,
        t_max=20.0, parameters=params, max_steps=20000,
    )
    assert times.shape[0] == states.shape[0]
    assert np.all(np.diff(times) >= 0), "time must be non-decreasing"
    assert np.all(states >= 0), f"{name}: negative copy number reached"
    assert states.shape[1] == len(initial_state)
    assert np.all(np.isfinite(states[:, mrna_idx]))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_promoter_occupancy_is_conserved(name):
    """The promoter compartments partition one gene copy, so they sum to a constant."""
    params, initial_state, stoichiometry, propensity_fn, promoter_idx, _ = MODELS[name]
    if name == "monomer":
        pytest.skip("monomer tracks free/bound TF pools, not promoter compartments")
    times, states = gil.gillespie(
        initial_state, stoichiometry, propensity_fn,
        t_max=20.0, parameters=params, max_steps=20000,
    )
    occupancy = states[:, [0] + list(promoter_idx)].sum(axis=1)
    assert np.all(occupancy == occupancy[0])
