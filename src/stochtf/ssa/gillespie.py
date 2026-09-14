"""Gillespie direct-method simulator and dwell-time extraction."""

import numpy as np
import threading

def gillespie(
    initial_state,
    stoichiometry,
    propensity_fn,
    t_max,
    parameters=None,
    max_steps=1000000,
):
    """Simulates a reaction network by Gillespie's direct method.

    Args:
        initial_state: Copy number of each species, shape (num_species,).
        stoichiometry: Change in species counts per reaction, shape
            (num_reactions, num_species). Rows may instead be callables of
            (state, parameters) for state-dependent changes such as bursts.
        propensity_fn: Callable (state, parameters) returning propensities of
            shape (num_reactions,).
        t_max: Simulation time to stop at.
        parameters: Rate constants and anything else propensity_fn needs.
        max_steps: Step cap, so a fast network cannot loop indefinitely.

    Returns:
        A tuple (times, states): the times at which reactions fired and the
        state after each, both including the initial condition at t = 0.
    """
    state = np.array(initial_state, dtype=float)

    is_static = True
    try:
        stoich = np.array(stoichiometry, dtype=float)
        num_reactions, num_species = stoich.shape
    except (ValueError, TypeError):
        is_static = False
        stoich = stoichiometry
        num_reactions = len(stoich)
        num_species = len(state)

    assert (
        len(state) == num_species
    ), f"Initial state has length {len(state)}, but stoichiometry expects {num_species} species."

    times = [0.0]
    states = [state.copy()]

    t = 0.0
    step = 0

    rng = np.random.default_rng()

    while t < t_max and step < max_steps:
        propensities = np.array(propensity_fn(state, parameters), dtype=float)

        a0 = np.sum(propensities)

        # No propensity left: the network is stuck, so stop here.
        if a0 <= 0:
            break

        r1, r2 = rng.random(2)

        tau = -np.log(r1) / a0

        cum_prop = np.cumsum(propensities)
        val = r2 * a0

        reaction_idx = np.searchsorted(cum_prop, val)
        if reaction_idx >= num_reactions:
            reaction_idx = num_reactions - 1

        if is_static:
            state += stoich[reaction_idx]
        else:
            change = stoich[reaction_idx]
            if callable(change):
                state += np.array(change(state, parameters), dtype=float)
            else:
                state += np.array(change, dtype=float)

        t += tau

        times.append(t)
        states.append(state.copy())

        step += 1

    return np.array(times), np.array(states)

def burst_stoichiometry(state, p_params,):
    mean_burst = p_params["mean_burst_size"]
    p = 1.0 / mean_burst
    # Geometric draw, supported on 1, 2, ... so a burst is never empty.
    burst_size = np.random.default_rng().geometric(p)
    new_state = np.zeros(len(state))
    new_state[-1] = burst_size
    return new_state

def extract_on_off(times, states, species_idx, active_start, active_end):
    """Extracts ON and OFF dwell times from a simulated time course.

    The first and last dwell times are discarded, since they are censored by
    the start and end of the simulation.

    Args:
        times: Simulation times returned by :func:`gillespie`.
        states: Simulation states, shape (num_steps, num_species).
        species_idx: Indices of the columns holding the promoter state.
        active_start: First value of the active block, inclusive.
        active_end: Last value of the active block, inclusive.

    Returns:
        A tuple (on_times, off_times) of durations spent active and silent.
    """
    promoter_states = np.sum(states[:, species_idx], axis=1)
    is_on = (promoter_states >= active_start) & (promoter_states <= active_end)

    # A change between i and i+1 means the new state starts at i + 1.
    changes = np.where(is_on[:-1] != is_on[1:])[0]
    on_times = []
    off_times = []
    
    # Intermediate intervals only: the outer two are censored.
    for i in range(len(changes) - 1):
        idx_start = changes[i] + 1
        idx_end = changes[i+1] + 1
        duration = times[idx_end] - times[idx_start]
        if is_on[idx_start]:
            on_times.append(duration)
        else:
            off_times.append(duration)
            
    return np.array(on_times), np.array(off_times)


