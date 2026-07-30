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
    """
    Stochastic Simulation Algorithm (SSA) / Gillespie's Direct Method.

    Parameters:
    -----------
    initial_state : array-like of shape (num_species,)
        Initial copy number of each species.
    stoichiometry : array-like of shape (num_reactions, num_species)
        The stoichiometry matrix where row i represents the change in species counts
        due to reaction i.
    propensity_fn : callable
        A function propensity_fn(state, parameters) that returns an array-like of
        propensities of shape (num_reactions,).
    t_max : float
        Maximum simulation time.
    parameters : dict or array-like, optional
        Reaction rate constants and other parameters passed to propensity_fn.
    max_steps : int, optional
        Maximum number of simulation steps to prevent infinite loop.

    Returns:
    --------
    times : ndarray of shape (num_steps,)
        The times at which reactions occurred (including t=0).
    states : ndarray of shape (num_steps, num_species)
        The state of the system at each time (including the initial state).
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

    # Random number generator
    rng = np.random.default_rng()

    while t < t_max and step < max_steps:
        # Calculate propensities
        propensities = np.array(propensity_fn(state, parameters), dtype=float)

        # Calculate total propensity
        a0 = np.sum(propensities)

        # If total propensity is zero, no more reactions can occur
        if a0 <= 0:
            break

        # Draw two random numbers
        r1, r2 = rng.random(2)

        # Compute time to next reaction
        tau = -np.log(r1) / a0

        # Determine which reaction occurs
        cum_prop = np.cumsum(propensities)
        val = r2 * a0

        reaction_idx = np.searchsorted(cum_prop, val)
        if reaction_idx >= num_reactions:
            reaction_idx = num_reactions - 1

        # Update state and time
        if is_static:
            state += stoich[reaction_idx]
        else:
            change = stoich[reaction_idx]
            if callable(change):
                state += np.array(change(state, parameters), dtype=float)
            else:
                state += np.array(change, dtype=float)

        t += tau

        # Record state and time
        times.append(t)
        states.append(state.copy())

        step += 1

    return np.array(times), np.array(states)

def burst_stoichiometry(state, p_params,):
    mean_burst = p_params["mean_burst_size"]
    p = 1.0 / mean_burst
    # Draw from geometric distribution starting at 1
    burst_size = np.random.default_rng().geometric(p)
    new_state = np.zeros(len(state))
    new_state[-1] = burst_size
    return new_state

def extract_on_off(times, states, species_idx, active_start, active_end):
    """
    Extract the complete on and off times (dwell times) from time courses 
    simulated using the gillespie function.
    
    Note: The first and last incomplete dwell times (at the boundaries of 
    the simulation) are discarded to avoid censored data.
    
    Parameters:
    -----------
    times : ndarray
        1D array of simulation times returned by gillespie().
    states : ndarray
        2D array of simulation states (shape: num_steps x num_species).
    species_idx : ndarray
        The indicies of the species (column in states) that represents the promoter states.
    active_start : float
        Start value of the active block (inclusive).
    active_end : float
        End value of the active block (inclusive).
        
    Returns:
    --------
    on_times : ndarray
        Durations spent in the ON (active) state.
    off_times : ndarray
        Durations spent in the OFF (inactive) state.
    """
    # Identify ON states (1) and OFF states (0)
    promoter_states = np.sum(states[:, species_idx], axis=1)
    is_on = (promoter_states >= active_start) & (promoter_states <= active_end)

    # Find indices where state changes
    # A change happens between i and i+1 when is_on[i] != is_on[i+1]
    # So the new state starts at index `change + 1`
    changes = np.where(is_on[:-1] != is_on[1:])[0]
    on_times = []
    off_times = []
    
    # We only consider intermediate intervals to avoid right/left censoring
    for i in range(len(changes) - 1):
        idx_start = changes[i] + 1
        idx_end = changes[i+1] + 1
        duration = times[idx_end] - times[idx_start]
        if is_on[idx_start]:
            on_times.append(duration)
        else:
            off_times.append(duration)
            
    return np.array(on_times), np.array(off_times)


