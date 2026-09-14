"""Shared plotting helpers for the FSP example drivers."""

def plot_solution_and_domain(measurement, domains):
    """Plots the solution and the domain at a sequence of times.

    Args:
        recorder: Recorder holding the solution at each stored time.
        domain_states: Domain states stored alongside each solution.
    """
    import pylab
    # plot the solution
    pylab.figure(figsize=(10, 10))
    shape = (41, 41)
    for i, (t, distribution) in enumerate(zip(measurement.times, measurement.distributions)):
        pylab.subplot(3, 3, i + 1)
        dense_distribution = distribution.to_dense(shape)
        pylab.imshow(
            dense_distribution,
            interpolation = 'nearest',
            origin = 'lower'
        )
    pylab.tight_layout()
    pylab.figure(figsize=(10, 10))
    # plot the states in the domain
    for i, (t, domain) in enumerate(zip(measurement.times, domains)):
        pylab.subplot(3, 3, i + 1)
        domain_x, domain_y = domain
        pylab.scatter(domain_x, domain_y, marker = 'o', c = 'k', s = 6)
        pylab.xlim(0, 45)
        pylab.ylim(0, 45)
    pylab.tight_layout()
    pylab.show()

def plot_solution_and_domain_5d(measurement, domains, dim1=4, dim2=3, shape=(80, 2)):
    """Plots the solution and domain of a 5-D model, projected to two axes.

    Args:
        recorder: Recorder holding the solution at each stored time.
        domain_states: Domain states stored alongside each solution.
        dim1: Index of the first plotted dimension. A tuple is summed over.
        dim2: Index of the second plotted dimension. A tuple is summed over.
    """
    import pylab
    import math
    
    n_plots = len(measurement.times)
    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols) if ncols > 0 else 1

    pylab.figure(figsize=(ncols * 3, nrows * 2.5))
    # plot the solution
    for i, (t, distribution) in enumerate(zip(measurement.times, measurement.distributions)):
        pylab.subplot(nrows, ncols, i + 1)
        dense_distribution = distribution.to_dense(shape)
        # Transpose so x-axis is dim1 and y-axis is dim2
        pylab.imshow(
            dense_distribution.T,
            interpolation = 'nearest',
            origin = 'lower',
            aspect = 'auto'
        )
        pylab.title('Sol t=%.3g' % t)
    
    pylab.tight_layout()
    pylab.figure(figsize=(ncols * 3, nrows * 2.5))
    # plot the states in the domain
    for i, (t, domain) in enumerate(zip(measurement.times, domains)):
        pylab.subplot(nrows, ncols, i + 1)
        # domain is expected to have shape (5, N)
        if isinstance(dim1, tuple):
            domain_x = sum(domain[d] for d in dim1)
        else:
            domain_x = domain[dim1]
            
        if isinstance(dim2, tuple):
            domain_y = sum(domain[d] for d in dim2)
        else:
            domain_y = domain[dim2]
            
        pylab.scatter(domain_x, domain_y, marker = 'o', c = 'k', s = 6)
        pylab.xlim(0, shape[0])
        pylab.ylim(-0.5, shape[1] - 0.5)
        pylab.title('Dom t=%.3g' % t)
    pylab.tight_layout()
    pylab.show()