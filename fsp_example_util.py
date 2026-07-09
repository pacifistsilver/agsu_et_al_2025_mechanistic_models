"""
common utility functions for the fsp example scripts
"""

def plot_solution_and_domain(measurement, domains):
    """
    displays plots of the solution and the domain at various times
    
    used by the three three fsp example scripts
    """
    import pylab
    # plot the solution
    shape = (41, 41)
    for i, (t, distribution) in enumerate(zip(measurement.times, measurement.distributions)):
        pylab.subplot(3, 3, i + 1)
        dense_distribution = distribution.to_dense(shape)
        pylab.imshow(
            dense_distribution,
            interpolation = 'nearest',
            origin = 'lower'
        )
    pylab.figure()
    # plot the states in the domain
    for i, (t, domain) in enumerate(zip(measurement.times, domains)):
        pylab.subplot(3, 3, i + 1)
        domain_x, domain_y = domain
        pylab.scatter(domain_x, domain_y, marker = 'o', c = 'k', s = 6)
        pylab.xlim(0, 45)
        pylab.ylim(0, 45)
    pylab.show()

def plot_solution_and_domain_5d(measurement, domains, dim1=4, dim2=3, shape=(80, 2)):
    """
    displays plots of the solution and the domain at various times for a 5D model
    dim1 and dim2 are the indices of the dimensions to plot in the scatter plot.
    If dim1 or dim2 is a tuple, the dimensions will be summed.
    """
    import pylab
    import math
    
    n_plots = len(measurement.times)
    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols) if ncols > 0 else 1

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
        pylab.title('Sol t=%g' % t)
    
    pylab.figure()
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
        pylab.title('Dom t=%g' % t)
    pylab.show()