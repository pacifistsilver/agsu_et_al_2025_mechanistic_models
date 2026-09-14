"""FSP driver for the homodimer promoter model.

Sets up the recorder targets and projection the model needs, then solves
to a fixed error bound with support-based domain expansion.
"""

from stochtf.cme.models import homodimer_model

import numpy
import stochtf.cme.recorder
import stochtf.cme.fsp.solver
import stochtf.cme.fsp.support_expander
import stochtf.cme.domain
import stochtf.cme.statistics
from stochtf.cme import fsp_example_util

from stochtf import cme
def main():
    """Solves the homodimer model by FSP with support-based expansion."""
    
    model = homodimer_model.create_model()
    initial_states = cme.domain.from_iter((model.initial_state, ))
    
    # SolutionExpander grows the domain only around the support of the
    # current solution, rather than expanding everywhere.
    expander = cme.fsp.support_expander.SupportExpander(
        model.transitions,
        depth = 10,
        epsilon = 1.0e-7
    )
    
    # Time dependencies for the burr08 model are supplied too.
    fsp_solver = cme.fsp.solver.create(
        model,
        initial_states,
        expander
    )
    
    # Initially stiff, so start with fine steps and coarsen later.
    time_steps = numpy.linspace(1, 50, 50)
    
    
    # Bound the error of the final solution by epsilon.
    epsilon = 1.0e-2
    num_steps = numpy.size(time_steps)
    # Per-step error budget.
    max_error_per_step = epsilon / num_steps
    
    recorder = cme.recorder.create(
        (model.species, model.species_counts)
    )
    recorder.add_target(
        ('bound_sum',),
        (lambda n00, n10, n01, n11, y: n10 + n01 + n11,)
    )
    
    domains = []
    
    for i, t in enumerate(time_steps):        
        print('STEP t = %g' % t)
        fsp_solver.step(t, max_error_per_step)
        if i % 3 == 0:
            print('recording solution and domain')
            p, _ = fsp_solver.y
            recorder.write(t, p)
            # Copy the domain so it can be plotted afterwards.
            domains.append(numpy.array(fsp_solver.domain_states))
    print('OK')
    
    print('plotting solution and domain')
    total_promoters = int(sum(model.initial_state[0:4]))
    fsp_example_util.plot_solution_and_domain_5d(
        recorder[('y', 'bound_sum')],
        domains,
        dim1=4,
        dim2=(1, 2, 3),
        shape=(80, total_promoters + 1)
    )



if __name__ == '__main__':
    main()