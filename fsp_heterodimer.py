
from cme.models import heterodimer_model

import numpy
import cme.recorder
import cme.fsp.solver
import cme.fsp.support_expander
import cme.domain
import cme.statistics
import fsp_example_util

def main():
    """
    solve heterodimer_model model using FSP with better expansion approach
    """
    
    # create model and initial states for domain
    model = heterodimer_model.create_model()
    initial_states = cme.domain.from_iter((model.initial_state, ))
    
    # Create expander for FSP expansion strategy.
    # The SolutionExpander only expands states around the
    # support of the current solution, instead of
    # expanding the entire domain
    expander = cme.fsp.support_expander.SupportExpander(
        model.transitions,
        depth = 10,
        epsilon = 1.0e-7
    )
    
    # create fsp solver for model, initial states, expander
    # - time dependencies for the burr08 model are also supplied
    fsp_solver = cme.fsp.solver.create(
        model,
        initial_states,
        expander
    )
    
    # define time steps:
    # this problem is initially stiff so
    # we begin with some finer time steps
    # before changing to coarser steps
    time_steps = numpy.linspace(1, 50, 50)

    
    # we want the error of the solution at the
    # final time to be bounded by epsilon
    epsilon = 1.0e-2
    num_steps = numpy.size(time_steps)
    # define how much error is tolerated per step
    max_error_per_step = epsilon / num_steps
    
    # create recorder to record species counts
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
            # record the solution
            p, _ = fsp_solver.y
            recorder.write(t, p)
            # store a copy of the domain so we can plot it later
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