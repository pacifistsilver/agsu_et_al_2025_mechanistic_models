
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
    time_steps = numpy.concatenate((
        numpy.linspace(0.0, 1.0, 10),
        numpy.linspace(2.0, 100, 100)
    ))
    
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
        ('bound_nanog',),
        (lambda n00, n10, n01, n11, y: n10 + n11,)
    )
    recorder.add_target(
        ('bound_sox2',),
        (lambda n00, n10, n01, n11, y: n01 + n11,)
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
    # fsp_example_util.plot_solution_and_domain_5d(
    #     recorder[('bound_nanog', 'bound_sox2')],
    #     domains,
    #     dim1=(1, 3), # Nanog bound (n10 + n11)
    #     dim2=(2, 3), # Sox2 bound (n01 + n11)
    #     shape=(total_promoters + 1, total_promoters + 1)
    # )

    import fsp_flux_util
    print('plotting probability flux vector field')
    # Get the final probability distribution and domain states
    final_p, _ = fsp_solver.y
    final_domain = numpy.array(fsp_solver.domain_states)
    
    # Project X: Total Bound TFs = n10 + n01 + 2*n11
    # Project Y: mRNA count = y
    fsp_flux_util.plot_probability_flux(
        model=model,
        p=final_p,
        domain_states=final_domain,
        proj_x_func=lambda s: s[1] + s[2] + 2 * s[3],
        proj_x_delta_func=lambda v: v[1] + v[2] + 2 * v[3],
        proj_y_func=lambda s: s[4],
        proj_y_delta_func=lambda v: v[4],
        title="Probability Flux: mRNA vs Total Bound TFs"
    )


if __name__ == '__main__':
    main()