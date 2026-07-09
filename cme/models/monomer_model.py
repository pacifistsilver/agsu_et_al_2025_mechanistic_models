"""
Monomer Model CME Implementation
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from ssa_params import monomer_params
from cme import model

def create_model():
    params, initial_state, stoichiometry = monomer_params
    
    alpha_s = params["alpha_s"]
    beta_s = params["beta_s"]
    alpha_n = params["alpha_n"]
    beta_n = params["beta_n"]
    k_y = params["k_y"]
    gamma_y = params["gamma_y"]
    mean_burst = params["mean_burst_size"]

    max_n = int(initial_state[0] + initial_state[2]) + 1
    max_s = int(initial_state[1] + initial_state[3]) + 1
    shape = (max_n, max_s, max_n, max_s, 80)

    count_nf = lambda *x: x[0]
    count_sf = lambda *x: x[1]
    count_nb = lambda *x: x[2]
    count_sb = lambda *x: x[3]
    count_y  = lambda *x: x[4]

    def prop_bind_s(*x): return alpha_s * count_sf(*x)
    def prop_unbind_s(*x): return beta_s * count_sb(*x)
    def prop_bind_n(*x): return alpha_n * count_nf(*x)
    def prop_unbind_n(*x): return beta_n * count_nb(*x)
    def prop_transcription(*x): return k_y * count_nb(*x) + count_sb(*x)
    def prop_degradation(*x): return gamma_y * count_y(*x)

    return model.create(
        name='Monomer Model',
        reactions=(
            'sf->sb',
            'sb->sf',
            'nf->nb',
            'nb->nf',
            '0->y (burst)',
            'y->0'
        ),
        propensities=(
            prop_bind_s,
            prop_unbind_s,
            prop_bind_n,
            prop_unbind_n,
            prop_transcription,
            prop_degradation
        ),
        transitions=(
            (0, -1, 0, 1, 0),
            (0, 1, 0, -1, 0),
            (-1, 0, 1, 0, 0),
            (1, 0, -1, 0, 0),
            (0, 0, 0, 0, mean_burst),
            (0, 0, 0, 0, -1)
        ),
        species=('nf', 'sf', 'nb', 'sb', 'y'),
        species_counts=(
            count_nf,
            count_sf,
            count_nb,
            count_sb,
            count_y
        ),
        shape=shape,
        initial_state=tuple(initial_state)
    )

def main():
    import pylab
    from cme import solver, recorder

    m = create_model()
    s = solver.create(model=m, sink=True)
    r = recorder.create((m.species, m.species_counts))

    t_final = 20.0
    steps_per_time = 1
    time_steps = np.linspace(0.0, t_final, int(steps_per_time * t_final) + 1)

    print("Running Monomer CME model...")
    for t in time_steps:
        s.step(t)
        p, p_sink = s.y
        print('t : %.2f, truncation error : %.2g' % (t, p_sink))
        r.write(t, p)

    measurement = r[('y',)]
    epsilon = 1.0e-5
    for t, marginal in zip(measurement.times, measurement.distributions):
        pylab.figure()
        y_shape = (m.shape[4],)
        p_dense = marginal.compress(epsilon).to_dense(y_shape)
        pylab.plot(np.arange(y_shape[0]), p_dense, color='red')
        pylab.title('P(y; t = %.f)' % t)
        pylab.ylabel('Probability Density')
        pylab.xlabel('mRNA count')
    pylab.show()

if __name__ == '__main__':
    main()
