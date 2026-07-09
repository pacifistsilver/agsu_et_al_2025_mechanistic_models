"""
Homodimer Model CME Implementation
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from ssa_params import homodimer_params
from cme import model

def create_model():
    params, initial_state, stoichiometry = homodimer_params
    
    alpha_n = params["alpha_n"]
    beta_n = params["beta_n"]
    k_y = params["k_y"]
    gamma_y = params["gamma_y"]
    mean_burst = params["mean_burst_size"]

    total_promoters = int(sum(initial_state[0:4]))
    max_n = total_promoters + 1
    shape = (max_n, max_n, max_n, max_n, 80)

    count_n00 = lambda *x: x[0]
    count_n10 = lambda *x: x[1]
    count_n01 = lambda *x: x[2]
    count_n11 = lambda *x: x[3]
    count_y   = lambda *x: x[4]

    def prop_bind_n10(*x): return alpha_n * count_n00(*x)
    def prop_bind_n01(*x): return alpha_n * count_n00(*x)
    def prop_unbind_n10(*x): return beta_n * count_n10(*x)
    def prop_unbind_n01(*x): return beta_n * count_n01(*x)
    def prop_bind_N_n10_n11(*x): return alpha_n * count_n10(*x)
    def prop_bind_N_n01_n11(*x): return alpha_n * count_n01(*x)
    def prop_unbind_N_n11_1(*x): return beta_n * count_n11(*x)
    def prop_unbind_N_n11_2(*x): return beta_n * count_n11(*x)
    def prop_transcription(*x): return k_y * count_n01(*x) + count_n11(*x) + count_n10(*x)
    def prop_degradation(*x): return gamma_y * count_y(*x)

    return model.create(
        name='Homodimer Model',
        reactions=(
            'n00->n10',
            'n00->n01',
            'n10->n00',
            'n01->n00',
            'n10->n11',
            'n01->n11',
            'n11->n10',
            'n11->n01',
            '0->y (burst)',
            'y->0'
        ),
        propensities=(
            prop_bind_n10,
            prop_bind_n01,
            prop_unbind_n10,
            prop_unbind_n01,
            prop_bind_N_n10_n11,
            prop_bind_N_n01_n11,
            prop_unbind_N_n11_1,
            prop_unbind_N_n11_2,
            prop_transcription,
            prop_degradation
        ),
        transitions=(
            (-1, 1, 0, 0, 0),
            (-1, 0, 1, 0, 0),
            (1, -1, 0, 0, 0),
            (1, 0, -1, 0, 0),
            (0, -1, 0, 1, 0),
            (0, 0, -1, 1, 0),
            (0, 1, 0, -1, 0),
            (0, 0, 1, -1, 0),
            (0, 0, 0, 0, mean_burst),
            (0, 0, 0, 0, -1)
        ),
        species=('n00', 'n10', 'n01', 'n11', 'y'),
        species_counts=(
            count_n00,
            count_n10,
            count_n01,
            count_n11,
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

    print("Running Homodimer CME model...")
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
        pylab.plot(np.arange(y_shape[0]), p_dense, color='purple')
        pylab.title('P(y; t = %.f)' % t)
        pylab.ylabel('Probability Density')
        pylab.xlabel('mRNA count')
    pylab.show()

if __name__ == '__main__':
    main()
