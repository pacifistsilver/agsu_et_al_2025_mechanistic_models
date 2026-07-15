import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

data = np.random.normal(loc=0, scale=1, size=1000)
def normal_sim(rng, a, b, size=1000):
    return rng.normal(a, b, size=size)
if __name__ == '__main__':
    with pm.Model() as example:
        a = pm.Normal("a", mu=0, sigma=5)
        b = pm.HalfNormal("b", sigma=1)
        s = pm.Simulator("s", normal_sim, params=(a, b), sum_stat="sort", epsilon=1, observed=data)

        idata = pm.sample_smc()
        idata.extend(pm.sample_posterior_predictive(idata))


