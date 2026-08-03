import pymc as pm
import arviz as az
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os 
az.style.use("arviz-darkgrid")
trace = "results/synthetic_heterodimer_data_dimer.nc"
trace = az.from_netcdf(trace)

az.plot_trace_dist(trace, compact=True)
az.combine_plots(trace, plots= [(az.plot_rank, {"thin": False}), (az.plot_autocorr, {}), (az.plot_ess, {})])
print(az.summary(trace, kind="all"))
plt.show()