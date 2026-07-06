import numpy as np
import stochpy
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

smod = stochpy.SSA()
smod.Model("independent_cme.psc", dir=os.getcwd())
smod.ChangeParameter("alpha_s", 0.5)
smod.ChangeParameter("alpha_n", 0.3)
smod.ChangeParameter("Stot", 10.0)
smod.ChangeParameter("Ntot", 10.0)
smod.ChangeParameter("NS", 5.0)
smod.ChangeParameter("NN", 5.0)

smod.ChangeInitialSpeciesCopyNumber("ns", 0)
smod.ChangeInitialSpeciesCopyNumber("nn", 0)
smod.DoStochSim(end=50.0, mode="time", trajectories=1, quiet=True)

idx_ns = smod.data_stochsim.species_labels.index("ns")
idx_nn = smod.data_stochsim.species_labels.index("nn")
print(smod.data_stochsim.species[-1, idx_ns])
print(smod.data_stochsim.species[-1, idx_nn])

smod.ChangeInitialSpeciesCopyNumber("ns", 0)
smod.ChangeInitialSpeciesCopyNumber("nn", 0)
smod.DoStochSim(end=50.0, mode="time", trajectories=1, quiet=True)
print(smod.data_stochsim.species[-1, idx_ns])
print(smod.data_stochsim.species[-1, idx_nn])
