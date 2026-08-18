import numpy as np
NGRID  = 50
BS0 = 1
BN0 = 1
al = np.logspace(-2.5,2.5, NGRID)
build = lambda x, y: (x, BS0, y, BN0)
print(build)