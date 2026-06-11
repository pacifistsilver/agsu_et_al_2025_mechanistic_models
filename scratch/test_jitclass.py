from numba.experimental import jitclass
from numba import int32, float64, boolean, int8
import numpy as np

spec = [
    ('total_sites', int32),
    ('vacant_sites', int32[:]),
]

@jitclass(spec)
class FastModelState:
    def __init__(self, total_sites):
        self.total_sites = total_sites
        self.vacant_sites = np.arange(total_sites, dtype=np.int32)
        
    def do_something(self):
        return self.vacant_sites[0]

f = FastModelState(10)
print(f.do_something())
