import re

with open("src/model_reactions.py", "r") as f:
    code = f.read()

# 1. Update list(self.state.vacant_chromatin_sites)
code = code.replace(
    "vacant_sites = list(self.state.vacant_chromatin_sites)",
    "vacant_sites = self.state.vacant_sites[:self.state.vacant_count]"
)

# 2. Update bound_sites
code = code.replace(
    "list(self.state.bound_sites.get(tf_type, set()))",
    "self.state.bound_sites[tf_type][:self.state.bound_count[tf_type]]"
)

# 3. Update undimered_sites
code = code.replace(
    "self.state.undimered_sites.get(1, set())",
    "self.state.undimered_sites[1][:self.state.undimered_count[1]]"
)
code = code.replace(
    "self.state.undimered_sites.get(2, set())",
    "self.state.undimered_sites[2][:self.state.undimered_count[2]]"
)

# 4. Update dimered_dimer_sites
code = code.replace(
    "list(self.state.dimered_dimer_sites)",
    "self.state.dimered_sites[:self.state.dimered_count]"
)

# 5. Fix _sample_spatial_dimer
old_func = '''    def _sample_spatial_dimer(self, weight_matrix, is_tether=False):
        """Method selects a random site obtained from weight_matrix to dimerise.

        Args:
            weight_matrix (NDarray): 2D NxN matrix containing site to site weights.
            is_tether (bool, optional): Defaults to False.

        Returns:
            tuple: Randomised index in weight_matrix
        """
        if not is_tether:
            weight_matrix = np.triu(weight_matrix, k=1)
            
            
        flat_weights = weight_matrix.flatten()
        chosen_idx = np.random.choice(
            len(flat_weights), p=(flat_weights / np.sum(flat_weights))
        )
        
        
        return divmod(chosen_idx, self.state.total_sites)'''

new_func = '''    def _sample_spatial_dimer(self, weight_matrix, is_tether=False):
        """Method selects a random site obtained from weight_matrix to dimerise."""
        if not is_tether:
            weight_matrix = np.triu(weight_matrix, k=1)
            
        row_sums = weight_matrix.sum(axis=1)
        total_sum = np.sum(row_sums)
        
        if total_sum <= 0:
            return 0, 0
            
        r1 = np.random.rand() * total_sum
        row_idx = np.searchsorted(np.cumsum(row_sums), r1)
        
        col_weights = weight_matrix[row_idx, :]
        r2 = np.random.rand() * row_sums[row_idx]
        col_idx = np.searchsorted(np.cumsum(col_weights), r2)
        
        return row_idx, col_idx'''

code = code.replace(old_func, new_func)

with open("src/model_reactions.py", "w") as f:
    f.write(code)
print("Updated model_reactions.py")
