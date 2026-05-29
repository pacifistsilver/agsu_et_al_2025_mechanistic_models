import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Import your model classes (adjust the import path if necessary)
from model import ModelState, SPECIES_MAP, config

# --- MOCK FIXTURES ---

@pytest.fixture
def mock_state():
    """
    Creates a simplified ModelState with a 5-site lattice.
    We mock the logger and reactions to prevent downstream errors during testing.
    """
    # Assuming ModelState takes (total_binding_sites, initial_species_states, promoter_site)
    initial_counts = {name: 0 for name in config.SPECIES_NAMES}
    state = ModelState(total_binding_sites=5, initial_species_states=initial_counts, promoter_site=0)
    
    # Mocking external dependencies attached to the model
    state.logger = MagicMock()
    
    return state

@pytest.fixture
def model_with_tether(mock_state):
    """
    Sets up a specific scenario:
    - Lattice size: 5 (Indices 0, 1, 2, 3, 4)
    - Site 2 is occupied by SOX2 (tf_type=1) and has a dangling NANOG (-3)
    - Sites 1 and 3 are empty
    - Site 4 is occupied by something else
    """
    # Force lattice state
    mock_state.chromatin_lattice = np.array([0, 0, 1, 0, 1])  # 1 = SOX2, 0 = Empty
    mock_state.chromatin_site_is_vacant = np.array([True, True, False, True, False])
    mock_state.chromatin_all_undimered_monomers = np.array([False, False, False, False, False])
    
    # -3 means tethered to NANOGf. -1 means no partner.
    mock_state.chromatin_partner_state = np.array([-1, -1, -3, -1, -1]) 
    
    # Initialize the spatial matrices
    indices = np.arange(5)
    chromatin_potential_dimer_partners = np.abs(indices[:, None] - indices[None, :])
    mock_state.dist_weighted_dimer_partners = np.exp(-chromatin_potential_dimer_partners / 1)
    np.fill_diagonal(mock_state.dist_weighted_dimer_partners, 0.0)
    
    mock_state.bivalent_transition_matrix = np.zeros((5, 5))
    
    return mock_state


# --- TEST CASES ---

def test_tether_spatial_weighting(model_with_tether):
    """
    Tests that update_site_weights correctly populates the bivalent_transition_matrix
    ONLY for vacant sites, and applies the exponential distance penalty.
    """
    state = model_with_tether
    
    # Call the update function specifically for the occupied site
    state.update_site_weights(site=2)
    
    # Extract the transition probabilities for the dangling foot at Site 2
    transition_row = state.bivalent_transition_matrix[2, :]
    print(transition_row)
    
    # 1. It should NOT be able to bind to itself
    assert transition_row[2] == 0.0
    
    # 2. It should NOT be able to bind to Site 4 (which is already occupied)
    assert transition_row[4] == 0.0
    
    # 3. It SHOULD be able to bind to adjacent Sites 1 and 3
    assert transition_row[1] > 0.0
    assert transition_row[3] > 0.0
    
    # 4. Site 1 and 3 are distance=1 away. Site 0 is distance=2.
    # Therefore, the probability of hitting 1 or 3 should be higher than 0.
    assert transition_row[1] == transition_row[3]
    assert transition_row[1] > transition_row[0]

