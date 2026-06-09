"""Tests for model files

In more detail, we test for:
    1. Chromatin arrays and dictionaries are initialised correctly.
    2. set_site_state method correctly updates states.
    3. promoter logic is correct. i.e. the correct tf binds.
    4. get_species_label method correctly translates species > label
    5. _calculate_propensities returns correct propensities given a system snapshot.
"""


import pytest
import numpy as np
from .model_utils import TranscriptionFactor
from .model_state import ModelState
from .model_call import ModelCall


## initialise model
@pytest.fixture
def sample_tfs():
    return [
        TranscriptionFactor(id=1, name="SOX2", valency=2, is_activator=True),
        TranscriptionFactor(id=2, name="NANOG", valency=2, is_activator=False)
    ]

@pytest.fixture
def initial_counts():
    return {
        "sox2_monomer_free": 10,
        "nanog_monomer_free": 5,
        "mRNA": 0
    }

@pytest.fixture
def base_state(sample_tfs, initial_counts):
    return ModelState(
        tfs=sample_tfs, 
        total_binding_sites=10, 
        initial_species_states=initial_counts,
        promoter_site=4
    )

## TF logic tests
def test_tf_dangling_id(sample_tfs):
    sox2, nanog = sample_tfs
    
    assert sox2.dangling_id == -2
    assert nanog.dangling_id == -3

## ModelState tests
def test_model_state_initialisation(base_state):
    assert base_state.total_sites == 10
    assert base_state.promoter_site == 4
    
    # check if all sites start vacant
    assert np.all(base_state.chromatin_site_is_vacant == True)
    assert len(base_state.vacant_chromatin_sites) == 10
    
    # check dictionaries built correctly
    assert 1 in base_state.bound_sites
    assert 2 in base_state.bound_sites
    assert base_state.dangling_counts[1] == 0

## test binding site transitions
def test_set_site_state_binding(base_state):
    # bind a SOX2 (id=1) to site 3
    base_state.set_site_state(site=3, is_vacant=False, tf_id=1, is_undimered=True)
    
    assert base_state.chromatin_lattice[3] == 1
    assert base_state.chromatin_site_is_vacant[3] == False
    assert 3 not in base_state.vacant_chromatin_sites
    assert 3 in base_state.bound_sites[1]
    assert base_state.chromatin_all_undimered_monomers[3] == True

def test_set_site_state_unbinding(base_state):
    # bind, then unbind
    base_state.set_site_state(site=3, is_vacant=False, tf_id=1)
    base_state.set_site_state(site=3, is_vacant=True, tf_id=0)
    
    assert base_state.chromatin_lattice[3] == 0
    assert base_state.chromatin_site_is_vacant[3] == True
    assert 3 in base_state.vacant_chromatin_sites
    assert 3 not in base_state.bound_sites[1]

def test_set_site_state_dimerisation(base_state):
    # bind SOX2 to site 0, NANOG to site 1, and tether them together
    base_state.set_site_state(site=0, tf_id=1, partner_state=1)
    base_state.set_site_state(site=1, tf_id=2, partner_state=0)
    
    assert base_state.chromatin_partner_state[0] == 1
    assert base_state.chromatin_partner_state[1] == 0
    assert 0 in base_state.dimered_dimer_sites
    assert 1 in base_state.dimered_dimer_sites

def test_set_site_state_dangling_dimer(base_state):
    # bind SOX2 to site 5, but it has a dangling NANOG (partner_state = -3)
    base_state.set_site_state(site=5, tf_id=1, partner_state=-3)
    
    assert base_state.chromatin_partner_state[5] == -3
    assert base_state.dangling_counts[2] == 1 # Nanog (id=2) count increases

## test promoter logic
def test_is_promoter_active_vacant(base_state):
    assert base_state.is_promoter_active() == False

def test_is_promoter_active_direct_activator(base_state):
    # bind SOX2 (activator) directly to promoter
    p_site = base_state.promoter_site
    base_state.set_site_state(site=p_site, is_vacant=False, tf_id=1)
    
    assert base_state.is_promoter_active() == True

def test_is_promoter_active_direct_non_activator(base_state):
    # bind NANOG (non-activator) directly to promoter
    p_site = base_state.promoter_site
    base_state.set_site_state(site=p_site, is_vacant=False, tf_id=2)
    
    assert base_state.is_promoter_active() == False

def test_is_promoter_active_bridged_activator(base_state):
    # bind NANOG to promoter (non-activator), tethered to SOX2 at another site
    p_site = base_state.promoter_site
    other_site = p_site + 1
    
    # setup tether
    base_state.set_site_state(site=p_site, is_vacant=False, tf_id=2, partner_state=other_site)
    base_state.set_site_state(site=other_site, is_vacant=False, tf_id=1, partner_state=p_site)
    
    assert base_state.is_promoter_active() == True

# test label generation

def test_get_species_label(base_state):
    # test Vacant
    assert base_state.get_species_label(0) == "EMPTY"
    
    # test Monomer Bound
    base_state.set_site_state(site=0, is_vacant=False, tf_id=1, partner_state=-1)
    assert base_state.get_species_label(0) == "SOX2b"
    
    # test Dangling
    base_state.set_site_state(site=0, partner_state=-3) # Dangling NANOG
    assert base_state.get_species_label(0) == "SOX2b:NANOGf"
    
    # test Bivalent Heterodimer
    base_state.set_site_state(site=1, is_vacant=False, tf_id=2, partner_state=0)
    base_state.set_site_state(site=0, partner_state=1)
    assert base_state.get_species_label(0) == "NANOGb:SOX2b"

# test propensities calculation

def test_calculate_propensities_no_rates(sample_tfs, initial_counts):
    # if all rates are 0, propensities must be 0
    empty_rates = {}
    
    model = ModelCall(
        tfs=sample_tfs, 
        model_param=empty_rates, 
        model_var=initial_counts, 
        model_binding_sites=10, 
        sim_max_time=100
    )
    
    props, total = model._calculate_propensities()
    assert total == 0.0
    assert np.all(props == 0.0)

def test_calculate_propensities_mrna_production(sample_tfs, initial_counts):
    rates = {"k_prod_m": 5.0}
    model = ModelCall(sample_tfs, rates, initial_counts, 10, 100)
    
    # force promoter to be ON
    p_site = model.state.promoter_site
    model.state.set_site_state(site=p_site, is_vacant=False, tf_id=1)
    
    props, total = model._calculate_propensities()
    
    # index 8 is mRNA production in your matrix
    assert props[8] == 5.0
    assert total == 5.0

def test_calculate_propensities_mrna_degradation(sample_tfs, initial_counts):
    rates = {"k_deg_m": 0.5}
    initial_counts["mRNA"] = 10 # Start with 10 mRNA
    
    model = ModelCall(sample_tfs, rates, initial_counts, 10, 100)
    
    props, total = model._calculate_propensities()
    
    assert props[9] == 5.0