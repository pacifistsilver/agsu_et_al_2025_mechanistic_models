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
from .constants import SiteState


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
    assert np.all(base_state.chromatin_state == SiteState.EMPTY)
    assert base_state.vacant_count == 10
    
    # check trackers built correctly
    assert base_state.bound_sites.shape == (3, 10)

## test binding site transitions
def test_set_site_state_binding(base_state):
    base_state.set_site_state(3, SiteState.SOX2b)
    assert base_state.get_base_tf(base_state.chromatin_state[3]) == 1
    assert base_state.chromatin_state[3] != SiteState.EMPTY
    assert 3 not in base_state.vacant_sites[:base_state.vacant_count]

def test_set_site_state_unbinding(base_state):
    base_state.set_site_state(3, SiteState.SOX2b)
    base_state.set_site_state(3, SiteState.EMPTY)
    assert base_state.chromatin_state[3] == SiteState.EMPTY
    assert 3 in base_state.vacant_sites[:base_state.vacant_count]

def test_set_site_state_dimerisation(base_state):
    # Bind two SOX2 molecules
    base_state.set_site_state(0, SiteState.SOX2b)
    base_state.set_site_state(1, SiteState.SOX2b)
    
    # Dimerise them: In our model SOX2/SOX2 dimer isn't defined explicitly as homo dimer, but let's test SOX2/NANOG
    base_state.set_site_state(1, SiteState.NANOGb)
    
    base_state.set_site_state(0, SiteState.SOX2b_NANOGb, 1)
    base_state.set_site_state(1, SiteState.NANOGb_SOX2b, 0)
    
    assert base_state.chromatin_partner_site[0] == 1
    assert base_state.chromatin_partner_site[1] == 0
    assert 0 in base_state.dimered_sites[:base_state.dimered_count]
    assert 1 in base_state.dimered_sites[:base_state.dimered_count]

def test_set_site_state_dangling_dimer(base_state):
    # bind SOX2 to site 5, but it has a dangling NANOG 
    base_state.set_site_state(site=5, new_state=SiteState.SOX2b_NANOGf)
    
    assert base_state.chromatin_partner_site[5] == -1
    assert base_state.dangling_counts[2] == 1 # Nanog (id=2) count increases

## test promoter logic
def test_is_promoter_active_vacant(base_state):
    assert base_state.is_promoter_active() == False

def test_is_promoter_active_direct_activator(base_state):
    # bind SOX2 (activator) directly to promoter
    p_site = base_state.promoter_site
    base_state.set_site_state(site=p_site, new_state=SiteState.SOX2b)
    
    assert base_state.is_promoter_active() == True

def test_is_promoter_active_direct_non_activator(base_state):
    # bind NANOG (non-activator) directly to promoter
    p_site = base_state.promoter_site
    base_state.set_site_state(site=p_site, new_state=SiteState.NANOGb)
    
    assert base_state.is_promoter_active() == False

def test_is_promoter_active_bridged_activator(base_state):
    # bind NANOG to promoter (non-activator), tethered to SOX2 at another site
    p_site = base_state.promoter_site
    other_site = p_site + 1
    
    # setup tether
    base_state.set_site_state(site=p_site, new_state=SiteState.NANOGb_SOX2b, new_partner=other_site)
    base_state.set_site_state(site=other_site, new_state=SiteState.SOX2b_NANOGb, new_partner=p_site)
    
    assert base_state.is_promoter_active() == True

# test label generation

def test_get_species_label(base_state):
    # test Vacant
    assert base_state.get_species_label(0) == "EMPTY"
    
    # test Monomer Bound
    base_state.set_site_state(site=0, new_state=SiteState.SOX2b)
    assert base_state.get_species_label(0) == "SOX2b"
    
    # test Dangling
    base_state.set_site_state(site=0, new_state=SiteState.SOX2b_NANOGf) # Dangling NANOG
    assert base_state.get_species_label(0) == "SOX2b:NANOGf"
    
    # test Bivalent Heterodimer
    base_state.set_site_state(site=1, new_state=SiteState.NANOGb_SOX2b, new_partner=0)
    base_state.set_site_state(site=0, new_state=SiteState.SOX2b_NANOGb, new_partner=1)
    # The label for SOX2 base is SOX2b:NANOGb, for NANOG base it's NANOGb:SOX2b
    assert base_state.get_species_label(0) == "SOX2b:NANOGb"
    assert base_state.get_species_label(1) == "NANOGb:SOX2b"

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
    model.state.set_site_state(site=p_site, new_state=SiteState.SOX2b)
    
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