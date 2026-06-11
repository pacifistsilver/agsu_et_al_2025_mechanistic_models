import pytest
import pandas as pd


class TrajectoryTracker:
    def _track_run_trajectories(self, run_id, run_data):
        """Scans a single run to track individual molecules across sites using paired_site links."""
        active_lattice = {} # Maps site index to a shared Molecule dictionary
        completed = []
        molecule_counter = 0

        for row in run_data.itertuples():
            site = getattr(row, "dwell_site", -1)
            paired = getattr(row, "paired_site", -1)
            current_t = getattr(row, "current_sim_time", 0.0)
            old_sp = getattr(row, "old_species", "UNKNOWN")
            new_sp = getattr(row, "new_species", "UNKNOWN")
            
            
            # paired IS newsp NOT oldsp IS
            # paired NOT newsp NOT oldsp IS
            # paired IS newp IS oldsp IS
            # paired IS newp IS oldsp NOT
            # paired NOT newp IS oldsp NOT
            
            # obtain rows where EMPTY - > DIMER
            if old_sp == "EMPTY" and paired == -1 and new_sp != "EMPTY":
                molecule_counter += 1
                active_lattice[site] = {
                    "molecule_id": molecule_counter,
                    "starting_species": new_sp,
                    "start_time": current_t  
                }
            
            # obtain rows where sliding occurs             
            elif old_sp == "EMPTY" and paired != -1 and new_sp != "EMPTY":
                if paired in active_lattice:
                    active_lattice[site] = active_lattice[paired]

            # obtain rows where dissociation occurs
            elif old_sp != "EMPTY" and new_sp == "EMPTY" and paired == -1:
                if site in active_lattice:
                    dying_molecule = active_lattice.pop(site)

                    total_lifespan = current_t - dying_molecule["start_time"] 
                    
                    completed.append({
                        "run_id": run_id,
                        "molecule_id": dying_molecule["molecule_id"],
                        "starting_species": dying_molecule["starting_species"],
                        "bound_lifespan_s": total_lifespan
                    })
                
            # rows where one foot exits
            elif old_sp != "EMPTY" and new_sp == "EMPTY" and paired != -1:
                if site in active_lattice:
                    active_lattice.pop(site)

        return completed        
        
# ---------------------------------------------------------
# TESTS
# ---------------------------------------------------------

@pytest.fixture
def tracker():
    return TrajectoryTracker()

def test_case_a_and_c_simple_monomer(tracker):
    """Tests a simple monomer binding and then unbinding."""
    data = [
        # 1. Monomer binds at t=0.0 (Case A)
        {"dwell_site": 5, "paired_site": -1, "current_sim_time": 0.0, "old_species": "EMPTY", "new_species": "SOX2b"},
        # 2. Monomer unbinds at t=10.5 (Case C)
        {"dwell_site": 5, "paired_site": -1, "current_sim_time": 10.5, "old_species": "SOX2B", "new_species": "EMPTY"},
    ]
    df = pd.DataFrame(data)
    
    results = tracker._track_run_trajectories("run_1", df)
    
    assert len(results) == 1
    assert results[0]["molecule_id"] == 1
    assert results[0]["starting_species"] == "SOX2b"
    assert results[0]["bound_lifespan_s"] == 10.5
    assert results[0]["run_id"] == "run_1"
    
def test_multiple_independent_molecules(tracker):
    """Ensures multiple molecules on the lattice don't overwrite each other."""
    data = [
        # Molecule 1 lands on Site 0 at t=0.0
        {"dwell_site": 0, "paired_site": -1, "current_sim_time": 0.0, "old_species": "EMPTY", "new_species": "MOL_1"},
        # Molecule 2 lands on Site 9 at t=0.0
        {"dwell_site": 9, "paired_site": -1, "current_sim_time": 0.0, "old_species": "EMPTY", "new_species": "MOL_2"},
        
        # Molecule 1 unbinds at t=2.0s
        {"dwell_site": 0, "paired_site": -1, "current_sim_time": 2.0, "old_species": "MOL_1", "new_species": "EMPTY"},
        
        # Molecule 2 unbinds at t=15.0s
        {"dwell_site": 9, "paired_site": -1, "current_sim_time": 15.0, "old_species": "MOL_2", "new_species": "EMPTY"},
    ]
    df = pd.DataFrame(data)
    
    results = tracker._track_run_trajectories("run_multi", df)
    
    assert len(results) == 2
    
    # Verify Molecule 1
    assert results[0]["molecule_id"] == 1
    assert results[0]["starting_species"] == "MOL_1"
    assert results[0]["bound_lifespan_s"] == 2.0
    
    # Verify Molecule 2
    assert results[1]["molecule_id"] == 2
    assert results[1]["starting_species"] == "MOL_2"
    assert results[1]["bound_lifespan_s"] == 15.0
def test_case_b_and_d_bipedal_walking(tracker):
    """Tests the shared dictionary reference logic during a bipedal walk."""
    data = [
        {"dwell_site": 2, "paired_site": -1, "current_sim_time": 0.0, "old_species": "EMPTY", "new_species": "SOX2b:NANOGf"},
        {"dwell_site": 3, "paired_site": 2, "current_sim_time": 1.0, "old_species": "EMPTY", "new_species": "SOX2b:NANOGb"},
        {"dwell_site": 2, "paired_site": 3, "current_sim_time": 2.0, "old_species": "SOX2b:NANOGb", "new_species": "EMPTY"},
        {"dwell_site": 3, "paired_site": -1, "current_sim_time": 3.0, "old_species": "SOX2b:NANOGf", "new_species": "EMPTY"}
    ]
    df = pd.DataFrame(data)
    
    results = tracker._track_run_trajectories("run_walk", df)
    print(results)

    assert len(results) == 1
    assert results[0]["molecule_id"] == 1
    assert results[0]["starting_species"] == "SOX2b:NANOGf"
    print(results)
    print(results[0]["bound_lifespan_s"])
    assert results[0]["bound_lifespan_s"] == 3.0  # 5.0 + 3.0

