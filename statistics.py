"""
stats

"""
import polars as pl
import simulation_config as config
from model import ModelCall
from types import SimpleNamespace

"""
TODO: 
sites + promoter:
- mean time bound (i.e. dt between binding and an unbinding) X
- mean time unbound (i.e. dt been an unbinding and a binding) X
- mean time bound (monomer) X
    - sox2 X
    - nanog X
- mean time bound (dimer) X
    - sox2-nanog single X
    - sox2-nanog double X
    - nanog-nanog single X
    - nanog-nanog double X
    
monomers/dimers: 
    - ratio homodimerised vs heterodimerised, homodimer single vs heterodimer single
    - % homodimer/heterodimer/monomer/etc... 

1. add check for any unique values
2. separate function to also count how many species switches occur between one binding interval and another.

class statistics()
init 
# helper functions
get weighted t species counts
    get t weighted average

calc dwell
calc species dwell
calc site dwell

calc dimers
calc monomers?

"""

class statistics():
    def __init__(self, df_states, df_dwell_times, df_reactions, total_sim_time: int, total_binding_sites: int):
        self.total_sim_time = total_sim_time        
        self.df_states_log = pl.LazyFrame(df_states)
        self.dwell_times_log = pl.LazyFrame(df_dwell_times)
        self.df_reaction_log = pl.LazyFrame(df_reactions)
        self.dwell_time_species_labels = ["NANOG:NANOG", "SOX2:NANOG", "SOX2", "NANOG"]
        self.total_binding_sites = total_binding_sites

    def _get_t_weighted_species_counts(self, df: pl.DataFrame):
        def _get_time_weighted_avg(df: pl.DataFrame, columns: str):
            """Get weighted species count average for whole simulation.
            """
            # do i even need this?
            return df.select((pl.sum_horizontal(columns) * pl.col("time")).sum()).item() / self.total_sim_time
        
        species_map = {
            "n_free": ["nanog_monomer_free"],
            "n_bound": ["nanog_monomer_bound"],
            "nn_free": ["nanog_nanog_dimer_free"],
            "nn_bound": ["nanog_nanog_dimer_bound"],
            "nn_single": ["nanog_nanog_dimer_single_bound"],
            "ns_free": ["nanog_sox2_dimer_free"],
            "ns_bound": ["nanog_sox2_dimer_bound"],
            "ns_single": ["nanog_sox2_dimer_single_bound"],
            "s_free": ["sox2_monomer_free"],
            "s_bound": ["sox2_monomer_bound"],
        }
        
        data = {name: _get_time_weighted_avg(df=df, columns=species) for name, species in species_map.items()}
        return data 
    
    def _return_site_dwell_times(self):
        """Returns average dwell times per species.
        """
        lf = self.dwell_times_log.lazy()
        
        dwell_times_unique = (
            lf.filter(pl.col("dwell_site").is_unique())
            .collect()
        )
        
        dwell_times_duplicates = (
            lf.filter(pl.col("dwell_site").is_duplicated())
            .collect()
            .pivot(
                index="species",
                on="dwell_site",
                values="time",
                aggregate_function="mean"
            )
        )
        
        return dwell_times_unique, dwell_times_duplicates    
    def _return_site_unbound_times(self):
        """Extract mean unbound time intervals for all sites.
        
        extract all species labels
        
        take two binding event and return the interval of time between them (i.e. difference)
            get all binding events
            from this look at all duplicate entries via dwell_site
            for these duplicate sites, then take the difference between them 
            issue is though if there are >2 events. what to do? 
            
        extract all dwell_site events and set the number of columns to 
        the length of the site with the most events.
        populate with dwell site times, beginning with the first
        in the time sequence wise.
        
        from this find the first ever event, then return this value 
        as this is the first value of the simulation.
        
        and from this all values will begin to be referenced from this.
        progressively updating as we go.
        
        at the same time, record the number of times we see a switch in species label.
        
        return: mean unbound time per dwell_site.
        dwell_site | b1 | b2 | b3 | b4 | ... | bn
        """
        lf = self.dwell_times_log.lazy().sort(["dwell_site", "time"])        
        
        # TODO: does not cover the interval between t = 0 and first binding at tx
        # also doesnt cover the interval between last binding at tx and tmax
        stats = lf.group_by("dwell_site").agg(
            # difference between consecutive times (time[n] - time[n-1])
            pl.col("time").diff().mean().alias("mean_unbound_time"),
            pl.col("species").filter(
                (pl.col("species") != pl.col("species").shift()) & 
                pl.col("species").shift().is_not_null()
            ).alias("switched_to_species")
        )       
         
        pivoted = (
            lf.with_columns(
                pl.format("b{}", pl.int_range(1, pl.len() + 1).over("dwell_site")).alias("event_idx")
            )
            .collect()
            .pivot(
                index="dwell_site",
                on="event_idx",
                values="time"
            )
        )
        result = stats.collect().join(pivoted, on="dwell_site")
        return result
        
    def _calculate_species_statistics(self):
        data = self._get_t_weighted_species_counts(self.df_states_log)
        n = SimpleNamespace(**data)
        monomer_data = []    
        
        nanog_total_unbound = n.n_free + (2 * n.nn_free) + n.ns_free
        nanog_total_bound = n.n_bound + (2 * n.nn_bound) + (2 * n.nn_single) + n.ns_bound + n.ns_single
        nanog_total = nanog_total_unbound + nanog_total_bound
        sox2_total_unbound = n.s_free + n.ns_free
        sox2_total_bound = n.s_bound + n.ns_bound + n.ns_single
        sox2_total = sox2_total_unbound + sox2_total_bound

        for name, bnd, unbnd, tot in [("NANOG", nanog_total_bound, nanog_total_unbound, nanog_total), 
                                    ("SOX2", sox2_total_bound, sox2_total_unbound, sox2_total)]:
            
            pct_b = (bnd / tot * 100) if tot > 0 else 0
            pct_u = (unbnd / tot * 100) if tot > 0 else 0
                        
            # time spent bound/unbound incorrect
            monomer_data.append({
                "variable": name,
                "average dwell time (s)": self._return_species_dwell_times(column = "species", species = name).select(pl.mean("time")).item(),
                "% bound": pct_b,
                "% unbound": pct_u,
            })
            
        df_monomer_stats = pl.DataFrame(monomer_data)
        return df_monomer_stats
        
if __name__ == '__main__': 
    INITIAL_MODEL_STATE = {
        "sox2_monomer_free": 10, "nanog_monomer_free": 1, "mRNA": 0
    }
    MODEL_RATE_CONSTANTS = {
        "k_bind_s": 1.0, "k_unbind_s": 0.06,
        "k_bind_n": 1.0, "k_unbind_n": 0.24,
        "k_dimerise": 1.0, "k_prod_m": 1.0, "k_deg_m": 0.53, "k_dissociate": 1
    }
    
    sim = ModelCall(
        model_param=MODEL_RATE_CONSTANTS, 
        model_var=INITIAL_MODEL_STATE, 
        model_binding_sites=10, 
        sim_max_time=100
    )
    df_states, df_dwell, df_rxns = sim.run_trajectory()
    s = statistics(
        df_states=df_states,
        df_dwell_times=df_dwell,
        df_reactions=df_rxns,
        total_sim_time = 100,
        total_binding_sites=10
    )
    y, x= s._return_site_dwell_times()
    print(y, x)