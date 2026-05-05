from plot_trajectories import ModelPlot
from main import ModelCall

model_param = {
    "sox2_monomer_free": 1, 
    "nanog_monomer_free": 1, 
    "sox2_monomer_bound": 0, 
    "nanog_monomer_bound": 0, 
    "nanog_sox2_dimer_bound": 0, 
    "nanog_nanog_dimer_bound": 0,
    "nanog_sox2_dimer_free": 0,
    "nanog_nanog_dimer_free": 0,
    "mrna": 0
}

model_var = {
    "k_prod_s": 0.0, "k_deg_s": 0.0, 
    "k_prod_n": 0.0, "k_deg_n": 0.0,
    "k_bind_s": 1.0, "k_unbind_s": 0.06,
    "k_bind_n": 1.0, "k_unbind_n": 0.25,
    "k_dimerise": 1.0,  
    "k_prod_m": 1.0,    
    "k_deg_m": 0.53, 
}

sim_max_time = 100
model = ModelCall(
    model_param=model_param, 
    model_var=model_var, 
    model_binding_sites=10, 
    sim_max_time=sim_max_time, 
    record_interval=1.0, 
    track_history=True
)

model_output = model.run_trajectory()
model_plot = ModelPlot(model=model)
#model_plot.plot_trajectory_and_noise()
#model_plot.plot_site_occupancy_history()
model_plot.save_reaction_history_log()
dwell_times = model_plot.get_average_dwell_times_by_species()
print(dwell_times)
for species, avg_time in dwell_times.items():
    print(f"{species}: {avg_time:.2f} seconds")

print(model_output)