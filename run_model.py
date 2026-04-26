from plot_trajectories import ModelPlot
from main import ModelCall

model_param = {
    "sox2_free": 2, 
    "nanog_free": 1, 
    "sox2_bound": 0, 
    "nanog_bound": 0, 
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
    "k_bind_n": 1.0, "k_unbind_n": 1.4,
    "k_dimerise": 1,  
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

model.run_trajectory()
model_plot = ModelPlot(model=model)
model_plot.plot_trajectory_and_noise()
model_plot.plot_nucleosome_occupancy_history()