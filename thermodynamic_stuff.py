import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from main import ThermodynamicsCalculator

calc = ThermodynamicsCalculator()
alpha = 1.0
K_alpha = 0.5  
k1_values = np.linspace(0.1, 1.0, 10)
N_values = np.arange(1, 21) 
S_values = np.linspace(0.1, 20.0, 20)
results = []

for c_val in [0.5]:
    for k1 in k1_values:
        for N in N_values:
            for S in S_values:
                rate = calc.return_cluster_transcription_rate(N, k1, S, c_val, alpha)
                results.append({
                    "N": N,
                    "K1": k1,
                    "S": S,
                    "Transcription Rate": rate, 
                    "Condition": f"c = {c_val}",
                    "Experiment": "Maximal Transcription Rate"
                })

for mode_val in ["linear", "saturating"]:
    for k1 in k1_values:
        for N in N_values:
            for S in S_values:
                rate = calc.return_dwelltime_transcription_rate(N, k1, S, 1.0, alpha, K_alpha=K_alpha, mode=mode_val)
                results.append({
                    "N": N,
                    "K1": k1,
                    "S": S,
                    "Transcription Rate": rate, 
                    "Condition": f"mode = '{mode_val}'",
                    "Experiment": "Dwelltime"
                })

df = pd.DataFrame(results)

df_exp = df[df["Experiment"] == "Dwelltime"].copy()

fig = go.Figure()
n_unique = sorted(df_exp["N"].unique())

## plotting
for i, n_val in enumerate(n_unique):
    subset = df_exp[(df_exp["N"] == n_val) & (df_exp["Condition"] == "mode = 'linear'")]
    pivot = subset.pivot(index="S", columns="K1", values="Transcription Rate")
    
    fig.add_trace(go.Surface(
        z=pivot.values,
        x=pivot.columns, # K1 array
        y=pivot.index,   # S array
        visible=(i == 0),  
        name=f"N={n_val}",
        colorscale="Viridis",
        cmin=0, cmax=alpha 
    ))

# build sliders
steps = []
for i, n_val in enumerate(n_unique):
    step = dict(
        method="update",
        args=[
            {"visible": [False] * len(n_unique)},
            {"title": f"Transcription Rate Surface (N = {n_val})"}
        ],
        label=str(int(n_val))
    )
    step["args"][0]["visible"][i] = True 
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Number of Binding Sites (N): "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders,
    title=f"Transcription Rate Surface (N = {n_unique[0]})",
    scene=dict(
        xaxis_title="Affinity (K1)",
        yaxis_title="Concentration (S)",
        zaxis_title="Rate (\u03C1)", 
        zaxis=dict(range=[0, alpha]) 
    ),
    width=900,
    height=800
)

fig.write_html("dwell_time_saturating.html")
fig.show()