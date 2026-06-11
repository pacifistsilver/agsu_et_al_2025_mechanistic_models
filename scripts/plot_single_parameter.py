import argparse
import os
import sys
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

# Ensure absolute imports work from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eda_functions import Statistics
from src.compile_data import _return_binding_frequencies, _return_all_mfpt_trajectories
from src.plotting_utils import (
    plot_binding_frequencies,
    plot_occupancy_timeline,
    plot_occupancy_profile,
    plot_transcriptional_bursting,
    plot_sliding_distance_histogram,
    plot_survival_curve,
    plot_fano_histogram,
    plot_mfpt_histogram
)

def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for a single parameter set.")
    parser.add_argument("param_dir", type=str, help="Path to the parameter directory (e.g. hetmer_excl/sweep/kbn_0.23_kbs_0.57)")
    parser.add_argument("--run_id", type=int, default=0, help="Run ID to plot for timelines (default: 0)")
    parser.add_argument("--sim_time", type=int, default=1000, help="Total simulation time")
    parser.add_argument("--n_sites", type=int, default=10, help="Number of binding sites")
    args = parser.parse_args()

    param_dir = os.path.abspath(args.param_dir)
    if not os.path.isdir(param_dir):
        print(f"Error: Directory {param_dir} does not exist.")
        sys.exit(1)
        
    param_id = os.path.basename(param_dir.rstrip("/\\"))
    output_dir = os.path.join(param_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Parsing statistics for {param_dir}...")
    try:
        stats = Statistics.from_parquet_dir(
            param_set_dir=param_dir,
            total_sim_time=args.sim_time,
            total_binding_sites=args.n_sites
        )
    except Exception as e:
        print(f"Failed to load statistics: {e}")
        sys.exit(1)

    # 1. Occupancy Timeline
    print(f"Generating Occupancy Timeline for run_id {args.run_id}...")
    try:
        timeline_df = stats.extract_occupancy_timeline(args.run_id)
        if not timeline_df.empty:
            out_path = os.path.join(output_dir, f"{param_id}_timeline_run{args.run_id}.png")
            plot_occupancy_timeline(timeline_df, output_path=out_path, title=f"Binding Events Timeline (Run {args.run_id})")
        else:
            print("Timeline data empty, skipping plot.")
    except Exception as e:
        print(f"Failed to generate timeline plot: {e}")

    # 2. Time-Fraction Occupancy Profile
    print(f"Generating Time-Fraction Occupancy Profile...")
    try:
        profile_df = stats._calc_occupancy_fractions().collect().to_pandas()
        if not profile_df.empty:
            out_path = os.path.join(output_dir, f"{param_id}_occupancy_profile.png")
            plot_occupancy_profile(profile_df, output_path=out_path)
        else:
            print("Occupancy profile data empty, skipping plot.")
    except Exception as e:
        print(f"Failed to generate occupancy profile: {e}")
        
    # 3. Transcriptional Bursting Timeline
    print(f"Generating Transcriptional Bursting Timeline for run_id {args.run_id}...")
    try:
        mrna_df = stats.extract_mrna_timeseries(args.run_id)
        if not mrna_df.empty and 'timeline_df' in locals() and not timeline_df.empty:
            out_path = os.path.join(output_dir, f"{param_id}_transcriptional_bursting_run{args.run_id}.png")
            promoter_site =  int((stats.total_binding_sites - 1) / 2)
            plot_transcriptional_bursting(mrna_df, timeline_df, promoter_site, output_path=out_path)
        else:
            print("Bursting timeline data empty, skipping plot.")
    except Exception as e:
        print(f"Failed to generate bursting timeline plot: {e}")

    # 4. Sliding Distance Histogram
    print("Generating Sliding Distance Histogram...")
    try:
        traj_df = stats.extract_trajectories()
        if not traj_df.empty and "sliding_distance" in traj_df.columns:
            out_path = os.path.join(output_dir, f"{param_id}_sliding_distance.png")
            plot_sliding_distance_histogram(traj_df, output_path=out_path)
        else:
            print("No trajectory or sliding distance data, skipping.")
    except Exception as e:
        print(f"Failed to generate sliding distance histogram: {e}")
        
    # 5. Survival Curve
    print("Generating Survival Curve (Residence Time Decay)...")
    try:
        if 'traj_df' in locals() and not traj_df.empty and "bound_lifespan_s" in traj_df.columns:
            out_path = os.path.join(output_dir, f"{param_id}_survival_curve.png")
            plot_survival_curve(traj_df, lifespan_column="bound_lifespan_s", output_path=out_path)
        else:
            print("No trajectory or lifespan data, skipping.")
    except Exception as e:
        print(f"Failed to generate survival curve: {e}")

    # 6. Binding Frequencies
    print("Generating Binding Frequencies per site...")
    try:
        freq_lazy = stats._calc_binding_frequencies_per_site()
        if hasattr(freq_lazy, "collect"):
            freq_df = freq_lazy.collect().to_pandas()
        else:
            freq_df = freq_lazy.to_pandas()
            
        if not freq_df.empty:
            out_path = os.path.join(output_dir, f"{param_id}_binding_frequencies.png")
            plot_binding_frequencies(freq_df, output_path=out_path)
        else:
            print("Binding frequency data empty, skipping plot.")
    except Exception as e:
        print(f"Failed to generate binding frequencies plot: {e}")

    # 7. MFPT Histograms
    print("Generating MFPT Histograms...")
    try:
        if 'traj_df' in locals() and not traj_df.empty:
            out_path = os.path.join(output_dir, f"{param_id}_mfpt_histogram.png")
            plot_mfpt_histogram(
                traj_df, 
                lifespan_column="bound_lifespan_s", 
                output_path=out_path,
                title=f"MFPT Distribution ({param_id})"
            )
        else:
            print("Trajectory data empty, skipping MFPT plot.")
    except Exception as e:
        print(f"Failed to generate MFPT histogram: {e}")

    # 8. Fano Histograms
    print("Generating Fano Histograms...")
    try:
        model_params_path = os.path.join(param_dir, "model_parameters.txt")
        if os.path.exists(model_params_path):
            from src.compile_data import parse_parameters_txt
            rates, initial_state = parse_parameters_txt(model_params_path)
            features_df = stats.extract_per_run_features(param_id, rates, initial_state)
            
            fano_pandas = features_df.to_pandas()
            if "mrna_fano" in fano_pandas.columns:
                parts = param_id.split("_")
                try:
                    kbn = parts[1]
                    kbs = parts[3]
                    plot_fano_histogram(
                        fano_pandas,
                        kbn=kbn,
                        kbs=kbs,
                        output_dir=output_dir
                    )
                except IndexError:
                    print(f"Could not parse kbn and kbs from {param_id}")
        else:
            print("model_parameters.txt not found, skipping Fano histogram.")
    except Exception as e:
        print(f"Failed to generate Fano histogram: {e}")
        
    print(f"All plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
