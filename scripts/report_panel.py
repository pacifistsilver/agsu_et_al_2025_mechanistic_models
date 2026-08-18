"""Tabulate the inferred rates from a panel of joint fits.

Reads the checkpoint written by ``scripts/run_gene_panel.py`` and reports, per
gene, the posterior median and 94% interval for each free rate, together with
the burst statistics those rates imply. Works on a partial file, so it can be
run while the panel is still going.

Rates are in units of the mRNA degradation rate gamma: stationary counts fix
only the ratios, so multiply by a measured gamma to get absolute units. The two
off-rates are not inferred -- they are pinned, and their values are shown for
reference since every derived quantity depends on them.

Burst statistics are computed under whichever topology the fit favours, since
tau_on differs between them, and the column says which was used.

Usage
-----
    python scripts/report_panel.py
    python scripts/report_panel.py --csv results/panel_rates.csv
"""

import argparse
import csv
import json
import os

from stochtf import paths
from stochtf.analytical.bursts import rate_burst_stats
from stochtf.inference.models import FIXED_BETA_N, FIXED_BETA_S


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", default=paths.results("panel_joint_fits.json"))
    ap.add_argument("--csv", default=paths.results("panel_rates.csv"))
    ap.add_argument("--sort", default="mean",
                    choices=["mean", "gene", "k_y", "bayes_factor"])
    args = ap.parse_args()

    if not os.path.exists(args.fits):
        raise SystemExit(f"{args.fits} not found. Run "
                         "scripts/run_gene_panel.py first.")
    with open(args.fits, encoding="utf-8") as fh:
        records = json.load(fh)

    fitted = [r for r in records if r.get("status") == "fitted"]
    skipped = [r for r in records if r.get("status") != "fitted"]
    if not fitted:
        raise SystemExit(f"{args.fits} has no completed fits yet "
                         f"({len(skipped)} skipped)")

    key = {"mean": lambda r: -r["mean"], "gene": lambda r: r["gene"],
           "k_y": lambda r: -r["k_y"][1],
           "bayes_factor": lambda r: -r["bayes_factor"]}[args.sort]
    fitted.sort(key=key)

    print(f"{len(fitted)} genes fitted, {len(skipped)} skipped "
          f"(off-rates pinned at beta_s={FIXED_BETA_S:g}, "
          f"beta_n={FIXED_BETA_N:g}; all rates in units of gamma)\n")
    header = (f"{'gene':>12} {'mean':>7} {'Fano':>7} | "
              f"{'k_on,s (94%)':>26} {'k_on,n (94%)':>26} {'k_y (94%)':>24} | "
              f"{'topology':>11} {'ON':>6} {'f':>7} {'b':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    for r in fitted:
        a_s, a_n, k_y = r["alpha_s"], r["alpha_n"], r["k_y"]
        favoured = ("heterodimer" if r["p_heterodimer"] >= 0.5 else "monomer")
        on, freq, size = rate_burst_stats(favoured, a_s[1], a_n[1],
                                          k_y[1], FIXED_BETA_S,
                                          FIXED_BETA_N)
        print(f"{r['gene']:>12} {r['mean']:7.2f} {r['fano']:7.2f} | "
              f"{a_s[1]:8.3f} [{a_s[0]:7.3f},{a_s[2]:7.3f}] "
              f"{a_n[1]:8.3f} [{a_n[0]:7.3f},{a_n[2]:7.3f}] "
              f"{k_y[1]:7.1f} [{k_y[0]:6.1f},{k_y[2]:6.1f}] | "
              f"{favoured:>11} {on:6.3f} {freq:7.4f} {size:8.1f}")
        rows.append({
            "gene": r["gene"], "n": r["n"], "mean": r["mean"],
            "fano": r["fano"], "alleles": r["alleles"],
            "k_on_s": a_s[1], "k_on_s_lo": a_s[0], "k_on_s_hi": a_s[2],
            "k_on_n": a_n[1], "k_on_n_lo": a_n[0], "k_on_n_hi": a_n[2],
            "k_y": k_y[1], "k_y_lo": k_y[0], "k_y_hi": k_y[2],
            "beta_s_fixed": FIXED_BETA_S, "beta_n_fixed": FIXED_BETA_N,
            "favoured_topology": favoured,
            "p_heterodimer": r["p_heterodimer"],
            "bayes_factor": r["bayes_factor"],
            "on_fraction": on, "burst_frequency": freq, "burst_size": size,
        })

    if skipped:
        print(f"\nskipped: "
              + ", ".join(f"{r['gene']} ({r.get('reason', '')})"
                          for r in skipped))

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
