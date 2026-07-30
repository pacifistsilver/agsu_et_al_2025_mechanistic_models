# Figures

Each script writes into `output/`. Run any of them from anywhere once the
package is installed (`pip install -e .`):

```bash
python figures/fig01_burst_parameters.py
```

## Script to figure map

| Script | Output | Content |
|---|---|---|
| `fig01_burst_parameters.py` | `fig1_burst_parameters.svg` | Burst frequency and size, two-site OR promoter |
| `fig04_f_over_gamma.py` | `fig4_f_over_gamma.svg` | `f/gamma` as the NB shape parameter |
| `fig05_promoter_logic.py` | `fig5_promoter_logic.svg` | OR vs AND vs additive logic |
| `fig06_architecture.py` | `fig6_architecture.svg` | Promoter AND vs solution heterodimer |
| `fig07_burst_architectures.py` | `fig7_burst_architectures.svg` | Burst statistics across architectures |
| `fig09_monomer_heatmaps.py` | `fig9_monomer_heatmaps.svg` | Monomer sweeps in (alpha_s, alpha_n) |
| `standalone_f_over_gamma.py` | `fig4_f_over_gamma_standalone.svg` | Dependency-free duplicate of the `f/gamma` result |

## Numbering needs confirming

The figure numbers above come from the **output filenames**, which were the only
consistent signal in the original tree. They do not agree with the old script
names or with the docstrings:

- the script that emits `fig4_f_over_gamma` was named `plot_figure2.py`
- `plot_logic.py`, `plot_arch.py`, `plot_bursts_arch.py` have docstrings reading
  "Fig 5", "Fig 6", "Fig 7", which do match their output names
- there is no script for figures 2, 3 or 8 in this repository

Confirm these against the manuscript and rename if needed.

## Style

All scripts call `stochtf.plotting.use_paper_style()` rather than carrying their
own `rcParams` block. Colours come from `PALETTE` and `ARCH_COLOURS` in the same
module. `standalone_f_over_gamma.py` is the deliberate exception: it duplicates
both the model and the style so it can run without installing the package.

Set `STOCHTF_FIGURE_DIR` to write elsewhere.
