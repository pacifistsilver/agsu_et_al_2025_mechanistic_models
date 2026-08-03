# Stochastic modelling of NANOG/SOX2 dynamics in gene expression noise

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Code accompanying:

> G. G. Agsu et al., "Protein-protein interactions drive differences in the
> spatiotemporal dynamics of transcription factors NANOG and SOX2 in naïve
> pluripotent cells," Dec. 2025.
> doi: [10.64898/2025.12.03.691924](https://doi.org/10.64898/2025.12.03.691924)

## Install

```bash
git clone https://github.com/pacifistsilver/Srinjan_Modelling.git
cd Srinjan_Modelling
pip install -e .
```

Installing the package is what makes every script runnable from any working
directory. The core install covers the analytical results, the SSA, the CME
solver and all figures. The Bayesian inference stack is a heavier extra:

```bash
pip install -e '.[inference]'
```

```bash
pip install -e '.[all]'
```

### Docker

```bash
docker build -t stochtf .
```

```bash
docker run --rm -it -v $(pwd):/app stochtf python figures/fig01_burst_parameters.py
```

## Reproduce the figures

```bash
python figures/fig01_burst_parameters.py
python figures/fig04_f_over_gamma.py
python figures/fig05_promoter_logic.py
python figures/fig06_architecture.py
python figures/fig07_burst_architectures.py
python figures/fig09_monomer_heatmaps.py
```

Each writes an SVG into `figures/output/`. See
[figures/README.md](figures/README.md) for the script-to-figure map and a note
on the figure numbering, which needs confirming against the manuscript.

`figures/standalone_f_over_gamma.py` needs only numpy, scipy and matplotlib —
no install — and is the self-contained reference for the `f/gamma` result.

## Run the models

```bash
python scripts/run_ssa.py --model heterodimer
```

```bash
python scripts/run_fsp.py burr08 --expander support
```

```bash
python scripts/run_inference.py --gene esrrb --model dimer
```

Traces are written to `results/` and analysed in
`notebooks/05_abc_diagnostics.ipynb`.

### Inference uses the exact distribution, not a simulator

`stochtf.analytical.pgf` computes the stationary count distribution exactly, so
`scripts/run_inference.py` scores the whole distribution
(`log L = Σᵢ log P(yᵢ | θ)`) rather than comparing a Gillespie run to the data
through a summary statistic. There is no ABC tolerance to tune and no Monte
Carlo noise in the likelihood; sampling is still SMC.

The generating function `G(z) = E[z^y]` satisfies `γu·dG/du = Qᵀ G + u K G` with
`u = z − 1`. From that one equation:

```python
from stochtf.analytical import pgf

pgf.moments(0.5, 0.05, 0.3, 0.2, k_y=30.0, gamma=1.0, gate="OR")
# (mean, variance, Fano) -- exact, no truncation

pgf.stationary_pmf(0.5, 0.05, 0.3, 0.2, k_y=30.0, gamma=1.0, gate="OR")
# full P(y)
```

The ADD gate (the monomer model) has a closed-form PGF — a product of two
Kummer functions, since the two sites contribute additively. OR and AND do not,
so `stationary_pmf` solves the block-tridiagonal stationary CME, which is exact
for every gate and stable at the k_y/γ ≈ 250 the data implies. All routes agree
to machine precision; `pgf.pgf_ode` integrates the ODE directly as an
independent check.

Rates are inferred in units of γ, the mRNA degradation rate: stationary counts
determine only the ratios, so γ is fixed at 1 and cannot be identified separately.

## Data

Raw counts come from GEO accession
[GSE132589](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132589) and
are downloaded rather than committed:

```bash
python scripts/download_data.py
```

See [data/README.md](data/README.md) for the full pipeline and a known gap in
the preprocessing chain.

## Tests

```bash
pytest
```

```bash
pytest -m "not slow"
```

`tests/test_analytical.py` checks the identities the analytical module must
satisfy: `f = 1/(T_on + T_off)`, `<y> = b f / gamma`, closed-form Fano against
exact FSP, and the single-site limits.

## Layout

```text
├── src/stochtf/          installed package
│   ├── analytical/       closed-form + FSP results for two-site promoters
│   ├── ssa/              Gillespie simulator, parameters, promoter models
│   ├── cme/              chemical master equation solver (FSP)
│   ├── inference/        exact stationary likelihood + SMC models
│   ├── plotting.py       shared figure style
│   └── paths.py          project directory resolution
├── figures/              one script per paper figure -> figures/output/
├── scripts/              entry points (download, prepare, run, fit)
├── notebooks/            exploratory and diagnostic notebooks
├── tests/                pytest suite
└── data/                 processed and synthetic arrays (raw is downloaded)
```

## Citing

See [CITATION.cff](CITATION.cff). Please cite both the paper and the software.

## License

MIT — see [LICENSE](LICENSE).
