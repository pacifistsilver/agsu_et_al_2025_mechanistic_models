# Stochastic modelling of NANOG/SOX2 dynamics

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

## Figures

```bash
python figures/fig01_burst_parameters.py
python figures/fig04_f_over_gamma.py
python figures/fig05_promoter_logic.py
python figures/fig06_architecture.py
python figures/fig07_burst_architectures.py
python figures/fig09_monomer_heatmaps.py
```

## Model running

```bash
python -m stochtf.scripts.run_ssa --model heterodimer
```

```bash
python -m stochtf.scripts.run_fsp burr08 --expander support
```

```bash
python -m stochtf.scripts.run_inference --gene esrrb --model dimer
```

Traces are written to `results/` and analysed in
`notebooks/05_abc_diagnostics.ipynb`.


## Data

Raw counts come from GEO accession
[GSE132589](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132589) and
are downloaded rather than committed:

```bash
python -m stochtf.scripts.data_cleaning download
```

See [data/README.md](data/README.md) for the full pipeline and a known gap in
the preprocessing chain.


## License

MIT — see [LICENSE](LICENSE).
