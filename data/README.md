# Data

## Source

Allele-resolved single-cell RNA-seq counts from mouse ES cells.

- **Accession:** [GSE132589](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132589) (Ochiai et al.)
- **Files used:** `GSE132589_ASEcount_G1_129.txt`, `GSE132589_ASEcount_G1_CAST.txt` (~99 MB each)

These two files were previously committed to this repository. They are public
and reproducible, so they are now downloaded on demand instead:

```bash
python scripts/download_data.py
```

`scripts/download_data.py --check` verifies files already present. The SHA256
column in that script is not yet filled in — run `shasum -a 256 data/raw/*.txt`
against the copies used for the paper and commit the values.

## Layout

| Path | Tracked | Description |
|---|---|---|
| `data/raw/` | no | Downloaded GEO tables |
| `data/combined_allele_data.csv` | no | 196 MB derived join of the two raw tables |
| `data/processed/*.npy` | yes | Per-gene count vectors (nanog, sox2, rex1, esrrb) |
| `data/synthetic/*.npy` | yes | Simulated counts from known parameters, for ABC validation |

## Pipeline

```
scripts/download_data.py         GEO            -> data/raw/
scripts/prepare_alleles.py       join + mapping -> data/combined_allele_data.csv
scripts/fetch_transcript_ids.py  canonical isoform sums -> data/processed/<gene>.npy
scripts/generate_synthetic.py    known params   -> data/synthetic/<model>.npy
```

**Known gap:** `scripts/prepare_alleles.py` stops after building the
transcript-to-symbol map and does not write `combined_allele_data.csv`. The
committed `data/processed/*.npy` arrays were produced by a version of that step
that is not in the repository, so they cannot currently be regenerated
end-to-end from source. The arrays themselves are tracked so that the inference
and figure code remains runnable.
