"""Join the 129 and CAST allele count tables and map transcripts to gene symbols.

INCOMPLETE. This script stops after building ``id_gene_map`` -- it does not write
``combined_allele_data.csv`` or the per-gene arrays in ``data/processed/``, so the
step that actually produced those files is not currently in the repository.
The committed ``data/processed/*.npy`` arrays are therefore not reproducible from
this script alone. See data/README.md.

Usage
-----
    python scripts/download_data.py     # fetch the raw tables first
    python scripts/prepare_alleles.py
"""

import os

import mygene
import polars as pl

from stochtf.paths import raw

DATA_129 = raw("GSE132589_ASEcount_G1_129.txt")
DATA_CAST = raw("GSE132589_ASEcount_G1_CAST.txt")


def read_counts(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found. Run scripts/download_data.py first."
        )
    return pl.read_csv(
        source=path, has_header=True, separator=" ", truncate_ragged_lines=True
    )


def main():
    df_129 = read_counts(DATA_129)
    df_cast = read_counts(DATA_CAST)

    new_df = df_129.join(other=df_cast, on="TRANSCRIPT_ID")
    transcript_id_arr = new_df["TRANSCRIPT_ID"].to_numpy()

    mg = mygene.MyGeneInfo()
    id_gene_map = mg.querymany(
        transcript_id_arr,
        scopes="ensembl.transcript",
        fields="symbol, name",
        species="mouse",
    )

    print(f"joined table: {new_df.shape}")
    print(f"mapped {len(id_gene_map)} transcript ids to gene symbols")
    print("NOTE: this script does not yet write its output; see the module docstring.")
    return new_df, id_gene_map


if __name__ == "__main__":
    main()
