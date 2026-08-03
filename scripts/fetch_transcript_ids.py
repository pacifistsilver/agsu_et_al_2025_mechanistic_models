"""Extract per-gene count vectors for the four marker genes.

Looks up each gene's canonical transcript from Ensembl, then sums the matching
rows of the combined allele table into one count vector per gene, saved to
``data/processed/<gene>.npy``. Those arrays are the observed data for
``scripts/run_inference.py``.

Pipeline position
-----------------
    scripts/download_data.py        raw GEO tables   -> data/raw/
    scripts/prepare_alleles.py      join + gene map  -> data/combined_allele_data.csv
    scripts/fetch_transcript_ids.py                  -> data/processed/<gene>.npy

Note that ``prepare_alleles.py`` does not currently write
``combined_allele_data.csv``; see its docstring.
"""

import json
import os
import urllib.request

import numpy as np
import pandas as pd

from stochtf.paths import DATA_DIR, PROCESSED_DATA_DIR

GENES = {"nanog": "Nanog", "esrrb": "Esrrb", "rex1": "Zfp42", "sox2": "Sox2"}

COMBINED_CSV = os.path.join(DATA_DIR, "combined_allele_data.csv")


def get_canonical_transcript(symbol):
    url = f"https://rest.ensembl.org/lookup/symbol/mouse/{symbol}?expand=1"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                if "canonical_transcript" in data:
                    # Strip the version number (e.g. .6)
                    return data["canonical_transcript"].split(".")[0]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
    return None


def main():
    transcript_mapping = {}
    for name, symbol in GENES.items():
        t_id = get_canonical_transcript(symbol)
        if t_id:
            transcript_mapping[name] = [t_id]  # Keep it as a list for .isin() later
            print(f"{name} ({symbol}) canonical transcript: {t_id}")
        else:
            print(f"Failed to find canonical transcript for {name}")

    if not os.path.exists(COMBINED_CSV):
        raise SystemExit(
            f"{COMBINED_CSV} not found. Run scripts/download_data.py and "
            "scripts/prepare_alleles.py first."
        )

    print("Loading CSV...")
    df = pd.read_csv(COMBINED_CSV)
    print(f"Loaded {len(df)} rows.")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    for name, t_ids in transcript_mapping.items():
        # Try exact match first
        mask = df["TRANSCRIPT_ID"].isin(t_ids)
        subset = df[mask]

        if len(subset) == 0:
            # In case the CSV has version numbers like ENSMUST00000012345.1
            csv_t_ids = df["TRANSCRIPT_ID"].apply(lambda x: str(x).split(".")[0])
            mask = csv_t_ids.isin(t_ids)
            subset = df[mask]

        if len(subset) > 0:
            # One transcript per gene, but sum in case several rows matched
            counts = subset.drop("TRANSCRIPT_ID", axis=1).sum(axis=0).values
            out = os.path.join(PROCESSED_DATA_DIR, f"{name}.npy")
            np.save(out, counts)
            print(f"Saved {out} with shape {counts.shape} "
                  f"using ONLY canonical isoform: {t_ids[0]}")
        else:
            print(f"Warning: Failed to find {name} (canonical: {t_ids[0]}) in CSV.")


if __name__ == "__main__":
    main()
