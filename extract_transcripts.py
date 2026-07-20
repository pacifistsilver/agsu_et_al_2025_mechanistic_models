import pandas as pd
import urllib.request
import json
import numpy as np

genes = {"nanog": "Nanog", "esrrb": "Esrrb", "rex1": "Zfp42", "sox2": "Sox2"}

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

transcript_mapping = {}
for name, symbol in genes.items():
    t_id = get_canonical_transcript(symbol)
    if t_id:
        transcript_mapping[name] = [t_id]  # Keep it as a list for .isin() later
        print(f"{name} ({symbol}) canonical transcript: {t_id}")
    else:
        print(f"Failed to find canonical transcript for {name}")

print("Loading CSV...")
df = pd.read_csv("data/combined_allele_data.csv")
print(f"Loaded {len(df)} rows.")

for name, t_ids in transcript_mapping.items():
    # Try exact match first
    mask = df['TRANSCRIPT_ID'].isin(t_ids)
    subset = df[mask]
    
    if len(subset) == 0:
        # In case the CSV has version numbers like ENSMUST00000012345.1
        csv_t_ids = df['TRANSCRIPT_ID'].apply(lambda x: str(x).split('.')[0])
        mask = csv_t_ids.isin(t_ids)
        subset = df[mask]
        
    if len(subset) > 0:
        # We only have one transcript in t_ids, but we sum just in case multiple rows perfectly matched somehow
        counts = subset.drop('TRANSCRIPT_ID', axis=1).sum(axis=0).values
        np.save(f"data/{name}.npy", counts)
        print(f"Saved data/{name}.npy with shape {counts.shape} using ONLY canonical isoform: {t_ids[0]}")
    else:
        print(f"Warning: Failed to find {name} (canonical: {t_ids[0]}) in CSV.")
