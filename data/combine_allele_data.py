import polars as pl
import mygene 
import numpy as np
mg = mygene.MyGeneInfo()

data_129, data_cast = "./data/129.txt", "./data/CAST.txt"

df_129, data_cast = pl.read_csv(
    source=data_129, has_header=True, separator=" ", truncate_ragged_lines=True
), pl.read_csv(
    source=data_cast, has_header=True, separator=" ", truncate_ragged_lines=True
)
new_df = df_129.join(other=data_cast, on="TRANSCRIPT_ID")
transcript_id_arr = new_df["TRANSCRIPT_ID"].to_numpy()
id_gene_map = mg.querymany(transcript_id_arr, scopes="ensembl.transcript", fields="symbol, name", species="mouse")

