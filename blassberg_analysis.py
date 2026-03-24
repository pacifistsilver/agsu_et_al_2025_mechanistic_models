import pandas as pd
supp_table7_headers = ["Geneid","Nearest.PromoterID", "cluster"]
supp_table8_headers = ["sequence_name", "score"]
na_values = ["score", "cluster"]
file = "/Users/danielweiluo/Desktop/Srinjan_Modelling/41556_2022_910_MOESM3_ESM.xlsx"

file_table8 = pd.read_excel(file, sheet_name=["supplementary table 8"], header=2, usecols = supp_table8_headers, na_values = na_values)
file_table7 = pd.read_excel(file, sheet_name=["supplementary table 7"], header=1, usecols = supp_table7_headers)
file_table8 = file_table8["supplementary table 8"]
file_table7 = file_table7["supplementary table 7"]
file_table8["score"] = file_table8["score"].apply(lambda x: float(x))
file_table8['Geneid'] = file_table8['sequence_name'].astype(str).str.split('::').str[0]
file_table8 = file_table8.drop(columns = "sequence_name")
out_table = pd.merge(
    file_table8,
    file_table7,
    how="outer"
)

# geneid // gene_name // cluster // score
# report ratio of low to high affinity sites >20 and >20 for cluster3 
# extract high and low affinity to separate excel files. 

def process_clusters(cluster_df, affinity_high):
    current_cluster = cluster_df.loc[out_table["cluster"] == x]
    current_cluster = current_cluster.sort_values(by="score", ascending=False)
    current_cluster = current_cluster.loc[:, ["Geneid", "Nearest.PromoterID", "cluster", "score"]]
    if affinity_high is True:
        cluster_affinity = current_cluster.loc[out_table["score"] > -20.0]
    else:
        cluster_affinity =  current_cluster.loc[out_table["score"] <= -20.0]
    return cluster_affinity

with pd.ExcelWriter("./high_affinity_motifs.xlsx", engine="openpyxl", mode = "w") as writer:
    for x in range(1, 7):
        current_cluster_high_affinity = process_clusters(out_table,affinity_high=True)
        current_cluster_high_affinity.to_excel(excel_writer=writer, sheet_name=f"cluster_{x}")
with pd.ExcelWriter("./low_affinity_motifs.xlsx", engine="openpyxl", mode = "w") as writer:
    for x in range(1, 7):
        current_cluster_high_affinity = process_clusters(out_table,affinity_high=False)
        current_cluster_high_affinity.to_excel(excel_writer=writer, sheet_name=f"cluster_{x}")