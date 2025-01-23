import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import re

print("Loading Jaccard similarity data...")
jaccard_df = pd.read_excel(r"D:\Fuzzy\second_feature.xlsx", header=0, index_col=0)
protein_names = jaccard_df.index.values
jaccard_similarity = jaccard_df.values

def process_psiblast_out(file_path, output_csv_path):
    columns = ['query_acc', '% identity', 'alignment length', 'mismatches', 'gap opens', 'bit score', 'evalue']
    with open(file_path, 'r') as file, open(output_csv_path, 'w') as out_file:
        out_file.write(','.join(columns) + '\n')  # Write header
        for line in tqdm(file, desc="Processing PSI-BLAST file"):
            parts = line.split('\t')
            if len(parts) > 10:
                try:
                    query_acc = parts[0]
                    identity = float(parts[2])
                    alignment_length = int(parts[3])
                    mismatches = int(parts[4])
                    gap_opens = int(parts[5])
                    bit_score = float(parts[11])
                    evalue = float(parts[10])
                    out_file.write(
                        f"{query_acc},{identity},{alignment_length},{mismatches},{gap_opens},{bit_score},{evalue}\n"
                    )
                except ValueError:
                    continue

    return pd.read_csv(output_csv_path)

def process_oma_file(oma_file_path):
    oma_scores = {}
    with open(oma_file_path, 'r') as file:
        for line in tqdm(file, desc="Processing OMA file"):
            if not line.startswith("!"): 
                parts = line.split('\t')
                protein_id = parts[1] 
                score_match = re.search(r':([\d.]+)$', parts[7])  
                if score_match:
                    oma_score = float(score_match.group(1))
                    oma_scores[protein_id] = oma_score
    return oma_scores

psiblast_file_path = r"D:\Project outputs\results.out"
processed_psiblast_path = r"D:\Project outputs\processed_psiblast.csv"
print("Processing PSI-BLAST output data...")
df_psiblast = process_psiblast_out(psiblast_file_path, processed_psiblast_path)

oma_file_path = r"D:\Project outputs\OMA_output (1).txt"
print("Processing OMA output data...")
oma_scores = process_oma_file(oma_file_path)

oma_jaccard_combined = []
for i, query_acc in enumerate(protein_names):
    jaccard_values = jaccard_similarity[i, :]
    jaccard_mean = np.mean(jaccard_values)
    
    oma_score = oma_scores.get(query_acc, np.nan)
    if np.isnan(oma_score):
        oma_score = np.nanmean(list(oma_scores.values()))
    
    combined_score = 0.5 * jaccard_mean + 0.5 * oma_score  
    oma_jaccard_combined.append(combined_score)

combined_output_path = r"D:\Fuzzy\combined_data.csv"
columns = ['query_acc', '% identity', 'alignment length', 'mismatches', 
           'gap opens', 'bit score', 'evalue', 'OMA-Jaccard Combined']

print("Combining data from Jaccard, PSI-BLAST, and OMA sources...")
with open(combined_output_path, 'w') as out_file:
    out_file.write(','.join(columns) + '\n')
    for i, query_acc in tqdm(enumerate(protein_names), desc="Combining data for each protein", total=len(protein_names)):
        if query_acc in df_psiblast['query_acc'].values:
            psiblast_rows = df_psiblast[df_psiblast['query_acc'] == query_acc]
            
            identity_mean = psiblast_rows['% identity'].mean()
            alignment_length_mean = psiblast_rows['alignment length'].mean()
            mismatches_mean = 1 / (psiblast_rows['mismatches'].mean() + 1e-5) 
            gap_opens_mean = psiblast_rows['gap opens'].mean()
            bit_score_mean = psiblast_rows['bit score'].mean()
            evalue_mean = 1 / (psiblast_rows['evalue'].mean() + 1e-5)
            
            combined_score = oma_jaccard_combined[i]
            
            out_file.write(
                f"{query_acc},{identity_mean},{alignment_length_mean},{mismatches_mean},"
                f"{gap_opens_mean},{bit_score_mean},{evalue_mean},{combined_score}\n"
            )

print("Loading combined data for clustering...")
combined_df = pd.read_csv(combined_output_path)

combined_df['OMA-Jaccard Combined'].fillna(combined_df['OMA-Jaccard Combined'].mean(), inplace=True)

print("Scaling data for clustering...")
scaler = MinMaxScaler()
combined_scaled = scaler.fit_transform(combined_df.iloc[:, 1:]) 

n_clusters = 5 

print("Running Fuzzy C-Means clustering on scaled data...")

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    combined_scaled.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
)

for i in range(n_clusters):
    combined_df[f'Membership Cluster {i+1}'] = u[i]

combined_df['Max Membership'] = np.max(u, axis=0)
combined_df['Assigned Cluster'] = np.argmax(u, axis=0)

output_path = r"D:\Fuzzy\third_cluster_results(n=5).out"
print(f"Saving clustering results to {output_path}...")
combined_df.to_csv(output_path, sep='\t', index=False)

print(f"Clustering completed. Results saved to {output_path} as a .out file.")
