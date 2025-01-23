import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

print("Loading Jaccard similarity data...")
jaccard_df = pd.read_excel(r"D:\FuzzyKMeans\second_feature.xlsx", header=0, index_col=0) 
protein_names = jaccard_df.index.values
jaccard_similarity = jaccard_df.values

def process_psiblast_out(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Processing PSI-BLAST file"):
            parts = line.split('\t')
            if len(parts) > 10:
                query_acc = parts[0]
                subject_acc = parts[1]
                identity = float(parts[2])
                alignment_length = int(parts[3])
                mismatches = int(parts[4])
                gap_opens = int(parts[5])
                bit_score = float(parts[11])
                evalue = float(parts[10])
                data.append([query_acc, identity, alignment_length, mismatches, gap_opens, bit_score, evalue])

    columns = ['query_acc', '% identity', 'alignment length', 'mismatches', 'gap opens', 'bit score', 'evalue']
    return pd.DataFrame(data, columns=columns)

file_path = r"D:\FuzzyKMeans\results (1).out" 
df_psiblast = process_psiblast_out(file_path)

combined_data = []
for i, query_acc in enumerate(protein_names):
    if query_acc in df_psiblast['query_acc'].values:
        psiblast_rows = df_psiblast[df_psiblast['query_acc'] == query_acc]
        
        jaccard_values = jaccard_similarity[i, :]
        
        identity_mean = psiblast_rows['% identity'].mean()
        alignment_length_mean = psiblast_rows['alignment length'].mean()
        mismatches_mean = 1 / (psiblast_rows['mismatches'].mean() + 1e-5) 
        gap_opens_mean = psiblast_rows['gap opens'].mean()
        bit_score_mean = psiblast_rows['bit score'].mean()
        evalue_mean = 1 / (psiblast_rows['evalue'].mean() + 1e-5) 
        
        jaccard_mean = np.mean(jaccard_values)

        combined_data.append([query_acc, jaccard_mean, identity_mean, alignment_length_mean, mismatches_mean, gap_opens_mean, bit_score_mean, evalue_mean])

combined_df = pd.DataFrame(combined_data, columns=['query_acc', 'mean_jaccard', '% identity', 'alignment length', 'mismatches', 'gap opens', 'bit score', 'evalue'])

scaler = MinMaxScaler()
combined_scaled = scaler.fit_transform(combined_df.iloc[:, 1:]) 

n_clusters = 5 

print("Running Fuzzy K-Means clustering on combined data...")
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(combined_scaled.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)

for i in range(n_clusters):
    combined_df[f'Membership Cluster {i+1}'] = u[i]

combined_df['Max Membership'] = np.max(u, axis=0)
combined_df['Assigned Cluster'] = np.argmax(u, axis=0)

output_path = r"D:\FuzzyKMeans\final_combined_clusters(n=5).out" 

with open(output_path, 'w') as file:
    header = '\t'.join(combined_df.columns) + '\n'
    file.write(header)
    for i in tqdm(range(len(combined_df)), desc="Writing combined clustered data to .out file"):
        row = '\t'.join(map(str, combined_df.iloc[i])) + '\n'
        file.write(row)

print(f"Clustering completed. Results saved to {output_path} as a .out file.")
