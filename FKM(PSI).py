import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

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

file_path = r"C:\Users\arjun\Downloads\results.out"
df = process_psiblast_out(file_path)

aggregated_df = df.groupby('query_acc').mean().reset_index()

aggregated_df['mismatches'] = 1 / (aggregated_df['mismatches'] + 1e-5)
aggregated_df['evalue'] = 1 / (aggregated_df['evalue'] + 1e-5)

scaler = MinMaxScaler()
aggregated_scaled = scaler.fit_transform(aggregated_df.iloc[:, 1:]) 

n_clusters = 7

print("Running Fuzzy K-Means clustering...")
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(aggregated_scaled.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)

for i in range(n_clusters):
    aggregated_df[f'Membership Cluster {i+1}'] = u[i]

aggregated_df['Max Membership'] = np.max(u, axis=0)
aggregated_df['Assigned Cluster'] = np.argmax(u, axis=0)

output_path = r"D:\Fuzzy\final(n=7).out"

with open(output_path, 'w') as file:
    header = '\t'.join(aggregated_df.columns) + '\n'
    file.write(header)
    
    for i in tqdm(range(len(aggregated_df)), desc="Writing clustered data to .out file"):
        row = '\t'.join(map(str, aggregated_df.iloc[i])) + '\n'
        file.write(row)

print(f"Clustering completed. Results saved to {output_path} as a .out file.")
