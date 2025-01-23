import os
import pandas as pd
from goatools.obo_parser import GODag
from scipy.spatial.distance import pdist, squareform

obo_file = r"C:\Users\rohan\Downloads\go-basic.obo"
go_slim_file = r"C:\Users\rohan\Downloads\goslim_generic.obo"
gaf_file = r"C:\Users\rohan\Downloads\OMA_output (1).txt"

go = GODag(obo_file)
slim = GODag(go_slim_file)

def read_gaf(file_path):
    df = pd.read_csv(file_path, sep='\t', comment='!', header=None)
    print("GAF File Structure:")
    print(df.head()) 
    return df

gaf_df = read_gaf(gaf_file)

def map_go_to_slim(go, slim, go_id):
    if go_id in slim:
        return go_id

    def find_slim_term(term_id):
        term = go.get(term_id)
        if term:
            if any(parent.id in slim for parent in term.parents):
                return term_id
            for parent in term.parents:
                result = find_slim_term(parent.id)
                if result:
                    return result
        return None

    return find_slim_term(go_id)

protein_go_slim_mappings = {}
for index, row in gaf_df.iterrows():
    protein = row[1]  
    go_id = row[4]   
    slim_mapping = map_go_to_slim(go, slim, go_id)

    if slim_mapping:
        if protein not in protein_go_slim_mappings:
            protein_go_slim_mappings[protein] = set()  
        protein_go_slim_mappings[protein].add(slim_mapping)

proteins = list(protein_go_slim_mappings.keys())
go_slim_terms = sorted({term for terms in protein_go_slim_mappings.values() for term in terms})
matrix_data = [[1 if slim_term in protein_go_slim_mappings[protein] else 0 for slim_term in go_slim_terms] for protein in proteins]

df = pd.DataFrame(matrix_data, index=proteins, columns=go_slim_terms)

def compute_jaccard_similarity(df):
    dist_matrix = pdist(df, metric='jaccard')
    similarity_matrix = 1 - squareform(dist_matrix) 
    return similarity_matrix

similarity_matrix = compute_jaccard_similarity(df)

similarity_df = pd.DataFrame(similarity_matrix, index=proteins, columns=proteins)


print("Functional Relationship Scores (Jaccard Similarity):")
print(similarity_df)

downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
output_file = os.path.join(downloads_path, "functional_relationship_scores.csv")
similarity_df.to_csv(output_file)

print(f"Functional relationship scores saved to '{output_file}'.")
