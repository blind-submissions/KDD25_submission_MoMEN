import pandas as pd 


gene_smiles = pd.read_csv('../data/l1k_smiles.csv')

cp_smiles = pd.read_csv('../data/cp_smiles_paths.csv')
union_series = pd.concat([gene_smiles['SMILES'], cp_smiles['SMILES']]).drop_duplicates()

union_df = union_series.to_frame(name='SMILES')
union_df.to_csv('../data/retrieval/union.csv',index = False)

overlap_smiles_set = gene_smiles_set.intersection(cp_smiles_set)
overlap_df = pd.DataFrame(list(overlap_smiles_set), columns=['SMILES'])
overlap_df.to_csv('../data/retrieval/overlap.csv', index=False)

from sklearn.model_selection import train_test_split
valid_data, train_data_overlap = train_test_split(overlap_df, test_size=0.3, random_state=42)
test_data, valid_data = train_test_split(valid_data, test_size=2/3, random_state=42)
train_data = union_df[~union_df['SMILES'].isin(valid_data['SMILES']) & ~union_df['SMILES'].isin(test_data['SMILES'])]

valid_data.to_csv('../data/retrieval/valid.csv', index=False)
test_data.to_csv('../data/retrieval/test.csv', index=False)
train_data.to_csv('../data/retrieval/train.csv', index=False)
train_data_overlap.to_csv('../data/retrieval/train_overlap.csv', index=False)