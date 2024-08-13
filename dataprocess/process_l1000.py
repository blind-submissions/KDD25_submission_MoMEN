import os
import yaml
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

with open('data_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

raw_path = config['raw_data_root']
def get_l1k_data(datapath):
    df = pd.read_csv(os.path.join(raw_path, datapath), compression='gzip') 

    gene_expression_columns = df.columns[0:977] 
    gene_expression_data = df[gene_expression_columns]

    df['gene_expression_data'] = gene_expression_data.apply(lambda row: row.to_numpy(), axis=1)


    df['pert_dose'] = df['pert_dose']

    def standardize_smiles(smiles):
        if pd.isnull(smiles) or not isinstance(smiles, str):
            return None
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None

    df['SMILES'] = df['CPD_SMILES'].apply(standardize_smiles)

    combined_df = df[['gene_expression_data', 'pert_dose', 'SMILES']]
    combined_df.dropna(subset=['SMILES'], inplace=True)
    return combined_df

df047 = get_l1k_data('CDRP-BBBC047-Bray/L1000/replicate_level_l1k.csv.gz')
df036 = get_l1k_data('CDRPBIO-BBBC036-Bray/L1000/replicate_level_l1k.csv.gz')
stacked_df = pd.concat([df047, df036], axis=0)
stacked_df.reset_index(drop=True, inplace=True)
stacked_df.to_pickle('../data/l1k.pkl')

sms = stacked_df['SMILES'].unique()
unique_smiles_df = pd.DataFrame(sms, columns=['SMILES'])

unique_smiles_df.to_csv('../data/l1k_smiles.csv', index=False)

# For gene-graph alignment
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(unique_smiles_df, test_size=0.3, random_state=42)
validation_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=42)
train_csv_path = '../data/gene_smiles/train.csv'
validation_csv_path = '../gene_smiles/valid.csv'
test_csv_path = '../gene_smiles/test.csv'

train_df.to_csv(train_csv_path, index=False)
validation_df.to_csv(validation_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)