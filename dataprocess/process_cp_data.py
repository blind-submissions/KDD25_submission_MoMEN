import os
import csv
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import PandasTools
with open('data_config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def standardize_smiles(smiles):
    if pd.isnull(smiles) or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None
#  Load chemical annotations
chemical_annotations_path = config['chemical_annotations_file']
chemical_annotations = pd.read_csv(chemical_annotations_path)
smiles_dict = pd.Series(chemical_annotations.CPD_SMILES.values, index=chemical_annotations.BROAD_ID).to_dict()

profiles_dir = config['profiles_dir']
plates = [d for d in os.listdir(profiles_dir) if os.path.isdir(os.path.join(profiles_dir, d))]
samples_dict = {}
sample_id = 1 

for plate in plates:
    profiles_csv_path = os.path.join(profiles_dir, plate, 'profiles', 'mean_well_profiles.csv')
    if os.path.exists(profiles_csv_path):
        profiles_df = pd.read_csv(profiles_csv_path)
        
        for _, row in profiles_df.iterrows():
            broad_id = row['Metadata_broad_sample']
            smiles = smiles_dict.get(broad_id)  # Get SMILES from the dictionary without default None
            
            if smiles:
                samples_dict[sample_id] = {
                    'plate_id': row['Metadata_Plate'],
                    'well_id': row['Metadata_Well'],
                    'smiles': standardize_smiles(smiles)
                }
                sample_id += 1


# Delete wells of incomplete views
if complete_views:
    required_views = ['s1', 's2', 's3', 's4', 's5', 's6']
    samples_to_remove = []
    img_path = config['data_root']
    for sample_id, sample_info in samples_dict.items():
        plate_id = sample_info['plate_id']
        well_id = sample_info['well_id']
        
        all_views_present = True
        for view in required_views:
            image_path = os.path.join(img_path, plate_id, f'{well_id}-{view}.npy')
            if not os.path.exists(image_path):
                all_views_present = False
                break
        
        if not all_views_present:
            samples_to_remove.append(sample_id)

    for sample_id in samples_to_remove:
        del samples_dict[sample_id]


# Delete repeated records
if unique_smiles:
    unique_smiles_set = set()
    samples_to_remove = []

    for sample_id, sample_info in samples_dict.items():
        smiles = sample_info['smiles']
        
        if smiles in unique_smiles_set:
            samples_to_remove.append(sample_id)
        else:
            unique_smiles_set.add(smiles)

    for sample_id in samples_to_remove:
        del samples_dict[sample_id]



output_csv_path = '../data/cp_smiles_paths.csv'

with open(output_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['SMILES', 'image_path']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for sample_info in samples_dict.values():
        smiles = sample_info['smiles']
        plate_id = sample_info['plate_id']
        well_id = sample_info['well_id']
        
        for s in range(1, 7):
            image_path = f"{plate_id}/{well_id}-s{s}.npy"
            writer.writerow({'SMILES': smiles, 'image_path': image_path})

print(f"Done!")