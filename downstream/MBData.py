from rdkit import Chem
import os
import torch
import numpy as np
import pandas as pd
from itertools import repeat
from torch_geometric.data import Data, Batch, InMemoryDataset
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from tqdm import *


def mol_to_graph_data_obj_simple(mol):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)  
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond) 


            edges_list.append((i, j))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data



def collate_fn(batch):
    batch_graph = Batch.from_data_list([item[1] for item in batch])
    batch_label = torch.tensor([item[0] for item in batch], dtype = torch.float32)
    return {
        'graph': batch_graph,
        'label': batch_label
    }

class MetabolismDataset(InMemoryDataset):
    def __init__(self, name, df, stage = "train", subset_size=None, transform=None, pre_transform=None, pre_filter=None):
        self.datasetname = 'MB_' + name + stage
        self.SMILES_list = df['Drug'].tolist()
        self.labels = df['Y'].tolist()
        root = None
        super(MetabolismDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.graphs, self.slices = torch.load(self.processed_paths[0])
        return

    @property
    def processed_dir(self):
        return './data/'

    @property
    def processed_file_names(self):
        return self.datasetname+'.pt'

    def process(self):
        graph_list = []
        for SMILES in tqdm(self.SMILES_list):
            RDKit_mol = Chem.MolFromSmiles(SMILES)
            graph = mol_to_graph_data_obj_simple(RDKit_mol)
            graph_list.append(graph)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])
        return

    def get(self, idx):
        SMILES = self.SMILES_list[idx]

        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        label = self.labels[idx]  

        return label, data

    def __len__(self):
        return len(self.SMILES_list)




import pandas as pd

def normalize_y(split):
    all_y_values = pd.concat([df['Y'] for df in split.values()])
    
    mean_y = all_y_values.mean()
    std_y = all_y_values.std()
    
    normalized_split = {}
    for key, df in split.items():
        normalized_df = df.copy()
        normalized_df['Y'] = (df['Y'] - mean_y) / std_y
        normalized_split[key] = normalized_df
    
    return normalized_split, mean_y, std_y
