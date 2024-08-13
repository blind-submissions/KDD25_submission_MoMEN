import torch
from torch.utils.data import Dataset
import pandas as pd 
from itertools import repeat
from torch_geometric.data import Data, Batch
import os
import numpy as np
import itertools


class GraphGeneDataset(Dataset):
    '''
    gene_df contrains all the molecules. 
    '''
    def __init__(self, gene_df, smiles_df, processed_graph_path, processed_smiles_path, is_train = True ):

        if is_train:
            gene_df = gene_df[~gene_df['SMILES'].isin(smiles_df['SMILES'])]
        else:
            gene_df = gene_df[gene_df['SMILES'].isin(smiles_df['SMILES'])]
            gene_df = gene_df.drop_duplicates(subset=['SMILES'])

        self.graphs, self.slices = torch.load(processed_graph_path)
        smiles_gene_dict = gene_df.groupby('SMILES').apply(lambda x: list(zip(x['gene_expression_data'], x['pert_dose']))).to_dict()

        CID_text_df = pd.read_csv(processed_smiles_path)
        processed_smiles_list = CID_text_df["smiles"].tolist()

        temp_smiles_to_graph = {}
        for idx, smiles in enumerate(processed_smiles_list):
            data = self._create_data_object(self.graphs, self.slices, idx)
            temp_smiles_to_graph[smiles] = data

        self.triples = []
        for smiles in set(processed_smiles_list):
            if smiles in smiles_gene_dict:
                for gene_expression_data, dose in smiles_gene_dict[smiles]:
                    self.triples.append((smiles, temp_smiles_to_graph[smiles], gene_expression_data, dose))


    def _create_data_object(self, graphs, slices, idx):
        data = Data()
        for key in graphs.keys:
            item, slice_idx = graphs[key], slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slice_idx[idx], slice_idx[idx + 1])
            data[key] = item[s]
        return data

    def __getitem__(self, idx):
        SMILES, graph, gene_expression_data, dose = self.triples[idx]
        return graph, gene_expression_data, dose

    def __len__(self):
        return len(self.triples)

def collate_mol_gene_fn(batch):
    batch_graph = Batch.from_data_list([item[0] for item in batch])
    batch_gene = torch.stack([torch.tensor(item[1], dtype = torch.float32) for item in batch], dim=0)
    dose = torch.stack([torch.tensor(item[-1]) for item in batch], dim=0)
    return {
        'gene': batch_gene,
        'dose': dose,
        'mols': batch_graph
    }


class GraphImageDataset(Dataset):
    def __init__(self, root, smiles_image_path_file, smiles_df, processed_graph_path, processed_smiles_path, transform, is_train):
        self.root = root
        
        path_df = pd.read_csv(smiles_image_path_file)
        if is_train:
            path_df = path_df[~path_df['SMILES'].isin(smiles_df['SMILES'])]
        else:
            path_df = path_df[path_df['SMILES'].isin(smiles_df['SMILES'])]
            path_df = path_df.drop_duplicates(subset=['SMILES'])


        smiles_image_dict = path_df.groupby('SMILES')['image_path'].apply(list).to_dict()


        self.graphs, self.slices = torch.load(processed_graph_path)
        CID_text_df = pd.read_csv(processed_smiles_path)
        processed_smiles_list = CID_text_df["smiles"].tolist()

        temp_smiles_to_graph = {}
        for idx, smiles in enumerate(processed_smiles_list):
            data = self._create_data_object(self.graphs, self.slices, idx)
            temp_smiles_to_graph[smiles] = data

        self.triples = []
        for smiles in set(processed_smiles_list):
            if smiles in smiles_image_dict and smiles in temp_smiles_to_graph:
                for sample_key in smiles_image_dict[smiles]:
                    self.triples.append((smiles, temp_smiles_to_graph[smiles], sample_key))

        self.transform = transform

    def _create_data_object(self, graphs, slices, idx):
        data = Data()
        for key in graphs.keys:
            item, slice_idx = graphs[key], slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slice_idx[idx], slice_idx[idx + 1])
            data[key] = item[s]
        return data

    def __getitem__(self, idx):
        SMILES, graph, sample_key = self.triples[idx]
        image_path = os.path.join(self.root, sample_key)
        image = np.load(image_path, allow_pickle=True).astype(np.float32)
        return graph, process_image(image, self.transform)

    def __len__(self):
        return len(self.triples)

        
def process_image(image_np, transform):
    return transform(image_np)

def collate_mol_img_fn(batch):
    batch_graph = Batch.from_data_list([item[0] for item in batch])
    batch_image = torch.stack([item[1] for item in batch], dim=0)
    return {
        'images': batch_image,
        'mols': batch_graph
    }


class GeneImageDataset(Dataset):
    def __init__(self, gene_df, image_paths_df, root_dir, transform, smiles_csv_path, is_train = True):
        smiles_filter_df = pd.read_csv(smiles_csv_path)
        valid_smiles_set = set(smiles_filter_df['SMILES'])

        gene_df = gene_df[gene_df['SMILES'].isin(valid_smiles_set)]
        image_paths_df = image_paths_df[image_paths_df['SMILES'].isin(valid_smiles_set)]

        if not is_train : #test
            gene_df = gene_df.drop_duplicates(subset=['SMILES'])
            image_paths_df = image_paths_df.drop_duplicates(subset=['SMILES'])

        smiles_gene_dict = gene_df.groupby('SMILES').apply(lambda x: list(zip(x['gene_expression_data'], x['pert_dose']))).to_dict()
        smiles_image_dict = image_paths_df.groupby('SMILES')['image_path'].apply(list).to_dict()

        self.triples = []
        for smiles in valid_smiles_set:
            if smiles in smiles_gene_dict and smiles in smiles_image_dict:
                # for gene_expression_data in smiles_gene_dict[smiles]:
                for gene_expression_data, code_index in smiles_gene_dict[smiles]:
                    for sample_key in smiles_image_dict[smiles]:
                        self.triples.append((smiles, gene_expression_data, code_index, sample_key))

        self.root = root_dir
        self.transform = transform

    def __getitem__(self, idx):
        smiles, gene_expression_data, code_index, sample_key = self.triples[idx]
        image_path = os.path.join(self.root, sample_key)
        image = np.load(image_path, allow_pickle=True).astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return gene_expression_data, code_index, image

    def __len__(self):
        return len(self.triples)

def collate_gene_image_fn(batch):
    batch_gene = torch.stack([torch.tensor(item[0], dtype = torch.float32) for item in batch], dim=0)
    batch_dose = torch.stack([torch.tensor(item[1]) for item in batch], dim=0)
    batch_image = torch.stack([item[-1] for item in batch], dim=0)
    return {
        'gene': batch_gene,
        'dose': batch_dose,
        'images': batch_image
    }


class TripLetDataset(Dataset):
    def __init__(self, gene_df, image_paths_df, img_root_dir, transform, smiles_csv_path, processed_graph_path, processed_smiles_path, is_train = True):
        valid_smiles_set = set(pd.read_csv(smiles_csv_path)['SMILES'])

        gene_df = gene_df[gene_df['SMILES'].isin(valid_smiles_set)]
        image_paths_df = image_paths_df[image_paths_df['SMILES'].isin(valid_smiles_set)]

        if not is_train : #test
            gene_df = gene_df.drop_duplicates(subset=['SMILES'])
            image_paths_df = image_paths_df.drop_duplicates(subset=['SMILES'])

        smiles_gene_dict = gene_df.groupby('SMILES').apply(lambda x: list(zip(x['gene_expression_data'], x['pert_dose']))).to_dict()
        smiles_image_dict = image_paths_df.groupby('SMILES')['image_path'].apply(list).to_dict()

        self.graphs, self.slices = torch.load(processed_graph_path)
        CID_text_df = pd.read_csv(processed_smiles_path)
        processed_smiles_list = CID_text_df["smiles"].tolist()

        temp_smiles_to_graph = {}
        for idx, smiles in enumerate(processed_smiles_list):
            data = self._create_data_object(self.graphs, self.slices, idx)
            temp_smiles_to_graph[smiles] = data

        self.triples = []
        for smiles in valid_smiles_set:
            if smiles in smiles_gene_dict and smiles in smiles_image_dict:
                for gene_expression_data, code_index in smiles_gene_dict[smiles]:
                    for sample_key in smiles_image_dict[smiles]:
                        self.triples.append((temp_smiles_to_graph[smiles], gene_expression_data, code_index, sample_key))

        self.root = img_root_dir
        self.transform = transform

    def _create_data_object(self, graphs, slices, idx):
        data = Data()
        for key in graphs.keys:
            item, slice_idx = graphs[key], slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slice_idx[idx], slice_idx[idx + 1])
            data[key] = item[s]
        return data

    def __getitem__(self, idx):
        graph, gene_expression_data, code_index, sample_key = self.triples[idx]
        image_path = os.path.join(self.root, sample_key)
        image = np.load(image_path, allow_pickle=True).astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return graph, gene_expression_data, code_index, image

    def __len__(self):
        return len(self.triples)


def collate_triplet_fn(batch):
    batch_graph = Batch.from_data_list([item[0] for item in batch])
    batch_gene = torch.stack([torch.tensor(item[1], dtype = torch.float32) for item in batch], dim=0)
    batch_dose = torch.stack([torch.tensor(item[2]) for item in batch], dim=0)
    batch_image = torch.stack([item[-1] for item in batch], dim=0)
    return {
        'gene': batch_gene,
        'dose': batch_dose,
        'images': batch_image,
        'mols': batch_graph
    }



class TripLetMaskDataset(Dataset):
    def __init__(self, gene_df, image_paths_df, img_root_dir, transform, smiles_csv_path, processed_graph_path, processed_smiles_path, is_train=True):
        valid_smiles_set = set(pd.read_csv(smiles_csv_path)['SMILES'])

        if not is_train : #test
            gene_df = gene_df.drop_duplicates(subset=['SMILES'])
            image_paths_df = image_paths_df.drop_duplicates(subset=['SMILES'])

        smiles_gene_dict = gene_df.groupby('SMILES').apply(lambda x: list(zip(x['gene_expression_data'], x['pert_dose']))).to_dict()
        smiles_image_dict = image_paths_df.groupby('SMILES')['image_path'].apply(list).to_dict()

        self.graphs, self.slices = torch.load(processed_graph_path)
        CID_text_df = pd.read_csv(processed_smiles_path)
        processed_smiles_list = CID_text_df["smiles"].tolist()

        temp_smiles_to_graph = {}
        for idx, smiles in enumerate(processed_smiles_list):
            data = self._create_data_object(self.graphs, self.slices, idx)
            temp_smiles_to_graph[smiles] = data

        self.triples = []
        self.gene_mask = []
        self.image_mask = []

        gene_expression_placeholder = np.zeros((977,), dtype=np.float32)
        image_placeholder = np.zeros((5, 360, 480), dtype=np.float32)

        for smiles in valid_smiles_set:
            graph_data = temp_smiles_to_graph[smiles]
            gene_data_list = smiles_gene_dict.get(smiles, [(gene_expression_placeholder, 0.0)])  # Default to placeholder and 0.0
            image_path_list = smiles_image_dict.get(smiles, [image_placeholder])  # Default to placeholder

            for gene_data, image_path in itertools.product(gene_data_list, image_path_list):
                gene_expression, code_index = gene_data
                gene_mask = 1 if np.any(gene_expression) else 0  # Check if gene_expression is not the placeholder
                image_mask = 1 if image_path is not image_placeholder else 0  # Check if image_path is not the placeholder

                self.triples.append((graph_data, gene_expression, code_index, image_path))
                self.gene_mask.append(gene_mask)
                self.image_mask.append(image_mask)

        self.root = img_root_dir
        self.transform = transform

    def _create_data_object(self, graphs, slices, idx):
        data = Data()
        for key in graphs.keys:
            item, slice_idx = graphs[key], slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slice_idx[idx], slice_idx[idx + 1])
            data[key] = item[s]
        return data

    def __getitem__(self, idx):
        graph, gene_expression_data, code_index, image_path = self.triples[idx]
        gene_mask = self.gene_mask[idx]
        image_mask = self.image_mask[idx]
        assert gene_mask != 0 or image_mask != 0, "Both gene_mask and image_mask cannot be zero at the same time."

        if image_mask:
            image_path = os.path.join(self.root, image_path)
            image = np.load(image_path, allow_pickle=True).astype(np.float32)
            if self.transform:
                image = self.transform(image).clone().detach()
        else:
            image = torch.zeros((5, 360, 480), dtype=torch.float32)  # Use the zero-filled image if there's no image data

        return graph, gene_expression_data, code_index, image, gene_mask, image_mask

    def __len__(self):
        return len(self.triples)


def collate_triplet_ms_fn(batch):
    batch_graph = Batch.from_data_list([item[0] for item in batch])
    batch_gene = torch.stack([torch.tensor(item[1], dtype=torch.float32) for item in batch], dim=0)
    batch_dose = torch.stack([torch.tensor(item[2], dtype=torch.float32) for item in batch], dim=0)
    batch_image = torch.stack([torch.tensor(item[3], dtype=torch.float32) for item in batch], dim=0)
    batch_gene_mask = torch.stack([torch.tensor(item[4], dtype=torch.float32) for item in batch], dim=0)
    batch_image_mask = torch.stack([torch.tensor(item[5], dtype=torch.float32) for item in batch], dim=0)

    return {
        'gene': batch_gene,
        'dose': batch_dose,
        'images': batch_image,
        'mols': batch_graph,
        'gene_mask': batch_gene_mask,
        'image_mask': batch_image_mask
    }