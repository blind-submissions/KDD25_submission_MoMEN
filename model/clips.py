import torch
import torch.nn as nn
import numpy as np
from encoders import MoE


class GICLIP(nn.Module):
    def __init__(self, embed_dim: int, image_encoder, gene_encoder, img_dim, mol_dim, T = 0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.gene_encoder = gene_encoder  

        self.gene_proj = nn.Linear(mol_dim, embed_dim)
        self.image_proj = nn.Linear(img_dim, embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / T))

    def encode_img(self, images):
        return self.image_encoder(images)

    def encode_gene(self, gene, gene_code):
        return self.gene_encoder(gene, gene_code)

    def forward(self, **kwargs):
        images = kwargs['images']
        gene = kwargs['gene']
        gene_code = kwargs['dose'] 

        image_features = self.image_proj(self.encode_img(images))
        mol_features = self.gene_proj(self.encode_gene(gene, gene_code))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        mol_features = mol_features / mol_features.norm(dim=-1, keepdim=True)

        return image_features, mol_features, self.logit_scale.exp()


class TCLIP(nn.Module):
    def __init__(self, embed_dim: int, mol_encoder, gene_encoder, image_encoder, mol_dim, gene_dim, img_dim, T = 0.07, mode = 'III'):
        super().__init__()
        self.gene_encoder = gene_encoder
        self.image_encoder = image_encoder
        self.mol_encoder = mol_encoder  

        self.mol_proj = nn.Linear(mol_dim, embed_dim)
        self.gene_proj = nn.Linear(gene_dim, embed_dim)
        self.image_proj = nn.Linear(img_dim, embed_dim)
        self.mode = mode
        self.logit_scale_g = nn.Parameter(torch.ones([]) * np.log(1 / T))
        self.logit_scale_i = nn.Parameter(torch.ones([]) * np.log(1 / T))
        self.logit_scale_gi = nn.Parameter(torch.ones([]) * np.log(1 / T))

    def encode_img(self, images):
        return self.image_encoder(images)

    def encode_gene(self, gene, gene_code):
        return self.gene_encoder(gene, gene_code)

    def encode_mol(self, mols):
        return self.mol_encoder(mols)

    def forward(self, **kwargs):
        images = kwargs['images']
        mols = kwargs['mols']

        gene = kwargs['gene']
        genecode = kwargs['dose']

        image_features = self.image_proj(self.encode_img(images))
        mol_features = self.mol_proj(self.encode_mol(mols))

        gene_features = self.gene_proj(self.encode_gene(gene, genecode))


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        mol_features = mol_features / mol_features.norm(dim=-1, keepdim=True)
        gene_features = gene_features / gene_features.norm(dim=-1, keepdim=True)


        return image_features, mol_features, gene_features, self.logit_scale_i.exp(), self.logit_scale_g.exp(), self.logit_scale_gi.exp()
 

class TCLIPMoE(nn.Module):
    def __init__(self, embed_dim: int, mol_encoder, gene_encoder, image_encoder, mol_dim, gene_dim, img_dim, T = 0.07, mode = 'III'):
        super().__init__()
        self.gene_encoder = gene_encoder
        self.image_encoder = image_encoder
        self.mol_encoder = mol_encoder  

        self.mol_proj = MoE(mol_dim, embed_dim) 
        self.gene_proj = MoE(gene_dim, embed_dim)
        self.image_proj = MoE(img_dim, embed_dim)
        self.mode = mode
        self.logit_scale_g = nn.Parameter(torch.ones([]) * np.log(1 / T))
        self.logit_scale_i = nn.Parameter(torch.ones([]) * np.log(1 / T))
        self.logit_scale_gi = nn.Parameter(torch.ones([]) * np.log(1 / T))

    def encode_img(self, images):
        return self.image_encoder(images)

    def encode_gene(self, gene, gene_code):
        return self.gene_encoder(gene, gene_code)

    def encode_mol(self, mols):
        return self.mol_encoder(mols)

    def forward(self, **kwargs):
        images = kwargs['images']
        mols = kwargs['mols']

        gene = kwargs['gene']
        genecode = kwargs['dose']

        image_features = self.image_proj(self.encode_img(images))
        mol_features = self.mol_proj(self.encode_mol(mols))

        gene_features = self.gene_proj(self.encode_gene(gene, genecode))


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        mol_features = mol_features / mol_features.norm(dim=-1, keepdim=True)
        gene_features = gene_features / gene_features.norm(dim=-1, keepdim=True)

        if self.mode == 'II':

            return image_features, mol_features, gene_features, self.logit_scale_i.exp(), self.logit_scale_g.exp()
        else:
            return image_features, mol_features, gene_features, self.logit_scale_i.exp(), self.logit_scale_g.exp(), self.logit_scale_gi.exp()
 

class TCLIPMoED(nn.Module):
    def __init__(self, embed_dim: int, mol_encoder, gene_encoder, image_encoder, mol_dim, gene_dim, img_dim, T = 0.07):
        super().__init__()
        self.gene_encoder = gene_encoder
        self.image_encoder = image_encoder
        self.mol_encoder = mol_encoder  

        self.mol_proj = MoE(mol_dim, embed_dim) 
        self.gene_proj = MoE(gene_dim, embed_dim)
        self.image_proj = MoE(img_dim, embed_dim)
        self.logit_scale_gi = nn.Parameter(torch.ones([]) * np.log(1 / T))
        self.logit_scale_ig = nn.Parameter(torch.ones([]) * np.log(1 / T))
        self.logit_scale_gm = nn.Parameter(torch.ones([]) * np.log(1 / T))
        self.logit_scale_mg = nn.Parameter(torch.ones([]) * np.log(1 / T))
        self.logit_scale_mi = nn.Parameter(torch.ones([]) * np.log(1 / T))
        self.logit_scale_im = nn.Parameter(torch.ones([]) * np.log(1 / T))

    def encode_img(self, images):
        return self.image_encoder(images)

    def encode_gene(self, gene, gene_code):
        return self.gene_encoder(gene, gene_code)

    def encode_mol(self, mols):
        return self.mol_encoder(mols)

    def forward(self, **kwargs):
        images = kwargs['images']
        mols = kwargs['mols']

        gene = kwargs['gene']
        genecode = kwargs['dose']

        image_features = self.image_proj(self.encode_img(images))
        mol_features = self.mol_proj(self.encode_mol(mols))

        gene_features = self.gene_proj(self.encode_gene(gene, genecode))


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        mol_features = mol_features / mol_features.norm(dim=-1, keepdim=True)
        gene_features = gene_features / gene_features.norm(dim=-1, keepdim=True)


        return image_features, mol_features, gene_features, self.logit_scale_gi.exp(), self.logit_scale_gm.exp(), self.logit_scale_im.exp(), self.logit_scale_ig.exp(), self.logit_scale_mg.exp(), self.logit_scale_mi.exp()