from MBData import *

import os
import sys
# sys.path.append(os.path.abspath(os.path.join( '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import argparse

from torch.utils.data import DataLoader
from mol_gnn import *
from torch.nn import BCEWithLogitsLoss
from tqdm import *
from sklearn.metrics import auc, roc_curve, precision_recall_curve, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from tdc.single_pred import ADME, Tox




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='2c19')
parser.add_argument('--model', type=str, default='momen')
parser.add_argument('--task', type=str, default='c', choices = ['c','r'], help = 'classification or regression')

args = parser.parse_args()

print(args)

if args.dataset == '2c19':
    data = ADME(name = 'CYP2C19_Veith')
elif args.dataset == '2d6':
    data = ADME(name = 'CYP2D6_Veith')
elif args.dataset == '3a4':
    data = ADME(name = 'CYP3A4_Veith')
elif args.dataset == '1a2':
    data = ADME(name = 'CYP1A2_Veith')
elif args.dataset == '2c9':
    data = ADME(name = 'CYP2C9_Veith')

elif args.dataset == 'ppbr':
    data = ADME(name = 'PPBR_AZ')
    args.task ='r'
elif args.dataset == 'vdss':
    data = ADME(name = 'VDss_Lombardo')
    args.task ='r'
elif args.dataset == '1um':
    data = Tox(name = 'herg_central', label_name = 'hERG_at_1uM')
    args.task ='r'
elif args.dataset == 'lipo':
    data = ADME(name = 'Lipophilicity_AstraZeneca')
    args.task ='r'


if args.model == 'momen':
    pretrained_model_path = '../checkpoints/momen.pth'
elif args.model == 'IG':
    pass
#... other baselines


split = data.get_split('scaffold')
if args.task == 'r':
    split, mean, std = normalize_y(split)


trainset  = MetabolismDataset(name = args.dataset, df = split['train'], stage = 'train')
validset  = MetabolismDataset(name = args.dataset, df = split['valid'], stage = 'valid')
testset  = MetabolismDataset(name = args.dataset, df = split['test'], stage = 'test')

trainloader = DataLoader(trainset, batch_size = 512, shuffle = True,  num_workers = 0, collate_fn = collate_fn)
validloader = DataLoader(validset, batch_size = 1024, shuffle = True,  num_workers = 0, collate_fn = collate_fn)
testloader = DataLoader(testset, batch_size = 1024, shuffle = True,  num_workers = 0, collate_fn = collate_fn)

class MolEncoder(nn.Module):
    def __init__(self, num_layer, emb_dim, JK, drop_ratio, gnn_type, graph_pooling, fix_gnn, output_dim):
        super(MolEncoder, self).__init__()
        self.molecule_node_model = GNN(
            num_layer=num_layer, emb_dim=emb_dim,
            JK=JK, drop_ratio=drop_ratio,
            gnn_type=gnn_type)
        self.molecule_model = GNN_graphpred(
            num_layer=num_layer, emb_dim=emb_dim, JK=JK, graph_pooling=graph_pooling,
            num_tasks=1, molecule_node_model=self.molecule_node_model)
        self.fix_gnn = fix_gnn
        self.projector = nn.Linear(300, output_dim)

    def forward(self, *argv):
        output, _ = self.molecule_model(*argv)
        output = self.projector(output)
        return output

def train(epoch, model, dataloader, optimizer, criterion, device, task):
    model.train()
    total_loss = 0
    for batch in dataloader:
    
        graph = batch['graph'].to(device)
        if task == 'c':
            labels = batch['label'].unsqueeze(1).to(device)

            outputs = model(graph)
        else:
            labels = batch['label'].to(device)
            outputs = model(graph).squeeze()

        optimizer.zero_grad()

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss/len(dataloader)

def evaluate_model(data_loader, criterion, model, device, task = 'c'):
    model.eval()  
    all_labels = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():  
        for batch in data_loader:
            graph = batch['graph'].to(device)
            if task == 'c':
                labels = batch['label'].unsqueeze(1).to(device)

                outputs = model(graph)
            else:
                labels = batch['label'].to(device)
                outputs = model(graph).squeeze()

            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)  
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    if task == 'c':
        fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        precision, recall, thresholds_pr = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall, precision)

        return avg_loss, roc_auc, pr_auc
    else:
        rmse = np.sqrt(mean_squared_error(all_labels, all_probs))
        r2 = r2_score(all_labels, all_probs)
        return avg_loss, rmse, r2


criterion = BCEWithLogitsLoss() if args.task == 'c' else torch.nn.MSELoss()
device = torch.device("cuda")
roc_ls = []
for _ in range(10):
    model = MolEncoder(num_layer=5, emb_dim=300, JK='last', drop_ratio=0.5, gnn_type='gin', graph_pooling='mean', fix_gnn=0, output_dim=300)

    model_dict = torch.load(pretrained_model_path)

    model_dict = {k: v for k, v in model_dict.items() if k != 'projector.weight' and k != 'projector.bias'}
    model.load_state_dict(model_dict, strict=False)
    model.projector = nn.Linear(300, 1)  
    tmp = init.kaiming_normal_(model.projector.weight, mode='fan_out', nonlinearity='relu')
    for name, param in model.named_parameters():
        if 'projector' not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=1e-2, weight_decay=0)
    model = model.to(device)
    best_loss = 10000
    for epoch in tqdm(range(100)):
        trainloss = train(epoch, model, trainloader, optimizer, criterion, device, args.task)
        valid_loss, res1, res2 = evaluate_model(validloader, criterion, model,device, args.task)
        # print("epoch: {}, loss: {}, eval1: {}".format(epoch,valid_loss, res2))
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), './{}_model.pth'.format(args.dataset))
    model.load_state_dict(torch.load('./{}_model.pth'.format(args.dataset)))

    _, roc_auc, pr_auc = evaluate_model(testloader, criterion, model,device, args.task)
    print(roc_auc, pr_auc)
    if roc_auc>0.50:
        roc_ls.append(roc_auc)


mean = np.mean(roc_ls)
std = np.std(roc_ls)

print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")
