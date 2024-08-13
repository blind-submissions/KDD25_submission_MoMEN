import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from datasets import * 
from torch.utils.data import DataLoader
from tqdm import *
from utils import *
from parser import *
import pandas as pd
from model.clips import *
from model.encoders import *
import wandb

args = parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device)) 
else:
    device = torch.device("cpu")

transform_train = get_transform(is_train = True)
transform_val = get_transform(is_train = False)

gene_df = pd.read_pickle('./data/l1k.pkl')
img_df = pd.read_csv('./data/cp_smiles_paths.csv')
train_sm = './data/retrieval/train_overlap.csv'
train_sm_c = './data/retrieval/train.csv'
valid_sm = './data/retrieval/valid.csv'
test_sm = './data/retrieval/test.csv'
graph_path = './data/geometric_data.pt'
smile_path = './data/union.csv'


gene_encoder = GeneEncoder(977, num_layers = args.gene_num_layers, hidden_dim = args.gene_hidden_dim, output_dim = args.gene_output_dim, dropout_rate = args.gene_dropout, combine_method = args.gene_method)
mol_encoder = MolEncoder(args.num_layer, args.gnn_emb_dim, args.JK, args.dropout_ratio, args.gnn_type, args.graph_pooling, args.fixed_GNN, output_dim = args.mol_output_dim)
img_encoder = ImgEncoder(args.img_output_dim)
gene_encoder.from_pretrained('./pretrained/encoder_state_dict.pth')


print(args.model)

wandb.init(
    project="momen",
    config={
    "model": args.model
    }
)


if args.model=="clip":
    trainset = TripLetMaskDataset(gene_df, img_df, args.img_root, transform_train, train_sm_c, graph_path, smile_path, is_train = True)
    validset = TripLetMaskDataset(gene_df, img_df, args.img_root, transform_val, valid_sm, graph_path, smile_path, is_train = False)
    testset = TripLetMaskDataset(gene_df, img_df, args.img_root, transform_val, test_sm, graph_path, smile_path, is_train = False)
    model = TCLIP(args.clip_hidden_dim, mol_encoder, gene_encoder, img_encoder, args.mol_output_dim, args.gene_output_dim, args.img_output_dim)
    trainloader = DataLoader(trainset, batch_size = 230, shuffle = True, num_workers = 16, collate_fn = collate_triplet_ms_fn, drop_last=False)
    validloader = DataLoader(validset, batch_size = 100, shuffle = False, num_workers = 16, collate_fn = collate_triplet_ms_fn, drop_last=True)
    testloader = DataLoader(testset, batch_size = 100, shuffle = False, num_workers = 16, collate_fn = collate_triplet_ms_fn, drop_last=True)


elif args.model=="momen":
    trainset = TripLetMaskDataset(gene_df, img_df, args.img_root, transform_train, train_sm_c, graph_path, smile_path, is_train = True)
    validset = TripLetMaskDataset(gene_df, img_df, args.img_root, transform_val, valid_sm, graph_path, smile_path, is_train = False)
    testset = TripLetMaskDataset(gene_df, img_df, args.img_root, transform_val, test_sm, graph_path, smile_path, is_train = False)
    model = TCLIPMoED(args.clip_hidden_dim, mol_encoder, gene_encoder, img_encoder, args.mol_output_dim, args.gene_output_dim, args.img_output_dim)
    trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True, num_workers = 16, collate_fn = collate_triplet_ms_fn, drop_last=False)
    validloader = DataLoader(validset, batch_size = 100, shuffle = False, num_workers = 16, collate_fn = collate_triplet_ms_fn, drop_last=True)
    testloader = DataLoader(testset, batch_size = 100, shuffle = False, num_workers = 16, collate_fn = collate_triplet_ms_fn, drop_last=True)


print(len(testset))

checkpoint_path  = './checkpoints/' + args.model 
model = model.to(device)

model_param_group = [
    {"params": model.image_encoder.parameters(), "lr": args.im_encoder_lr, "weight_decay": args.encoder_wd},
    {"params": model.gene_encoder.parameters(), "lr": args.ge_encoder_lr, "weight_decay": args.encoder_wd},
    {"params": model.mol_encoder.parameters(), "lr": args.ms_encoder_lr, "weight_decay": args.encoder_wd},
    {"params": model.image_proj.parameters(), "lr": args.projector_lr, "weight_decay": args.projector_wd },
    {"params": model.gene_proj.parameters(), "lr": args.projector_lr, "weight_decay": args.projector_wd },
    {"params": model.mol_proj.parameters(), "lr": args.projector_lr, "weight_decay": args.projector_wd }
]

optimizer = optim.AdamW(model_param_group)
num_training_steps = len(trainloader) * args.epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05*num_training_steps), num_training_steps)

current_step  = 0
optimal_acc = 0


for e in range(1, args.epochs):
    train(e, current_step, trainloader, model, optimizer,scheduler, device, args, validloader)
    current_step += len(trainloader)
    save, optimal_acc =  valid(e, optimal_acc, validloader, model, device, args)
    if save:
        save_model(checkpoint_path, model)


print('test ', args.model)

model.mol_encoder.load_state_dict(torch.load(os.path.join(checkpoint_path,'mol_encoder.pth')))
model.gene_encoder.load_state_dict(torch.load(os.path.join(checkpoint_path,'gene_encoder.pth')))
model.mol_proj.load_state_dict(torch.load(os.path.join(checkpoint_path,'mol_proj.pth')))
model.gene_proj.load_state_dict(torch.load(os.path.join(checkpoint_path,'gene_proj.pth')))
model.image_encoder.load_state_dict(torch.load(os.path.join(checkpoint_path,'img_encoder.pth')))
model.image_proj.load_state_dict(torch.load(os.path.join(checkpoint_path,'img_proj.pth')))

output = test(testloader, model, device, debug = args.debug)
if len(output)>1:
    for metrics in output:
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")
            
wandb.finish()
