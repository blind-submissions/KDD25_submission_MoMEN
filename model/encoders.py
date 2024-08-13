import torch
import torch.nn as nn
import torch.nn.functional as F
from cloome.model import *
from mol_gnn import *

class GeneEncoder(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden_dim=512, output_dim=300, dropout_rate=0.1, combine_method='con'):
        super(GeneEncoder, self).__init__()
        self.combine_method = combine_method
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.dose_encoder = nn.Linear(1, hidden_dim)
        mlp_input_dim = 2 * hidden_dim  if combine_method == 'con' else hidden_dim
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(mlp_input_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout_rate)

    def from_pretrained(self, pretrained_path):
        pretrained_dict = torch.load(pretrained_path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'mu_layer' not in k and 'logvar_layer' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print("load pretrained gene encoder")

    def forward(self, gene_expression_data, dose):
        x = self.initial_layer(gene_expression_data)
        x = F.relu(x)
        dose = dose.view(-1,1)
        code_embed = F.relu(self.dose_encoder(dose))
        
        if self.combine_method == 'con':
            x = torch.cat((x, code_embed), dim=1)
        else:
            x = x + code_embed
            
        x = self.dropout(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return x


class ImgEncoder(nn.Module):
    def __init__(self, img_output_dim, load_pretrained = True):
        super(ImgEncoder, self).__init__()
        with open('../pretrained/ResNet50/RN50.json', 'r') as f:
            model_info = json.load(f)
        self.img_encoder = ResNet(
            layers=model_info['vision_layers'],
            output_dim=model_info['embed_dim'],
            input_shape=model_info['input_channels']
        )
        if load_pretrained:
            self.img_encoder.load_state_dict(torch.load('../pretrained/ResNet50/resnet.pth'))
        
        in_features = self.img_encoder.fc.in_features
        self.img_encoder.fc = nn.Linear(in_features, img_output_dim)

    def forward(self, img):
        return self.img_encoder(img)


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
        pretrained_model_path = '../pretrained/GNN/model.pth'
        self.molecule_model.from_pretrained(pretrained_model_path)
        self.projector = nn.Linear(300, output_dim)

    def forward(self, *argv):
        output, _ = self.molecule_model(*argv)
        if not self.fix_gnn:
            output = self.projector(output)
        return output


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn, use_batch_norm):
        super(Expert, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.linear(x)
        if self.use_batch_norm:
            x = self.norm(x)
        if self.activation_fn == 'relu':
            x = F.relu(x)
        elif self.activation_fn == 'softplus':
            x = F.softplus(x)
        elif self.activation_fn == 'gelu':
            x = F.gelu(x)
        x = self.dropout(x)
        return x


class MoE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MoE, self).__init__()
        expert1 = Expert(input_dim, output_dim, activation_fn='relu', use_batch_norm=False)
        expert2 = Expert(input_dim, output_dim, activation_fn='relu', use_batch_norm=True)
        expert3 = Expert(input_dim, output_dim, activation_fn='softplus', use_batch_norm=True)
        expert4 = Expert(input_dim, output_dim, activation_fn='gelu', use_batch_norm=True)
        experts = [expert1, expert2, expert3, expert4]

        self.experts = nn.ModuleList(experts)
        self.gating_network = nn.Linear(input_dim, len(experts))
        self.combined_linear = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        gates = F.softmax(self.gating_network(x), dim=1)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gated_outputs = torch.bmm(gates.unsqueeze(1), expert_outputs).squeeze(1)
        output = self.combined_linear(gated_outputs)
        return output


