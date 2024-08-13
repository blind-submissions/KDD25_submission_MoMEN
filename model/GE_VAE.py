import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class GeneExpressionDataset(Dataset):
    def __init__(self, dataframe):
        self.gene_expression_data = np.vstack(dataframe['gene_expression_data'].values).astype(np.float32)
        self.pert_dose = dataframe['pert_dose'].values.astype(np.float32)
    
    def __len__(self):
        return len(self.gene_expression_data)
    
    def __getitem__(self, idx):
        gene_expression = self.gene_expression_data[idx]
        dose = self.pert_dose[idx]
        return torch.from_numpy(gene_expression), torch.tensor([dose], dtype=torch.float32)


class GeneEncoder(nn.Module):
    def __init__(self, input_dim = 977, num_layers=2, hidden_dim=512, output_dim=300, dropout_rate=0.1, combine_method='con'):
        super(GeneEncoder, self).__init__()
        self.combine_method = combine_method
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.dose_encoder = nn.Linear(1, hidden_dim)
        mlp_input_dim = 2 * hidden_dim if combine_method == 'con' else hidden_dim
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(mlp_input_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.mu_layer = nn.Linear(hidden_dim, output_dim)
        self.logvar_layer = nn.Linear(hidden_dim, output_dim)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, gene_expression_data, dose):
        x = self.initial_layer(gene_expression_data)
        x = F.relu(x)
        dose = dose.view(-1, 1)
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
        
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

class GeneDecoder(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden_dim=512, output_dim=977, dropout_rate=0.1):
        super(GeneDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z):
        x = z
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

class VAE(nn.Module):
    def __init__(self, gene_input_dim, hidden_dim=512, z_dim=300, num_layers=2, dropout_rate=0.1, combine_method='con'):
        super(VAE, self).__init__()
        self.encoder = GeneEncoder(gene_input_dim, num_layers, hidden_dim, z_dim, dropout_rate, combine_method)
        self.decoder = GeneDecoder(z_dim, num_layers, hidden_dim, gene_input_dim, dropout_rate)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, gene_expression_data, dose):
        mu, logvar = self.encoder(gene_expression_data, dose)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')  
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train_and_validate(model, optimizer, train_loader, val_loader, epochs=50):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for gene_expression_data, dose in train_loader:
            gene_expression_data, dose = gene_expression_data.to(device), dose.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(gene_expression_data, dose)
            loss = loss_function(recon_batch, gene_expression_data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for gene_expression_data, dose in val_loader:
                gene_expression_data, dose = gene_expression_data.to(device), dose.to(device)
                recon_batch, mu, logvar = model(gene_expression_data, dose)
                loss = loss_function(recon_batch, gene_expression_data, mu, logvar)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch: {epoch}, Average Training Loss: {avg_train_loss}, Average Validation Loss: {avg_val_loss}")


gene_df = pd.read_pickle('../data/l1k.pkl')
train_df, val_df = train_test_split(gene_df, test_size=0.2, random_state=42)

train_dataset = GeneExpressionDataset(train_df)
val_dataset = GeneExpressionDataset(val_df)

batch_size = 2048
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0") 
else:
    device = torch.device("cpu")

gene_input_dim = 977 
z_dim = 300
vae = VAE(gene_input_dim, z_dim=z_dim)
vae = vae.to(device)

optimizer = Adam(vae.parameters(), lr=1e-3, weight_decay = 1e-3)
train_and_validate(vae, optimizer, train_dataloader, val_dataloader, epochs = 120)
torch.save(vae.encoder.state_dict(), '../pretrained/encoder_state_dict.pth')
