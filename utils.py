from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, CenterCrop, RandomResizedCrop, InterpolationMode
from torchvision.transforms.functional import to_pil_image, to_tensor
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
import os
import wandb
import pandas as pd
def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, warmup: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    def lr_lambda(current_step):
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        progress = float(current_step - warmup) / float(max(1, num_training_steps - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)



def save_model(path, model):
    if not os.path.exists(path):
        os.makedirs(path)    
    torch.save(model.image_encoder.state_dict(), os.path.join(path,'img_encoder.pth'))
    torch.save(model.gene_encoder.state_dict(), os.path.join(path,'gene_encoder.pth'))
    torch.save(model.image_proj.state_dict(), os.path.join(path,'img_proj.pth'))
    torch.save(model.gene_proj.state_dict(), os.path.join(path,'gene_proj.pth'))
    torch.save(model.mol_encoder.state_dict(), os.path.join(path,'mol_encoder.pth'))
    torch.save(model.mol_proj.state_dict(), os.path.join(path,'mol_proj.pth'))
 
def save_model_gi(path, model):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.image_encoder.state_dict(), os.path.join(path,'img_encoder.pth'))
    torch.save(model.gene_encoder.state_dict(), os.path.join(path,'gene_encoder.pth'))
    torch.save(model.image_proj.state_dict(), os.path.join(path,'img_proj.pth'))
    torch.save(model.gene_proj.state_dict(), os.path.join(path,'gene_proj.pth'))
     

def save_model_gm(path, model):
    torch.save(model.mol_encoder.state_dict(), os.path.join(path,'mol_encoder_g.pth'))
    torch.save(model.gene_encoder.state_dict(), os.path.join(path,'gene_encoder.pth'))
    torch.save(model.mol_proj.state_dict(), os.path.join(path,'mol_proj_g.pth'))
    torch.save(model.gene_proj.state_dict(), os.path.join(path,'gene_proj.pth'))

def save_model_im(path, model):
    torch.save(model.mol_encoder.state_dict(), os.path.join(path,'mol_encoder_i.pth'))
    torch.save(model.image_encoder.state_dict(), os.path.join(path,'img_encoder.pth'))
    torch.save(model.mol_proj.state_dict(), os.path.join(path,'mol_proj_i.pth'))
    torch.save(model.image_proj.state_dict(), os.path.join(path,'img_proj.pth'))

def anchor_CL(anchor_rep, a_rep, b_rep, logit_scale):
    criterion = nn.CrossEntropyLoss()
    B = anchor_rep.size(0)  
    device = anchor_rep.device
    logits_a = torch.matmul(anchor_rep, a_rep.t()) * logit_scale
    logits_b = torch.matmul(anchor_rep, b_rep.t()) * logit_scale

    logits_combined = logits_a + logits_b

    labels = torch.arange(B).long().to(device)

    CL_loss = criterion(logits_combined, labels)
    return CL_loss

def get_metrics(modal_a_features, modal_b_features):
    metrics = {}
    logits_per_image = modal_a_features @ modal_b_features.t()
    logits_per_text = logits_per_image.t()

    logits = {"a2b": logits_per_image, "b2a": logits_per_text}
    ground_truth = (
        torch.arange(len(modal_b_features)).view(-1, 1).to(logits_per_image.device)
    )
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mrr"] = np.mean(1.0 / (preds + 1))  
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    return metrics



def balance_grad_weights(W_AB, W_AC, alpha = 0.2, use_balance = True):
    if not use_balance:
        return 1.0, 1.0
    rho_AB = W_AB / W_AC
    rho_AC = 1 / rho_AB
    weight_AB = 1 - torch.tanh(alpha * rho_AB) if rho_AB > 1 else 1
    weight_AC = 1 - torch.tanh(alpha * rho_AC) if rho_AC > 1 else 1
    if weight_AB < weight_AC:
        weight_AB = weight_AB / weight_AC
        weight_AC = 1.0
    else:
        weight_AC = weight_AC / weight_AB
        weight_AB = 1.0
    return weight_AB, weight_AC


def Balanced_CL(X, Y, Z, logit_scale_y, logit_scale_z, mask_y = None, mask_z = None, balance = True):
    criterion = nn.CrossEntropyLoss()  
    B = X.size(0)
    logits_y = torch.matmul(X, Y.T) * logit_scale_y
    if mask_y is not None:
        masked_indices = mask_y.bool()
        logits_y = logits_y[masked_indices][:, masked_indices]
        labels_y = torch.arange(logits_y.size(0), device=X.device)
    else:
        labels_y = torch.arange(B, device=X.device)
    w_y = logit_scale_y * F.softmax(logits_y, dim=1).mean()

    CL_loss_y = criterion(logits_y, labels_y).mean()

    logits_z = torch.matmul(X, Z.T) * logit_scale_y
    if mask_z is not None:
        # Apply the mask to filter out the samples where the mask is 0
        masked_indices = mask_z.bool()
        logits_z = logits_z[masked_indices][:, masked_indices]
        labels_z = torch.arange(logits_z.size(0), device=X.device)
    else:
        labels_z = torch.arange(B, device=X.device)
    w_z = logit_scale_z * F.softmax(logits_z, dim=1).mean()

    CL_loss_z = criterion(logits_z, labels_z).mean()

    with torch.no_grad():
        w_1, w_2 = balance_grad_weights(w_y.detach(), w_z.detach(), use_balance = balance)

    CL_loss = w_1 * CL_loss_y + w_2 * CL_loss_z
    return CL_loss

def do_CL(X, Y, logit_scale, mask=None):
    criterion = nn.CrossEntropyLoss()  
    B = X.size(0)
    logits = torch.matmul(X, Y.T) * logit_scale
    if mask is not None:
        masked_indices = mask.bool()
        logits = logits[masked_indices][:, masked_indices]
        labels = torch.arange(logits.size(0), device=logits.device)
    else:
        labels = torch.arange(B, device=logits.device)

    CL_loss = criterion(logits, labels).mean()

    return CL_loss

def get_transform(normalize:str = "dataset", is_train = True):
    if normalize == "img":
        normalize = NormalizeByImage()
    elif normalize == "dataset":
        normalize = Normalize(
    mean = [1.0473, 0.7050, 0.4546, 1.2903, 1.2335],
    std = [1.1714, 0.7722, 1.2241, 1.0318, 0.9567]
    )
    if is_train:
        resize = RandomResizedCrop([360, 480], scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC)
    else:

        resize = Compose([
                              CenterCrop([360, 480]),
                              ])
    if normalize:
        return Compose([
            ToTensor(),
            resize,
            normalize
        ])
    else:
        return Compose([
            ToTensor()
        ])

def train(epoch, current_step, dataloader, model, optimizer, scheduler, device, args, validloader=None):
    model.train()
    L = tqdm(dataloader)

    accum_loss = 0
    for step, batch in enumerate(L):

        batch = {k: v.to(device) for k, v in batch.items()}
        output =  model(**batch)
        if args.model == "clip*":
            images_repr, molecule_repr, scale = output
            loss_01, acc_01 = do_CL(images_repr, molecule_repr, scale)
            loss_02, acc_02 = do_CL(molecule_repr, images_repr, scale)
            loss = (loss_01 + loss_02) / 2

        elif args.model == "balance":
            use_b = True if current_step < 1500 else False
            gene_mask = batch['gene_mask']
            image_mask = batch['image_mask']
            combined_mask = gene_mask * image_mask

            images_repr, molecule_repr, gene_repr, logit_scale_gi, logit_scale_gm, logit_scale_im, logit_scale_ig, logit_scale_mg, logit_scale_mi = output
            loss_01 = Balanced_CL(molecule_repr, images_repr, gene_repr, logit_scale_mi, logit_scale_mg, mask_y = image_mask, mask_z = gene_mask, balance = use_b)
            loss_02 = Balanced_CL(images_repr, molecule_repr, gene_repr, logit_scale_im, logit_scale_ig, mask_y = image_mask, mask_z = combined_mask, balance = use_b)
            loss_03 = Balanced_CL(gene_repr, images_repr, molecule_repr, logit_scale_gi, logit_scale_gm, mask_y = combined_mask, mask_z = gene_mask, balance = use_b)

            loss = (loss_01 + loss_02 + loss_03 ) / 6

        else:
            images_repr, molecule_repr, gene_repr, logit_scale_i, logit_scale_g,  logit_scale_gi = output
            gene_mask = batch['gene_mask']
            image_mask = batch['image_mask']

            loss_01 = do_CL(images_repr, molecule_repr, logit_scale_i, mask=image_mask)
            loss_02 = do_CL(molecule_repr, images_repr, logit_scale_i, mask=image_mask)

            loss_03 = do_CL(gene_repr, molecule_repr, logit_scale_g, mask=gene_mask)
            loss_04 = do_CL(molecule_repr, gene_repr, logit_scale_g, mask=gene_mask)

            combined_mask = gene_mask * image_mask
            loss_05 = do_CL(gene_repr, images_repr, logit_scale_gi, mask=combined_mask)
            loss_06 = do_CL(images_repr, gene_repr, logit_scale_gi, mask=combined_mask)
            loss = (loss_01 + loss_02 + loss_03 + loss_04 + loss_05 + loss_06) / 6

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        current_step += 1
        accum_loss += loss.item()
        if args.debug:
            debugvalid(validloader, model,device, args)

    accum_loss /= len(L)
    print("Average CL Loss: {:.5f}".format(accum_loss))
    return

def debugvalid(dataloader, model, device, args):
    model.eval()
    L = tqdm(dataloader)  
    ms_im_all, ge_im_all, ms_ge_all = 0, 0, 0
    with torch.no_grad():  
        for step, batch in enumerate(L):
            batch = {k: v.to(device) for k, v in batch.items()}

            output =  model(**batch)
            if args.model == "clip*":
                images_repr, molecule_repr, scale = output

            elif args.model == "balance":
                use_b =  False
                gene_mask = batch['gene_mask']
                image_mask = batch['image_mask']
                combined_mask = gene_mask * image_mask
                images_repr, molecule_repr, gene_repr, logit_scale_gi, logit_scale_gm, logit_scale_im, logit_scale_ig, logit_scale_mg, logit_scale_mi = output
                ms_im = do_CL(images_repr, molecule_repr, logit_scale_im)+ do_CL(molecule_repr, images_repr, logit_scale_mi)
                ms_ge = do_CL(gene_repr, molecule_repr, logit_scale_gm)+ do_CL(molecule_repr, gene_repr, logit_scale_mg)
                ge_im = do_CL(images_repr, gene_repr, logit_scale_ig)+ do_CL(gene_repr, images_repr, logit_scale_gi)
            else:
                images_repr, molecule_repr, gene_repr, logit_scale_i, logit_scale_g,  logit_scale_gi = output
                ms_im = do_CL(images_repr, molecule_repr, logit_scale_i)+ do_CL(molecule_repr, images_repr, logit_scale_i)
                ms_ge = do_CL(gene_repr, molecule_repr, logit_scale_g)+ do_CL(molecule_repr, gene_repr, logit_scale_g)
                ge_im = do_CL(images_repr, gene_repr, logit_scale_gi)+ do_CL(gene_repr, images_repr, logit_scale_gi)
            ms_im_all += ms_im
            ge_im_all += ge_im
            ms_ge_all += ms_ge

        wandb.log({"MS-CP": ms_im_all/len(L), "CP-GE": ge_im_all/len(L), "GE-MS": ms_ge_all/len(L)})

def valid(epoch, optimal_mrr, dataloader, model, device, args):
    model.eval()
    L = tqdm(dataloader)    
    im_ms, ms_im, ge_ms, ms_ge, im_ge, ge_im = 0, 0, 0, 0, 0, 0
    accum_mrr = 0
    with torch.no_grad():  
        for step, batch in enumerate(L):
            batch = {k: v.to(device) for k, v in batch.items()}

            output =  model(**batch)
            if args.model == "clip*":
                images_repr, molecule_repr, scale = output
            elif args.model == "balance":
                use_b =  False
                gene_mask = batch['gene_mask']
                image_mask = batch['image_mask']
                combined_mask = gene_mask * image_mask

                images_repr, molecule_repr, gene_repr, logit_scale_gi, logit_scale_gm, logit_scale_im, logit_scale_ig, logit_scale_mg, logit_scale_mi = output

            else:
                images_repr, molecule_repr, gene_repr, logit_scale_i, logit_scale_g,  logit_scale_gi = output
            im = get_metrics(images_repr, molecule_repr)
            gm = get_metrics(gene_repr, molecule_repr)
            ig = get_metrics(images_repr, gene_repr)
            im_ms += im['a2b_mrr']
            ms_im += im['b2a_mrr'] 
            ge_ms += gm['a2b_mrr']  
            ms_ge += gm['b2a_mrr']  
            im_ge += ig['a2b_mrr']
            ge_im += ig['b2a_mrr']
            
    if len(L):
        accum_mrr  = im_ms + ms_im + ge_ms + ms_ge + im_ge + ge_im
        accum_mrr /= len(L)

    temp_mrr = accum_mrr
    if temp_mrr > optimal_mrr:
        optimal_mrr = temp_mrr
        save_model = True
    else:
        save_model = False

    print("Eval CL MRR CP2MS: {:.5f}, CP2GE: {:.5f}, GE2MS: {:.5f}".format(im_ms/len(L), im_ge/len(L), ge_ms/len(L)))
    return save_model, optimal_mrr



def test(dataloader, model, device, debug = False):
    model.eval()
    metrics = {}    #img, mol
    metrics2 = {}   #gene, mol
    metrics3 = {}   #img, gene

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(**batch)
            if len(output)  == 3:
                images_repr, molecule_repr, _ = output
                
                batch_metrics = get_metrics(images_repr.cpu(), molecule_repr.cpu())
                for key, value in batch_metrics.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            elif len(output)  == 5:

                images_repr, molecule_repr, gene_repr, _, _ = output


                batch_metrics = get_metrics(images_repr.cpu(), molecule_repr.cpu())
                for key, value in batch_metrics.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value) 

                batch_metrics = get_metrics(gene_repr.cpu(), molecule_repr.cpu())
                for key, value in batch_metrics.items():
                    if key not in metrics2:
                        metrics2[key] = []
                    metrics2[key].append(value)  

                batch_metrics = get_metrics(images_repr.cpu(), gene_repr.cpu())
                for key, value in batch_metrics.items():
                    if key not in metrics3:
                        metrics3[key] = []
                    metrics3[key].append(value)         

            elif len(output)  == 6:

                images_repr, molecule_repr, gene_repr, _, _, _ = output

                batch_metrics = get_metrics(images_repr.cpu(), molecule_repr.cpu())
                for key, value in batch_metrics.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value) 

                batch_metrics = get_metrics(gene_repr.cpu(), molecule_repr.cpu())
                for key, value in batch_metrics.items():
                    if key not in metrics2:
                        metrics2[key] = []
                    metrics2[key].append(value)  

                batch_metrics = get_metrics(images_repr.cpu(), gene_repr.cpu())
                for key, value in batch_metrics.items():
                    if key not in metrics3:
                        metrics3[key] = []
                    metrics3[key].append(value)         


            else:

                images_repr, molecule_repr, gene_repr, _, _, _ ,_,_,_= output

                batch_metrics = get_metrics(images_repr.cpu(), molecule_repr.cpu())
                for key, value in batch_metrics.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value) 

                batch_metrics = get_metrics(gene_repr.cpu(), molecule_repr.cpu())
                for key, value in batch_metrics.items():
                    if key not in metrics2:
                        metrics2[key] = []
                    metrics2[key].append(value)  

                batch_metrics = get_metrics(images_repr.cpu(), gene_repr.cpu())
                for key, value in batch_metrics.items():
                    if key not in metrics3:
                        metrics3[key] = []
                    metrics3[key].append(value) 

            if debug:
                torch.save(images_repr.cpu(), "./feas/images_repr.pt")
                torch.save(molecule_repr.cpu(), "./feas/molecule_repr.pt")
                torch.save(gene_repr.cpu(), "./feas/gene_repr.pt")
                return 
    for key, value in metrics.items():
        metrics[key] = np.mean(value)
    
    if metrics2:
        for key, value in metrics2.items():
            metrics2[key] = np.mean(value)
        for key, value in metrics3.items():
            metrics3[key] = np.mean(value)
        return metrics, metrics2, metrics3

    return metrics