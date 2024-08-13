# KDD25_submission_MoMEN
Source code for Multi-modal Learning for Phenotypic Drug Discovery Using Cell Morphological Image and Gene Expression Data.


##  Environments
MoMEN requires anaconda with python 3.10 or later, cudatoolkit=11.3 and below packages
```
ogb==1.3.5 
torch-geometric==2.3.1   
rdkit>=2023.3.3        
pytorch>=1.12.1+cu113   
torchvision>=0.13.1+cu113
PyTDC>=0.4.1
```

## Data
For dataset download, please follow [here](https://www.nature.com/articles/s41592-022-01667-0)

For the pretraining data, please check 
scripts and instructions in [dataprocess](https://github.com/blind-submissions/KDD25_submission_MoMEN/tree/main/dataprocess)

## Pre-training
```
python pretrain.py
```
check configs.json for parameter setting.


## Downstream
Raw data of Pharmacokinetics can be found in [here](https://tdcommons.ai/single_pred_tasks/adme/)

Raw data of clinical trial outcome prediction can be found in [here](https://github.com/futianfan/clinical-trial-outcome-prediction)

Check the scripts and insturcutions in [downstream](https://github.com/blind-submissions/KDD25_submission_MoMEN/tree/main/downstream)

