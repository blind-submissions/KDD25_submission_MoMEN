# Dataset Preprocessing

## Raw Data

For cell painting data, please check [here](https://www.nature.com/articles/nprot.2016.105)

For l1000 data, please check [here](https://www.nature.com/articles/s41592-022-01667-0)

## Preprocessing

After downloading the raw data, write their paths into data_config.yaml

Then please use the following commands:

```
python process_cp_image.py
python process_cp_data.py
python process_l1000.py
python split.py
python process_graph.py
```

If successful, you will find the following files in the data directory.

- l1k.pkl
- geometric_data.pt
- l1k_smiles.csv
- cp_smiles_paths.csv
- gene_smiles
- retrieval


Note that we converted all SMILES strings into canonical SMILES