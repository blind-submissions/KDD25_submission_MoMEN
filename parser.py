import argparse
import json

def parse_args(filepath = './configs.json'):
    parser = argparse.ArgumentParser()
    
    with open(filepath, 'r') as f:
        config = json.load(f)

    parser.add_argument("--seed", type=int, default=config.get('seed', 42), help="random seed, default is 42")
    parser.add_argument("--enable_wandb", type=str, default=config.get('wandb', 'False'))
    parser.add_argument("--notes", type=str, default=config.get('notes', '--'))
    parser.add_argument("--l1k_path", type=str, default=config.get('l1k_path', None))
    parser.add_argument("--img_root", type=str, default=config.get('img_root', None))
    parser.add_argument("--checkpoint_path", type=str, default=config.get('checkpoint_path', None))
 
    #--------pretrain GNN------------#      
    parser.add_argument("--pretrain_gnn_mode", type=str, default="GraphMVP_G", choices=["GraphMVP_G"])
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')
    parser.add_argument('--fixed_GNN', type=int, default=config.get('fixed_GNN', 0))


    #--------Gene Expression Encoder-----------#
    parser.add_argument("--gene_hidden_dim", type=int, default=config.get('gene_hidden_dim', 512))
    parser.add_argument("--gene_num_layers", type=int, default=config.get('gene_num_layers', 3))
    parser.add_argument("--gene_dropout", type=float, default=config.get('gene_dropout', 0.1))
    parser.add_argument("--gene_method", type=str, default=config.get('gene_method', 'con'))

 
    #--------Model---------#
    parser.add_argument("--model", type=str, default=config.get('model', 'clip'))
    parser.add_argument("--T", type=float, default=config.get('temperature', 0.07))
    parser.add_argument("--gene_output_dim", type=int, default=config.get('gene_output_dim', 512))
    parser.add_argument("--mol_output_dim", type=int, default=config.get('mol_output_dim', 512))
    parser.add_argument("--img_output_dim", type=int, default=config.get('img_output_dim', 512))
    parser.add_argument("--clip_hidden_dim", type=int, default=config.get('clip_hidden_dim', 256))
    parser.add_argument("--load_pretrained", type=int, default=config.get('load_pretrained', 1))


    #--------Training--------#
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=config.get('epochs', 30))
    parser.add_argument("--batch_size", type=int, default=config.get('batch_size', 240))
    parser.add_argument("--im_encoder_lr", type=float, default=config.get('im_encoder_lr', 5e-5))
    parser.add_argument("--ge_encoder_lr", type=float, default=config.get('ge_encoder_lr', 1e-4))
    parser.add_argument("--ms_encoder_lr", type=float, default=config.get('ms_encoder_lr', 5e-5))
    parser.add_argument("--projector_lr", type=float, default=config.get('projector_lr', 1e-4))
    parser.add_argument("--encoder_wd", type=float, default=config.get('encoder_wd', 0.01))
    parser.add_argument("--projector_wd", type=float, default=config.get('projector_wd', 0))
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    return args