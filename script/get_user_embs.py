import sys, os
sys.path.append("../src")
from user_emb import *

import json
import yaml
import random
import argparse
import numpy as np

import torch
from accelerate.utils import set_seed
from transformers import AutoModel, AutoTokenizer

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def set_seeds(seed):
    set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to the config file')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    args = parser.parse_args()
    
    assert os.path.isfile(args.config), f"Invalid config path: {args.config}"
    config = load_config(args.config)
    config['seed'] = args.seed
    set_seeds(config['seed'])
    
    embs_model = AutoModel.from_pretrained(config['model_name'])
    embs_tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    if config['data_type'] == "LaMP2":
        lamp2_make_item_embeddings(
            model = embs_model,
            tokenizer = embs_tokenizer,
            batch_size = config['batch_size'],
            tmp_dataset = json.load(open(config['train']['data_path'],"r")), 
            output_path = config['train']['save_path'],
            is_test=False
            )
        lamp2_make_item_embeddings(
            model = embs_model,
            tokenizer = embs_tokenizer,
            batch_size = config['batch_size'],
            tmp_dataset = json.load(open(config['test']['data_path'],"r")), 
            output_path = config['test']['save_path'],
            is_test=True
            )
        
    elif config['data_type'] == "LaMP3":
        lamp3_make_item_embeddings(
            model = embs_model,
            tokenizer = embs_tokenizer,
            batch_size = config['batch_size'],
            tmp_dataset = json.load(open(config['train']['data_path'],"r")), 
            output_path = config['train']['save_path'],
            is_test=False
            )
        lamp3_make_item_embeddings(
            model = embs_model,
            tokenizer = embs_tokenizer,
            batch_size = config['batch_size'],
            tmp_dataset = json.load(open(config['test']['data_path'],"r")), 
            output_path = config['test']['save_path'],
            is_test=True
            )    
    
    elif config['data_type'] == "LaMP4":
        lamp4_make_item_embeddings(
            model = embs_model,
            tokenizer = embs_tokenizer,
            batch_size = config['batch_size'],
            tmp_dataset = json.load(open(config['train']['data_path'],"r")), 
            output_path = config['train']['save_path'],
            is_test=False
            )
        lamp4_make_item_embeddings(
            model = embs_model,
            tokenizer = embs_tokenizer,
            batch_size = config['batch_size'],
            tmp_dataset = json.load(open(config['test']['data_path'],"r")), 
            output_path = config['test']['save_path'],
            is_test=True
            )    

    elif config['data_type'] == "LaMP5":
        lamp5_make_item_embeddings(
            model = embs_model,
            tokenizer = embs_tokenizer,
            batch_size = config['batch_size'],
            tmp_dataset = json.load(open(config['train']['data_path'],"r")), 
            output_path = config['train']['save_path'],
            is_test=False
            )
        lamp5_make_item_embeddings(
            model = embs_model,
            tokenizer = embs_tokenizer,
            batch_size = config['batch_size'],
            tmp_dataset = json.load(open(config['test']['data_path'],"r")), 
            output_path = config['test']['save_path'],
            is_test=True
            )   
    else:
        raise ValueError(f"Invalid data_type: {config['data_type']}")