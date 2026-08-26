import sys, os
sys.path.append("../src")
from clustering import *

import json
import yaml
import random
import argparse
import numpy as np

import torch
from accelerate.utils import set_seed

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
    
    if config['data_type'] == "LaMP2":
        lamp2_get_cluster_based_reranker_augment_train_dataset(
            user_embs_path = config['train']['user_embs_path'], 
            cluster_num = config['cluster_num'], 
            random_query_num = config['random_query_num'],
            currnet_dataset = json.load(open(config['train']['data_path'],"r")),
            save_path = config['train']['save_path'],
            )
        lamp2_get_cluster_based_reranker_augment_fit_eval_dataset(
            user_embs_path = config['test']['user_embs_path'], 
            cluster_num = config['cluster_num'], 
            random_query_num = config['random_query_num'],
            currnet_dataset = json.load(open(config['test']['data_path'],"r")),
            save_path = config['test']['save_path'],
            )
        
    elif config['data_type'] == "LaMP3":
        lamp3_get_cluster_based_reranker_augment_train_dataset(
            user_embs_path = config['train']['user_embs_path'], 
            cluster_num = config['cluster_num'], 
            random_query_num = config['random_query_num'],
            currnet_dataset = json.load(open(config['train']['data_path'],"r")),
            save_path = config['train']['save_path'],
            )
        lamp3_get_cluster_based_reranker_augment_fit_eval_dataset(
            user_embs_path = config['test']['user_embs_path'], 
            cluster_num = config['cluster_num'], 
            random_query_num = config['random_query_num'],
            currnet_dataset = json.load(open(config['test']['data_path'],"r")),
            save_path = config['test']['save_path'],
            )
    
    elif config['data_type'] == "LaMP4":
        lamp4_get_cluster_based_reranker_augment_train_dataset(
            user_embs_path = config['train']['user_embs_path'], 
            cluster_num = config['cluster_num'], 
            random_query_num = config['random_query_num'],
            currnet_dataset = json.load(open(config['train']['data_path'],"r")),
            save_path = config['train']['save_path'],
            )
        lamp4_get_cluster_based_reranker_augment_fit_eval_dataset(
            user_embs_path = config['test']['user_embs_path'], 
            cluster_num = config['cluster_num'], 
            random_query_num = config['random_query_num'],
            currnet_dataset = json.load(open(config['test']['data_path'],"r")),
            save_path = config['test']['save_path'],
            )
    
    elif config['data_type'] == "LaMP5":
        lamp5_get_cluster_based_reranker_augment_train_dataset(
            user_embs_path = config['train']['user_embs_path'], 
            cluster_num = config['cluster_num'], 
            random_query_num = config['random_query_num'],
            currnet_dataset = json.load(open(config['train']['data_path'],"r")),
            save_path = config['train']['save_path'],
            )
        lamp5_get_cluster_based_reranker_augment_fit_eval_dataset(
            user_embs_path = config['test']['user_embs_path'], 
            cluster_num = config['cluster_num'], 
            random_query_num = config['random_query_num'],
            currnet_dataset = json.load(open(config['test']['data_path'],"r")),
            save_path = config['test']['save_path'],
            )
    
    else:
        raise ValueError(f"Invalid data_type: {config['data_type']}")