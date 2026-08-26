import os, sys
sys.path.append("../src")

import warnings
warnings.simplefilter('ignore')
import argparse
import yaml

from train import RerankerTrainer, AdapterTrainer

import torch
import random
import numpy as np
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
    parser.add_argument('-v', '--version', type=str, help='Version of this try')
    parser.add_argument('--debug', default='train', type=str, help='debug')
    args = parser.parse_args()

    assert os.path.isfile(args.config), f"Invalid config path: {args.config}"
    config = load_config(args.config)
    config['version'] = args.version
    config['type'] = args.debug
    set_seeds(config['seed'])
    
    if config['inference']['type'] == "reranker":
        if args.debug == "train":
            rerankertrainer = RerankerTrainer(config)
            rerankertrainer.train()
        elif args.debug == "train_eval":
            rerankertrainer = RerankerTrainer(config)
            rerankertrainer.eval(is_fit=False)
        elif args.debug == "fit":
            rerankertrainer = RerankerTrainer(config)
            rerankertrainer.fit()
        elif args.debug == "fit_eval":
            rerankertrainer = RerankerTrainer(config)
            rerankertrainer.eval(is_fit=True)
    elif config['inference']['type'] == "adapter":
        if args.debug == "train":
            adaptertrainer = AdapterTrainer(config)
            adaptertrainer.train()
        elif args.debug == "train_eval":
            adaptertrainer = AdapterTrainer(config)
            adaptertrainer.eval(is_fit=False)
        elif args.debug == "fit":
            adaptertrainer = AdapterTrainer(config)
            adaptertrainer.fit()
        elif args.debug == "fit_eval":
            adaptertrainer = AdapterTrainer(config)
            adaptertrainer.eval(is_fit=True)
    