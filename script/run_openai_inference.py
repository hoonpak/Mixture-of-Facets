import sys, os
sys.path.append("../src")
from openai_inference import BatchInference
from data import *

import yaml
import random
import argparse
import numpy as np
from transformers import AutoTokenizer

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
    parser.add_argument('-k', '--key_path', type=str, help='Path to the key file')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    args = parser.parse_args()
    
    assert os.path.isfile(args.config), f"Invalid config path: {args.config}"
    config = load_config(args.config)
    config['seed'] = args.seed
    set_seeds(config['seed'])
    
    train_batch_instance = BatchInference(
        batch_input_file_path=config['inference']['train']['input_file_path'],
        batch_output_file_path=config['inference']['train']['output_file_path'],
        key_path=args.key_path,
        config=config
        )
    fit_batch_instance = BatchInference(
        batch_input_file_path=config['inference']['fit']['input_file_path'],
        batch_output_file_path=config['inference']['fit']['output_file_path'],
        key_path=args.key_path,
        config=config
        )
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    train_data = load_json_data(config['inference']['train']['data_path'])
    fit_data = load_json_data(config['inference']['fit']['data_path'])
    
    if config['inference']['type'] == 'reranker':
        prompt_function = get_prompt_function_for_reranker(config['data_type'])
        train_iter_data = [
            {
                'input': prompt_function(
                    inp = train_batch_instance.question_template.format(text = sample["origin_query"]),
                    profile = [{'augment_query': sample['augment_query'], 'augment_label': sample['augment_label']}],
                    max_length = 768,
                    tokenizer = tokenizer
                    ) ,
                'id': sample['id']
            } 
            for sample in train_data
            ]
        fit_iter_data = [
            {
                'input': prompt_function(
                    inp = fit_batch_instance.question_template.format(text = sample["origin_query"]),
                    profile = [{'augment_query': sample['augment_query'], 'augment_label': sample['augment_label']}],
                    max_length = 768,
                    tokenizer = tokenizer
                    ) ,
                'id': sample['id']
            } 
            for sample in fit_data
            ]
        
        train_batch_instance.make_batch_zeroshot_input_data(train_iter_data)
        fit_batch_instance.make_batch_zeroshot_input_data(fit_iter_data)
        train_batch_instance.call_batch_inference()
        fit_batch_instance.call_batch_inference()
        train_batch_instance.wait_until_done()
        fit_batch_instance.wait_until_done()
        train_batch_instance.save_batch_output_file()
        fit_batch_instance.save_batch_output_file()      
        
    elif config['inference']['type'] == 'adapter':
        prompt_function = get_prompt_function_for_adapter(config['data_type'])
        
        eval_batch_instance = BatchInference(
            batch_input_file_path=config['inference']['eval']['input_file_path'],
            batch_output_file_path=config['inference']['eval']['output_file_path'],
            key_path=args.key_path,
            config=config
            )
        eval_data = load_json_data(config['inference']['eval']['data_path'])

        train_iter_data = [
            {
                'input': prompt_function(
                    inp = train_batch_instance.question_template.format(text = sample["query"]),
                    profile = sample['profile'],
                    max_length = 768,
                    tokenizer = tokenizer
                    ) ,
                'id': sample['id']
            } 
            for sample in train_data
            ]
        fit_iter_data = [
            {
                'input': prompt_function(
                    inp = fit_batch_instance.question_template.format(text = sample["query"]),
                    profile = sample['profile'],
                    max_length = 768,
                    tokenizer = tokenizer
                    ) ,
                'id': sample['id']
            } 
            for sample in fit_data
            ]
        eval_iter_data = [
            {
                'input': prompt_function(
                    inp = eval_batch_instance.question_template.format(text = sample["query"]),
                    profile = sample['profile'],
                    max_length = 768,
                    tokenizer = tokenizer
                    ) ,
                'id': sample['id']
            } 
            for sample in eval_data
            ]

        train_batch_instance.make_batch_adapter_input_data(train_iter_data)
        fit_batch_instance.make_batch_adapter_input_data(fit_iter_data)
        eval_batch_instance.make_batch_adapter_input_data(eval_iter_data)
        train_batch_instance.call_batch_inference()
        fit_batch_instance.call_batch_inference()
        eval_batch_instance.call_batch_inference()
        train_batch_instance.wait_until_done()
        fit_batch_instance.wait_until_done()
        eval_batch_instance.wait_until_done()
        train_batch_instance.save_batch_output_file()
        fit_batch_instance.save_batch_output_file()
        eval_batch_instance.save_batch_output_file()