import sys, os
sys.path.append("../src")
from openai_inference import BatchInference
from data import *

import yaml
import random
import argparse
import numpy as np

import evaluate
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
    parser.add_argument('-t', '--type', type=str, help='Type of the mof; train or fit')
    parser.add_argument('-v', '--version', type=str, help='version of the mof')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    args = parser.parse_args()
    
    assert os.path.isfile(args.config), f"Invalid config path: {args.config}"
    config = load_config(args.config)
    config['seed'] = args.seed
    set_seeds(config['seed'])

    if config['inference']['type'] == 'reranker':
        
        eval_batch_instance = BatchInference(
            batch_input_file_path=config['inference']['eval']['input_file_path']+args.type+"_v"+args.version+".jsonl",
            batch_output_file_path=config['inference']['eval']['output_file_path']+args.type+"_v"+args.version+".jsonl",
            key_path=args.key_path,
            config=config
            )
        
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

        prompt_function = get_prompt_function_for_reranker(config['data_type'])
        eval_data = load_json_data(config['inference']['eval']['data_path']+args.type+"_v"+args.version+".json")
        
        eval_iter_data = [
            {
                'input': prompt_function(
                    inp = eval_batch_instance.question_template.format(text = sample["query"]),
                    profile = sample['profile'][:config['inference']['eval']['num_items']],
                    max_length = 768,
                    tokenizer = tokenizer
                    ) ,
                'id': sample['id']
            } 
            for sample in eval_data
            ]
        
        eval_batch_instance.mask_batch_zeroshot_input_data(eval_iter_data)
        eval_batch_instance.call_batch_inference()
        eval_batch_instance.wait_until_done()
        eval_batch_instance.save_batch_output_file()
        
        predicts = eval_batch_instance.parse_output()
        labels = [sample['query_label'] for sample in eval_data]

    elif config['inference']['type'] == 'adapter':
        eval_data = json.load(open(config["eval"]["save_path"]+args.type+f"_v{args.version}"+".json", "r"))
        predicts = [sample['profile'][0]['response'] for sample in eval_data]
        labels = [sample['query_label'] for sample in eval_data]
    
    if config['data_type'] == "LaMP2":
        all_labels = ['sci-fi', 'based on a book', 'comedy', 'action', 'twist ending', 'dystopia', 'dark comedy', 'classic', 'psychology', 'fantasy', 'romance', 'thought-provoking', 'social commentary', 'violence', 'true story']
        
        label_dict = {k:v for v,k in enumerate(all_labels)}
        label_dict['unknown'] = len(label_dict)
        preds = []
        labs = []
        
        predicts = [postprocess_output(pred) for pred in predicts]
        
        for pred, lab in zip(predicts, labels):
            if pred.lower() in label_dict:
                preds.append(label_dict[pred])
            else:
                preds.append(label_dict['unknown'])
            labs.append(label_dict[lab])
        
        f1_metric = evaluate.load("f1")
        accuracy_metric = evaluate.load("accuracy")
        
        result_acc = accuracy_metric.compute(predictions=preds, references=labs)
        result_f1 = f1_metric.compute(predictions=preds, references=labs, labels=list(range(len(all_labels))), average = "macro")
        result = {"accuracy" : result_acc["accuracy"], "f1" : result_f1["f1"]}
        print(result)
        result['compares'] = [(predict, label) for predict, label in zip(predicts, labels)]
        json.dump(
            result,
            open(config['eval']['save_path']+args.type+"_v"+args.version+"_result"+".json", "w"),
            ensure_ascii=False, indent=4
        )
        
    elif config['data_type'] == "LaMP3":
        all_labels = ["1","2","3","4","5"]
        mse_metric = evaluate.load("mse")
        mae_metric = evaluate.load("mae")
        
        def create_mapping(x, y):
            try:
                return float(x)
            except:
                print(x)
                y = float(y)
                if abs(1 - y) > abs(5 - y):
                    return 1.0
                else:
                    return 5.0
                
        predicts = [postprocess_output(pred) for pred in predicts]

        predicts = [create_mapping(x,y) for x,y in zip(predicts, labels)]
        labels = [create_mapping(x,x) for x in labels]
        result_mae = mae_metric.compute(predictions=predicts, references=labels)
        result_rmse = mse_metric.compute(predictions=predicts, references=labels, squared = False)
        result = {"mae" : result_mae["mae"], "rmse" : result_rmse["mse"]}
        print(result)
        
        result['compares'] = [(predict, label) for predict, label in zip(predicts, labels)]
        json.dump(
            result,
            open(config['eval']['save_path']+args.type+"_v"+args.version+"_result"+".json", "w"),
            ensure_ascii=False, indent=4
        )

    elif config['data_type'] in ["LaMP4", "LaMP5"]:
        def postprocess_text(preds, labels):
            preds = [pred.strip().replace('"', "") for pred in preds]
            labels = [[label.strip().replace('"', "")] for label in labels]

            return preds, labels
        
        bleu_metric = evaluate.load("sacrebleu")
        rouge_metric = evaluate.load('rouge')
        meteor_metric = evaluate.load('meteor')
        
        decoded_preds, decoded_labels = postprocess_text(predicts, labels)
        result_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"rouge-1" : result_rouge["rouge1"], "rouge-2" : result_rouge["rouge2"], "rouge-L" : result_rouge["rougeL"], "bleu" : result_bleu["score"], "rouge-LSum" : result_rouge["rougeLsum"], "meteor" : result_meteor['meteor']}
        print(result)
        
        result['compares'] = [(predict, label) for predict, label in zip(predicts, labels)]
        json.dump(
            result,
            open(config['eval']['save_path']+args.type+"_v"+args.version+"_result"+".json", "w"),
            ensure_ascii=False, indent=4
        )

    else:
        raise ValueError(f"Invalid data_type: {config['data_type']}")