import sys, os
sys.path.append("../src")
from clustering import *
from model import *

import json
import yaml
import random
import argparse
import numpy as np
from collections import defaultdict

import torch
from transformers import AutoTokenizer
from datasets import Dataset
from accelerate.utils import set_seed

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_user_emb_dict(user_emb_ls, user_embs_dict, cut_his_len):
    for sample in user_emb_ls:
        tmp_embs = sample["user_emb_tensor"].flip(0)[:cut_his_len]
        pad_size = cut_his_len - tmp_embs.shape[0]
        if pad_size != 0:
            tmp_user_mask = [False]*tmp_embs.shape[0] + [True]*pad_size
            padding = torch.zeros(pad_size, tmp_embs.shape[-1])
            tmp_embs = torch.cat([tmp_embs, padding], dim=0)
        else:
            tmp_user_mask = [False]*tmp_embs.shape[0]
        user_embs_dict[sample["id"]] = (tmp_embs.tolist(), tmp_user_mask)

def get_reranker_score(reranker_model, tokenizer, cur_dataset, user_emb_ls, batch_size, cut_his_len):
    
    def data_formatting(tokenizer, origin, augment):
        data_template = "[CLS] {origin} [SEP] {augment} [SEP]"
        tokenized_origin = tokenizer(origin, add_special_tokens=False, truncation=False)
        tokenized_augment = tokenizer(augment, add_special_tokens=False, truncation=False)
        origin_length = len(tokenized_origin.input_ids)
        augment_length = len(tokenized_augment.input_ids)
        per_max_length = (tokenizer.model_max_length - 3) // 2

        if origin_length + augment_length + 3 > tokenizer.model_max_length:
            origin_save_tokens = 0
            augment_save_tokens = 0

            if (origin_length < per_max_length) and (augment_length > per_max_length):
                origin_save_tokens = per_max_length - origin_length
                saved_origin = tokenized_origin.input_ids
                saved_augment = tokenized_augment.input_ids[:(per_max_length+origin_save_tokens)]

            elif (origin_length > per_max_length) and (augment_length < per_max_length):
                augment_save_tokens = per_max_length - augment_length
                saved_origin = tokenized_origin.input_ids[:(per_max_length+augment_save_tokens)]
                saved_augment = tokenized_augment.input_ids
            
            elif (origin_length >= per_max_length) and (augment_length >= per_max_length):
                saved_origin = tokenized_origin.input_ids[:per_max_length]
                saved_augment = tokenized_augment.input_ids[:per_max_length]

            else:
                saved_origin = tokenized_origin.input_ids
                saved_augment = tokenized_augment.input_ids
            
            total_text = [tokenizer.cls_token_id] + saved_origin + [tokenizer.sep_token_id] + saved_augment + [tokenizer.sep_token_id]
            
            return tokenizer.decode(total_text, skip_special_tokens=False)
        else:
            return data_template.format(origin=origin, augment=augment)
        
    reranker_model.to("cuda:0").eval()
    reranker_model.config.pad_token_id = tokenizer.pad_token_id
    reranker_model.config.eos_token_id = tokenizer.eos_token_id
    
    user_embs_dict = {}
    get_user_emb_dict(user_emb_ls, user_embs_dict, cut_his_len)
    
    solution_scores_dict = defaultdict(
        lambda: defaultdict(
            lambda: {"query_label": None, "profile": []}
        )
    )
    
    total_len = len(cur_dataset)

    for batch_start in tqdm(
        range(0, total_len, batch_size), 
        total=(total_len + batch_size - 1) // batch_size,
        dynamic_ncols=True, ncols=None, leave=True
        ):
        
        batch_items = cur_dataset[batch_start: batch_start + batch_size]

        batch_user_ids = []
        batch_origins = []
        batch_origin_labels = []
        batch_augments = []
        batch_augment_labels = []
        batch_texts = []
        batch_user_embs = []
        batch_user_masks = []
        
        for item in batch_items:
            origin = item["origin_query"]
            augment = item["augment_query"]

            assert origin != augment, 'origin_query and augment_query are equal'

            user_id = item["id"].split("_")[0]
            user_emb, user_mask = user_embs_dict[user_id]

            batch_user_ids.append(user_id)
            batch_origins.append(origin)
            batch_origin_labels.append(item["origin_label"])
            batch_augments.append(augment)
            batch_augment_labels.append(item["augment_label"])
            batch_texts.append(data_formatting(tokenizer=tokenizer, origin=origin, augment=augment))
            batch_user_embs.append(user_emb)
            batch_user_masks.append(user_mask)        
            
        input_batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt"
            )
        
        input_batch["user_embs"] = torch.as_tensor(batch_user_embs, dtype=torch.float32)
        input_batch["user_masks"] = torch.as_tensor(batch_user_masks, dtype=torch.bool)
        input_batch = {k: v.to("cuda:0") for k, v in input_batch.items()}

        with torch.inference_mode():
            outputs = reranker_model(**input_batch)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].detach().cpu().tolist()

        for user_id, origin, origin_label, augment, augment_label, score in zip(
            batch_user_ids,
            batch_origins,
            batch_origin_labels,
            batch_augments,
            batch_augment_labels,
            probs
        ):            
            if solution_scores_dict[user_id][origin]["query_label"] is None:
                solution_scores_dict[user_id][origin]["query_label"] = origin_label
                
            solution_scores_dict[user_id][origin]["profile"].append(
                {
                    "augment_query": augment,
                    "augment_label": augment_label,
                    "score": score
                }
            )

    sorted_ranked_data = []
    for user_id, origin_queries in solution_scores_dict.items():
        for idx, (origin_query, val) in enumerate(origin_queries.items()):
            unique_data = list({(d["augment_query"], d["augment_label"], d["score"]): d for d in val["profile"]}.values())
            sorted_ranked_data.append(
                {
                    "id": f"{user_id}_{idx}",
                    "query": origin_query,
                    "query_label": val["query_label"],
                    "profile": sorted(unique_data, key=lambda x: x["score"], reverse=True)[:4]
                }
            )

    return sorted_ranked_data

def set_seeds(seed):
    set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rc", "--reranker_config", type=str)
    parser.add_argument("-ac", "--adapter_config", type=str)
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--version", default=0, type=int, help="seed")
    args = parser.parse_args()
    
    reranker_config = load_config(args.reranker_config)
    adapter_config = load_config(args.adapter_config)
    adapter_config["seed"] = args.seed
    reranker_config["version"] = args.version
    adapter_config["version"] = args.version
    set_seeds(adapter_config["seed"])
    
    save_path = "/".join(adapter_config["inference"]["eval"]["data_path"].split("/")[:-1] + ["before"])
    print(save_path)
    os.makedirs(save_path, exist_ok=True) 

    if adapter_config["data_type"] == "LaMP2":
        lamp2_get_cluster_based_reranker_augment_train_dataset(
            user_embs_path = adapter_config["trainer"]["user_embs_path"], 
            cluster_num = adapter_config["cluster_num"], 
            random_query_num = adapter_config["random_query_num"],
            currnet_dataset = json.load(open(adapter_config["origin"]["train"]["data_path"],"r")),
            save_path = save_path,
            )
        lamp2_get_cluster_based_reranker_augment_fit_eval_dataset(
            user_embs_path = adapter_config["fit"]["user_embs_path"], 
            cluster_num = adapter_config["cluster_num"], 
            random_query_num = adapter_config["random_query_num"],
            currnet_dataset = json.load(open(adapter_config["origin"]["test"]["data_path"],"r")),
            save_path = save_path
            )
        
    elif adapter_config["data_type"] == "LaMP3":
        lamp3_get_cluster_based_reranker_augment_train_dataset(
            user_embs_path = adapter_config["trainer"]["user_embs_path"], 
            cluster_num = adapter_config["cluster_num"], 
            random_query_num = adapter_config["random_query_num"],
            currnet_dataset = json.load(open(adapter_config["origin"]["train"]["data_path"],"r")),
            save_path = save_path,
            )
        lamp3_get_cluster_based_reranker_augment_fit_eval_dataset(
            user_embs_path = adapter_config["fit"]["user_embs_path"], 
            cluster_num = adapter_config["cluster_num"], 
            random_query_num = adapter_config["random_query_num"],
            currnet_dataset = json.load(open(adapter_config["origin"]["test"]["data_path"],"r")),
            save_path = save_path
            )
        
    elif adapter_config["data_type"] == "LaMP4":
        lamp4_get_cluster_based_reranker_augment_train_dataset(
            user_embs_path = adapter_config["trainer"]["user_embs_path"], 
            cluster_num = adapter_config["cluster_num"], 
            random_query_num = adapter_config["random_query_num"],
            currnet_dataset = json.load(open(adapter_config["origin"]["train"]["data_path"],"r")),
            save_path = save_path,
            )
        lamp4_get_cluster_based_reranker_augment_fit_eval_dataset(
            user_embs_path = adapter_config["fit"]["user_embs_path"], 
            cluster_num = adapter_config["cluster_num"], 
            random_query_num = adapter_config["random_query_num"],
            currnet_dataset = json.load(open(adapter_config["origin"]["test"]["data_path"],"r")),
            save_path = save_path
            )

    elif adapter_config["data_type"] == "LaMP5":
        lamp5_get_cluster_based_reranker_augment_train_dataset(
            user_embs_path = adapter_config["trainer"]["user_embs_path"], 
            cluster_num = adapter_config["cluster_num"], 
            random_query_num = adapter_config["random_query_num"],
            currnet_dataset = json.load(open(adapter_config["origin"]["train"]["data_path"],"r")),
            save_path = save_path,
            )
        lamp5_get_cluster_based_reranker_augment_fit_eval_dataset(
            user_embs_path = adapter_config["fit"]["user_embs_path"], 
            cluster_num = adapter_config["cluster_num"], 
            random_query_num = adapter_config["random_query_num"],
            currnet_dataset = json.load(open(adapter_config["origin"]["test"]["data_path"],"r")),
            save_path = save_path
            )
    else:
        raise ValueError(f"Invalid data_type: {adapter_config["data_type"]}")
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_config["model_name"])
    
    train_reranker_model = FacetAdapter.from_pretrained(reranker_config["trainer"]["output_dir"]+str(reranker_config["version"]))
    fit_reranker_model = FacetAdapter.from_pretrained(reranker_config["fit"]["output_dir"]+str(reranker_config["version"]))

    before_train_data = load_json_data(save_path+"/train.json")
    before_fit_history_data = load_json_data(save_path+"/fit_history.json")
    before_eval_data = load_json_data(save_path+"/eval.json")
    
    train_user_emb_ls = make_user_item_list_dict(torch.load(adapter_config["trainer"]["user_embs_path"]))
    fit_eval_user_emb_ls = make_user_item_list_dict(torch.load(adapter_config["fit"]["user_embs_path"]))
    
    print("START TRAIN DATA SCORING")
    train_sorted_ranked_data = get_reranker_score(reranker_model=train_reranker_model, tokenizer=tokenizer, cur_dataset=before_train_data, user_emb_ls=train_user_emb_ls, batch_size=128, cut_his_len=reranker_config["cut_his_len"])
    print("START FIT HISTORY DATA SCORING")
    fit_sorted_ranked_data = get_reranker_score(reranker_model=fit_reranker_model, tokenizer=tokenizer, cur_dataset=before_fit_history_data, user_emb_ls=fit_eval_user_emb_ls, batch_size=128, cut_his_len=reranker_config["cut_his_len"])
    print("START EVAL DATA SCORING")
    eval_sorted_ranked_data = get_reranker_score(reranker_model=fit_reranker_model, tokenizer=tokenizer, cur_dataset=before_eval_data, user_emb_ls=fit_eval_user_emb_ls, batch_size=128, cut_his_len=reranker_config["cut_his_len"])
    
    json.dump(
        train_sorted_ranked_data, 
        open(adapter_config["inference"]["train"]["data_path"], "w"),
        ensure_ascii=False, 
        indent=4
        )
    json.dump(
        fit_sorted_ranked_data, 
        open(adapter_config["inference"]["fit"]["data_path"], "w"),
        ensure_ascii=False, 
        indent=4
        )
    json.dump(
        eval_sorted_ranked_data, 
        open(adapter_config["inference"]["eval"]["data_path"], "w"),
        ensure_ascii=False, 
        indent=4
        )