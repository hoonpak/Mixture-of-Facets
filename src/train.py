import os
import math
import torch
import evaluate
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer

from model import FacetAdapter
from data import *
from user_emb import *

class CustomDataCollator:
    def __init__(self, tokenizer, user_embs_dict):
        self.tokenizer = tokenizer
        self.user_embs_dict = user_embs_dict

    def __call__(self, features):
        user_ids = [f["user_id"] for f in features]

        token_features = []
        for f in features:
            token_features.append({
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
                "labels": f["labels"],
            })

        batch = self.tokenizer.pad(
            token_features,
            padding=True,
            return_tensors="pt",
        )

        batch["user_embs"] = torch.tensor(
            [self.user_embs_dict[user_id][0] for user_id in user_ids],
            dtype=torch.float32,
        )
        batch["user_masks"] = torch.tensor(
            [self.user_embs_dict[user_id][1] for user_id in user_ids],
            dtype=torch.bool,
        )

        return batch

class RerankerTrainer():
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if config["data_type"] == "LaMP2":
            self.answer_candidate = ["sci-fi", "based on a book", "comedy", "action", "twist ending", "dystopia", "dark comedy", "classic", "psychology", "fantasy", "romance", "thought-provoking", "social commentary", "violence", "true story"]
        elif config["data_type"] == "LaMP3":
            self.answer_candidate = ["1", "2", "3", "4", "5"]
            
        self.train_dataset = None
        self.fit_dataset = None
    
    def get_user_emb_dict(self, user_emb_ls, user_embs_dict):
        for sample in user_emb_ls:
            tmp_embs = sample["user_emb_tensor"].flip(0)[:self.config["cut_his_len"]]
            pad_size = self.config["cut_his_len"] - tmp_embs.shape[0]
            if pad_size != 0:
                tmp_user_mask = [False]*tmp_embs.shape[0] + [True]*pad_size
                padding = torch.zeros(pad_size, tmp_embs.shape[-1])
                tmp_embs = torch.cat([tmp_embs, padding], dim=0)
            else:
                tmp_user_mask = [False]*tmp_embs.shape[0]
            user_embs_dict[sample["id"]] = (tmp_embs.tolist(), tmp_user_mask)
    
    def data_formatting(self, origin, augment):
        data_template = "[CLS] {origin} [SEP] {augment} [SEP]"
        tokenized_origin = self.tokenizer(origin, add_special_tokens=False, truncation=False)
        tokenized_augment = self.tokenizer(augment, add_special_tokens=False, truncation=False)
        origin_length = len(tokenized_origin.input_ids)
        augment_length = len(tokenized_augment.input_ids)
        per_max_length = (self.tokenizer.model_max_length - 3) // 2

        if origin_length + augment_length + 3 > self.tokenizer.model_max_length:
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
            
            total_text = [self.tokenizer.cls_token_id] + saved_origin + [self.tokenizer.sep_token_id] + saved_augment + [self.tokenizer.sep_token_id]
            
            return self.tokenizer.decode(total_text, skip_special_tokens=False)
        else:
            return data_template.format(origin=origin, augment=augment)

    def make_training_dataset(self, reranker_data, bbllms_outputs):
        rows = []

        if self.config["data_type"] in ["LaMP4", "LaMP5"]:
            rouge_metric = evaluate.load("rouge")
            user_buffer = []
            score_list = []
            current_user_id = None

        for r, b in tqdm(zip(reranker_data, bbllms_outputs), total=len(reranker_data), 
                         dynamic_ncols=True, ncols=None, leave=True):
            assert r["id"] == b["id"]
            if r["origin_query"] == r["augment_query"]:
                continue

            response = b["response"]
            user_id = b["id"].split("_")[0]

            if self.config["data_type"] in ["LaMP2", "LaMP3"]:
                if postprocess_output(response) not in self.answer_candidate:
                    continue
                labels = int(r["origin_label"] == postprocess_output(response))
                rows.append({
                    "user_id": user_id,
                    "text": self.data_formatting(
                        origin=r["origin_query"],
                        augment=r["augment_query"]
                    ),
                    "labels": labels,
                })

            elif self.config["data_type"] in ["LaMP4", "LaMP5"]:
                prediction = [b["response"].strip().replace("\"", "")]
                answer = [r["origin_label"].strip().replace("\"", "")]
                result_rouge = rouge_metric.compute(
                    predictions=prediction,
                    references=answer
                )
                score = (result_rouge["rouge1"] + result_rouge["rougeL"]) / 2

                if current_user_id is None:
                    current_user_id = user_id

                if user_id != current_user_id:
                    avg_score = sum(score_list) / len(score_list)
                    for buffered_row, buffered_score in zip(user_buffer, score_list):
                        buffered_row["labels"] = int(buffered_score > avg_score)
                        rows.append(buffered_row)

                    user_buffer = []
                    score_list = []
                    current_user_id = user_id

                user_buffer.append({
                    "user_id": user_id,
                    "text": self.data_formatting(
                        origin=r["origin_query"],
                        augment=r["augment_query"]
                    ),
                })
                score_list.append(score)

        if self.config["data_type"] in ["LaMP4", "LaMP5"] and user_buffer:
            avg_score = sum(score_list) / len(score_list)

            for buffered_row, buffered_score in zip(user_buffer, score_list):
                buffered_row["labels"] = int(buffered_score > avg_score)
                rows.append(buffered_row)

        print(f"DATA BALANCE: {sum(x['labels'] for x in rows)}/{len(rows)}")

        return Dataset.from_list(rows).shuffle(seed=self.config["seed"])
    
    def prepare_training_data(self, reranker_data_path, bbllms_outputs_path, user_embs_path):
        reranker_data = load_json_data(reranker_data_path)
        bbllms_outputs = load_open_ai_api_response(bbllms_outputs_path)
        user_emb_ls = make_user_item_list_dict(torch.load(user_embs_path))

        self.user_embs_dict = {}
        self.get_user_emb_dict(user_emb_ls, self.user_embs_dict)
        
        return self.make_training_dataset(reranker_data, bbllms_outputs)
    
    def preprocess_function(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            add_special_tokens=False,
        )
    
    def train(self):
        if self.train_dataset is None:
            self.train_dataset = self.prepare_training_data(self.config["inference"]["train"]["data_path"], self.config["inference"]["train"]["output_file_path"], self.config["trainer"]["user_embs_path"])
            self.train_dataset = self.train_dataset.map(self.preprocess_function, batched=True, remove_columns=["text"])
        else:
            user_emb_ls = make_user_item_list_dict(torch.load(self.config["trainer"]["user_embs_path"]))
            self.user_embs_dict = {}
            self.get_user_emb_dict(user_emb_ls, self.user_embs_dict)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        ckpt_pth = self.config["trainer"]["output_dir"] + self.config["version"]

        data_collator = CustomDataCollator(tokenizer=self.tokenizer,user_embs_dict=self.user_embs_dict)
        
        training_args = TrainingArguments(
            output_dir=ckpt_pth,
            learning_rate=self.config["trainer"]["learning_rate"],
            per_device_train_batch_size=self.config["trainer"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["trainer"]["gradient_accumulation_steps"],
            num_train_epochs=self.config["trainer"]["num_train_epochs"],
            weight_decay=self.config["trainer"]["weight_decay"],
            logging_steps=self.config["trainer"]["logging_steps"],
            logging_strategy=self.config["trainer"]["logging_strategy"],
            save_strategy=self.config["trainer"]["save_strategy"],
            save_total_limit=1,
            push_to_hub=self.config["trainer"]["push_to_hub"],
            dataloader_num_workers=1,
            remove_unused_columns=False
        )
        
        model_config = AutoConfig.from_pretrained(self.config["model_name"])
        model_config.cut_his_len = self.config["cut_his_len"]
        model_config.num_labels = 2
        model_config.num_facets = self.config["num_facets"]
        model_config.mof_top_k = self.config["mof_top_k"]

        base_model = AutoModel.from_pretrained(self.config["model_name"], config=model_config)
        self.model = FacetAdapter(model_config)
        self.model.bert.load_state_dict(base_model.state_dict(), strict=False)
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id 

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        if not os.path.exists(ckpt_pth):
            os.makedirs(ckpt_pth)
        trainer.model.save_pretrained(ckpt_pth)
        trainer.save_state()

    def fit(self):
        if self.fit_dataset is None:
            self.fit_dataset = self.prepare_training_data(self.config["inference"]["fit"]["data_path"], self.config["inference"]["fit"]["output_file_path"], self.config["fit"]["user_embs_path"])
            self.fit_dataset = self.fit_dataset.map(self.preprocess_function, batched=True, remove_columns=["text"])
        else:
            user_emb_ls = make_user_item_list_dict(torch.load(self.config["fit"]["user_embs_path"]))
            self.user_embs_dict = {}
            self.get_user_emb_dict(user_emb_ls, self.user_embs_dict)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        ckpt_pth = self.config["fit"]["output_dir"] + self.config["version"]

        data_collator = CustomDataCollator(tokenizer=self.tokenizer, user_embs_dict=self.user_embs_dict)
        training_args = TrainingArguments(
            output_dir=ckpt_pth,
            learning_rate=self.config["fit"]["learning_rate"],
            per_device_train_batch_size=self.config["fit"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["fit"]["gradient_accumulation_steps"],
            num_train_epochs=self.config["fit"]["num_train_epochs"],
            weight_decay=self.config["fit"]["weight_decay"],
            logging_steps=self.config["fit"]["logging_steps"],
            logging_strategy=self.config["fit"]["logging_strategy"],
            save_strategy=self.config["fit"]["save_strategy"],
            save_total_limit=1,
            push_to_hub=self.config["fit"]["push_to_hub"],
            dataloader_num_workers=1,
            remove_unused_columns=False
        )
        
        if self.model is None:
            self.model = FacetAdapter.from_pretrained(self.config["trainer"]["output_dir"] + self.config["version"])
        for param in self.model.bert.parameters():
            param.requires_grad = False
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id 

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.fit_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        if not os.path.exists(ckpt_pth):
            os.makedirs(ckpt_pth)
        trainer.model.save_pretrained(ckpt_pth)
        trainer.save_state()

    def eval(self, is_fit):
        eval_data = load_json_data(self.config["eval"]["data_path"])
        user_emb_ls = make_user_item_list_dict(torch.load(self.config["eval"]["user_embs_path"]))
        if is_fit:
            model = FacetAdapter.from_pretrained(self.config["fit"]["output_dir"]+self.config["version"])
        else:
            model = FacetAdapter.from_pretrained(self.config["trainer"]["output_dir"]+self.config["version"])
        model.to(self.device).eval()
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.config.eos_token_id = self.tokenizer.eos_token_id 
        
        user_ids = []
        user_embs = []
        user_masks = []
        origins = []
        origin_labels = []
        augments = []
        augment_labels = []
        samples = []
        
        user_embs_dict = {}
        self.get_user_emb_dict(user_emb_ls, user_embs_dict)
            
        for idx in range(len(eval_data)):
            assert eval_data[idx]["origin_query"] != eval_data[idx]["augment_query"] 
            user_id = eval_data[idx]["id"].split("_")[0]
            user_ids.append(user_id)
            user_embs.append(user_embs_dict[user_id][0])
            user_masks.append(user_embs_dict[user_id][1])
            origins.append(eval_data[idx]["origin_query"])
            origin_labels.append(eval_data[idx]["origin_label"].strip().replace("\"", ""))
            augments.append(eval_data[idx]["augment_query"])
            augment_labels.append(eval_data[idx]["augment_label"].strip().replace("\"", ""))
            samples.append(self.data_formatting(origin=eval_data[idx]["origin_query"], augment=eval_data[idx]["augment_query"]))
        
        eval_dict = {
            "user_ids": user_ids,
            "user_embs": user_embs,
            "user_masks": user_masks,
            "origins": origins,
            "origin_labels": origin_labels,
            "augments": augments,
            "augment_labels": augment_labels,
            "text": samples,
        }

        solution_scores_dict = {}
        batch_size = self.config["eval"]["batch_size"]
        num_samples = len(eval_dict["user_ids"])

        for batch_start in tqdm(
            range(0, num_samples, batch_size),
            total=math.ceil(num_samples / batch_size),
            dynamic_ncols=True, ncols=None, leave=True
        ):
            batch = {
                k: v[batch_start:batch_start + batch_size]
                for k, v in eval_dict.items()
            }

            input_batch = self.tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_batch["user_embs"] = torch.FloatTensor(batch["user_embs"])
            input_batch["user_masks"] = torch.BoolTensor(batch["user_masks"])
            input_batch = {k: v.to(self.device) for k, v in input_batch.items()}

            with torch.inference_mode():
                outputs = model(**input_batch)

            probs = torch.softmax(outputs.logits.detach().cpu(), dim=-1)
            scores = probs[:, 1].tolist()
            router_weight = outputs.router_weights.detach().cpu().numpy().tolist()
            topk_indices = outputs.topk_indices.detach().cpu().numpy().tolist()

            for i in range(len(batch["user_ids"])):
                user_id = batch["user_ids"][i]
                origin = batch["origins"][i]
                origin_label = batch["origin_labels"][i]
                augment = batch["augments"][i]
                augment_label = batch["augment_labels"][i]
                score = scores[i]
                rw = router_weight[i]
                ti = topk_indices[i]

                if user_id not in solution_scores_dict:
                    solution_scores_dict[user_id] = {"origin": origin, "origin_label": origin_label, "profile": []}
                solution_scores_dict[user_id]["profile"].append(
                        {
                            "augment_query": augment,
                            "augment_label": augment_label,
                            "score":score,
                            "rw": rw,
                            "ti": ti
                        }
                    )
                
        sorted_ranked_data = []
        for k, v in solution_scores_dict.items():
            sorted_ranked_data.append(
                    {
                        "id": k,
                        "query": v["origin"],
                        "query_label": v["origin_label"],
                        "profile": sorted(v["profile"], key=lambda x:x["score"], reverse=True)
                    }
                )
        
        json.dump(
            sorted_ranked_data, 
            open(self.config["inference"]["eval"]["data_path"]+self.config["type"]+f"_v{self.config["version"]}"+".json", "w"),
            ensure_ascii=False, 
            indent=4
            )

class AdapterTrainer():
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if config["data_type"] == "LaMP2":
            self.answer_candidate = ["sci-fi", "based on a book", "comedy", "action", "twist ending", "dystopia", "dark comedy", "classic", "psychology", "fantasy", "romance", "thought-provoking", "social commentary", "violence", "true story"]
        elif config["data_type"] == "LaMP3":
            self.answer_candidate = ["1","2","3","4","5"]
        self.train_dataset = None
        self.fit_dataset = None
        
    def get_user_emb_dict(self, user_emb_ls, user_embs_dict):
        for sample in user_emb_ls:
            tmp_embs = sample["user_emb_tensor"].flip(0)[:self.config["cut_his_len"]]
            pad_size = self.config["cut_his_len"] - tmp_embs.shape[0]
            if pad_size != 0:
                tmp_user_mask = [False]*tmp_embs.shape[0] + [True]*pad_size
                padding = torch.zeros(pad_size, tmp_embs.shape[-1])
                tmp_embs = torch.cat([tmp_embs, padding], dim=0)
            else:
                tmp_user_mask = [False]*tmp_embs.shape[0]
            user_embs_dict[sample["id"]] = (tmp_embs.tolist(), tmp_user_mask)
            
    def data_formatting(self, query, response):
        data_template = "[CLS] {query} [SEP] {response} [SEP]"
        tokenized_query = self.tokenizer(query, add_special_tokens=False, truncation=False)
        tokenized_response = self.tokenizer(response, add_special_tokens=False, truncation=False)
        query_length = len(tokenized_query.input_ids)
        response_length = len(tokenized_response.input_ids)
        per_max_length = (self.tokenizer.model_max_length - 3) // 2

        if query_length + response_length + 3 > self.tokenizer.model_max_length:
            origin_save_tokens = 0
            augment_save_tokens = 0

            if (query_length < per_max_length) and (response_length > per_max_length):
                origin_save_tokens = per_max_length - query_length
                saved_query = tokenized_query.input_ids
                saved_response = tokenized_response.input_ids[:(per_max_length+origin_save_tokens)]

            elif (query_length > per_max_length) and (response_length < per_max_length):
                augment_save_tokens = per_max_length - response_length
                saved_query = tokenized_query.input_ids[:(per_max_length+augment_save_tokens)]
                saved_response = tokenized_response.input_ids
            
            elif (query_length >= per_max_length) and (response_length >= per_max_length):
                saved_query = tokenized_query.input_ids[:per_max_length]
                saved_response = tokenized_response.input_ids[:per_max_length]

            else:
                saved_query = tokenized_query.input_ids
                saved_response = tokenized_response.input_ids

            total_text = [self.tokenizer.cls_token_id] + saved_query + [self.tokenizer.sep_token_id] + saved_response + [self.tokenizer.sep_token_id]

            return self.tokenizer.decode(total_text, skip_special_tokens=False)
        else:
            return data_template.format(query=query, response=response)

    def make_training_dataset(self, adapter_data, bbllms_outputs):
        rows = []

        for adapter_item, output_item in tqdm(zip(adapter_data, bbllms_outputs), total=len(adapter_data), 
                                              dynamic_ncols=True, ncols=None, leave=True):
            assert adapter_item["id"] == output_item["id"]

            user_id = output_item["id"].split("_")[0]
            query = adapter_item["query"]
            query_label = adapter_item["query_label"]

            if self.config['data_type'] in ['LaMP2', 'LaMP3']:
                for response in output_item["responses"]:
                    if postprocess_output(response) not in self.answer_candidate:
                        continue
                    label = int(query_label == postprocess_output(response))
                    rows.append({
                        "user_id": user_id,
                        "text": self.data_formatting(query=query, response=postprocess_output(response)),
                        "labels": label,
                    })
                
            elif self.config['data_type'] in ['LaMP4', 'LaMP5']:
                for response in output_item["responses"]:
                    label = 0
                    rows.append({
                        "user_id": user_id,
                        "text": self.data_formatting(query=query, response=response.strip().replace("\"", "")),
                        "labels": label,
                    })
                
                label = 1
                rows.append({
                    "user_id": user_id,
                    "text": self.data_formatting(query=query, response=query_label.strip().replace("\"", "")),
                    "labels": label,
                })

        pos_cnt = sum(row["labels"] for row in rows)
        print(f"DATA BALANCE: {pos_cnt}/{len(rows)}")

        training_dataset = Dataset.from_list(rows).shuffle(seed=self.config["seed"])
        return training_dataset
    
    def prepare_training_data(self, adapter_data_path, bbllms_outputs_path, user_embs_path):
        adapter_data = load_json_data(adapter_data_path)
        bbllms_outputs = load_open_ai_api_multi_response(bbllms_outputs_path)

        user_emb_ls = make_user_item_list_dict(torch.load(user_embs_path))
        self.user_embs_dict = {}
        self.get_user_emb_dict(user_emb_ls, self.user_embs_dict)

        training_data = self.make_training_dataset(adapter_data, bbllms_outputs)
        return training_data
    
    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, add_special_tokens=False)
    
    def train(self):
        if self.train_dataset is None:
            self.train_dataset = self.prepare_training_data(self.config["inference"]["train"]["data_path"], self.config["inference"]["train"]["output_file_path"], self.config["trainer"]["user_embs_path"])
            self.train_dataset = self.train_dataset.map(self.preprocess_function, batched=True, remove_columns=["text"])
        else:
            user_emb_ls = make_user_item_list_dict(torch.load(self.config["trainer"]["user_embs_path"]))
            self.user_embs_dict = {}
            self.get_user_emb_dict(user_emb_ls, self.user_embs_dict)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        ckpt_pth = self.config["trainer"]["output_dir"] + self.config["version"]

        data_collator = CustomDataCollator(
            tokenizer=self.tokenizer,
            user_embs_dict=self.user_embs_dict,
        )

        training_args = TrainingArguments(
            output_dir=ckpt_pth,
            learning_rate=self.config["trainer"]["learning_rate"],
            per_device_train_batch_size=self.config["trainer"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["trainer"]["gradient_accumulation_steps"],
            num_train_epochs=self.config["trainer"]["num_train_epochs"],
            weight_decay=self.config["trainer"]["weight_decay"],
            logging_steps=self.config["trainer"]["logging_steps"],
            logging_strategy=self.config["trainer"]["logging_strategy"],
            save_strategy=self.config["trainer"]["save_strategy"],
            save_total_limit=1,
            push_to_hub=self.config["trainer"]["push_to_hub"],
            dataloader_num_workers=1,
            remove_unused_columns=False, 
        )

        model_config = AutoConfig.from_pretrained(self.config["model_name"])
        model_config.cut_his_len = self.config["cut_his_len"]
        model_config.num_labels = 2
        model_config.num_facets = self.config["num_facets"]
        model_config.mof_top_k = self.config["mof_top_k"]

        base_model = AutoModel.from_pretrained(self.config["model_name"], config=model_config)
        self.model = FacetAdapter(model_config)
        self.model.bert.load_state_dict(base_model.state_dict(), strict=False)

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        if not os.path.exists(ckpt_pth):
            os.makedirs(ckpt_pth)
        trainer.model.save_pretrained(ckpt_pth)
        trainer.save_state()

    def fit(self):
        if self.fit_dataset is None:
            self.fit_dataset = self.prepare_training_data(self.config["inference"]["fit"]["data_path"], self.config["inference"]["fit"]["output_file_path"], self.config["fit"]["user_embs_path"])
            self.fit_dataset = self.fit_dataset.map(self.preprocess_function, batched=True, remove_columns=["text"])
        else:
            user_emb_ls = make_user_item_list_dict(torch.load(self.config["fit"]["user_embs_path"]))
            self.user_embs_dict = {}
            self.get_user_emb_dict(user_emb_ls, self.user_embs_dict)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        ckpt_pth = self.config["fit"]["output_dir"] + self.config["version"]

        data_collator = CustomDataCollator(
            tokenizer=self.tokenizer,
            user_embs_dict=self.user_embs_dict,
        )

        training_args = TrainingArguments(
            output_dir=ckpt_pth,
            learning_rate=self.config["fit"]["learning_rate"],
            per_device_train_batch_size=self.config["fit"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["fit"]["gradient_accumulation_steps"],
            num_train_epochs=self.config["fit"]["num_train_epochs"],
            weight_decay=self.config["fit"]["weight_decay"],
            logging_steps=self.config["fit"]["logging_steps"],
            logging_strategy=self.config["fit"]["logging_strategy"],
            save_strategy=self.config["fit"]["save_strategy"],
            save_total_limit=1,
            push_to_hub=self.config["fit"]["push_to_hub"],
            dataloader_num_workers=1,
            remove_unused_columns=False, 
        )

        if self.model is None:
            self.model = FacetAdapter.from_pretrained(self.config["trainer"]["output_dir"] + self.config["version"])
        for param in self.model.bert.parameters():
            param.requires_grad = False

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.fit_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        if not os.path.exists(ckpt_pth):
            os.makedirs(ckpt_pth)
        trainer.model.save_pretrained(ckpt_pth)
        trainer.save_state()
        
    def eval(self, is_fit):
        eval_data = load_json_data(self.config["inference"]["eval"]["data_path"])
        bbllms_outputs = load_open_ai_api_multi_response(self.config["inference"]["eval"]["output_file_path"])
        user_emb_ls = make_user_item_list_dict(torch.load(self.config["eval"]["user_embs_path"]))
        if is_fit:
            model = FacetAdapter.from_pretrained(self.config["fit"]["output_dir"]+self.config["version"])
        else:
            model = FacetAdapter.from_pretrained(self.config["trainer"]["output_dir"]+self.config["version"])
            
        model.to(self.device).eval()
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.config.eos_token_id = self.tokenizer.eos_token_id

        user_ids = []
        user_embs = []
        user_masks = []
        querys = []
        query_labels = []
        responses = []
        samples = []
        
        user_embs_dict = {}
        self.get_user_emb_dict(user_emb_ls, user_embs_dict)
            
        for idx in range(len(eval_data)):
            user_id = eval_data[idx]["id"].split("_")[0]
            assert (eval_data[idx]["id"] == bbllms_outputs[idx]["id"]) or (user_id == bbllms_outputs[idx]["id"])
            for response in bbllms_outputs[idx]["responses"]:
                user_ids.append(user_id)
                user_embs.append(user_embs_dict[user_id][0])
                user_masks.append(user_embs_dict[user_id][1])
                querys.append(eval_data[idx]["query"])
                
                if self.config['data_type'] in ['LaMP2', 'LaMP3']:
                    query_labels.append(eval_data[idx]["query_label"])
                    responses.append(postprocess_output(response))
                    samples.append(self.data_formatting(query=eval_data[idx]["query"], response=postprocess_output(response)))
                elif self.config['data_type'] in ['LaMP4', 'LaMP5']:
                    query_labels.append(eval_data[idx]["query_label"].strip().replace("\"", ""))
                    responses.append(response.strip().replace("\"", ""))
                    samples.append(self.data_formatting(query=eval_data[idx]["query"], response=response.strip().replace("\"", "")))
        
        eval_dict = {
            "user_ids": user_ids,
            "user_embs": user_embs,
            "user_masks": user_masks,
            "querys": querys,
            "query_labels": query_labels,
            "responses": responses,
            "text": samples,
        }

        solution_scores_dict = {}
        batch_size = self.config["eval"]["batch_size"]
        num_samples = len(eval_dict["user_ids"])

        for batch_start in tqdm(range(0, num_samples, batch_size),
                                total=math.ceil(num_samples / batch_size),
                                dynamic_ncols=True, ncols=None, leave=True):
            batch = {
                k: v[batch_start:batch_start + batch_size]
                for k, v in eval_dict.items()
            }

            input_batch = self.tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_batch["user_embs"] = torch.FloatTensor(batch["user_embs"])
            input_batch["user_masks"] = torch.BoolTensor(batch["user_masks"])
            input_batch = {k: v.to(self.device) for k, v in input_batch.items()}

            with torch.inference_mode():
                outputs = model(**input_batch)
                scores = torch.softmax(outputs.logits, dim=-1)[:, 1].detach().cpu().tolist()

            router_weight = outputs.router_weights.detach().cpu().numpy().tolist()
            topk_indices = outputs.topk_indices.detach().cpu().numpy().tolist()

            for i in range(len(batch["user_ids"])):
                user_id = batch["user_ids"][i]
                query = batch["querys"][i]
                query_label = batch["query_labels"][i]
                response = batch["responses"][i]
                score = scores[i]
                rw = router_weight[i]
                ti = topk_indices[i]

                if user_id not in solution_scores_dict:
                    solution_scores_dict[user_id] = {
                        "query": query,
                        "query_label": query_label,
                        "profile": [],
                    }

                solution_scores_dict[user_id]["profile"].append(
                    {
                        "response": response,
                        "score": score,
                        "rw": rw,
                        "ti": ti,
                    }
                )
                
        sorted_ranked_data = []
        for k, v in solution_scores_dict.items():
            sorted_ranked_data.append(
                    {
                        "id": k,
                        "query": v["query"],
                        "query_label": v["query_label"],
                        "profile": sorted(v["profile"], key=lambda x:x["score"], reverse=True)
                    }
                )
        
        json.dump(
            sorted_ranked_data, 
            open(self.config["eval"]["save_path"]+self.config["type"]+f"_v{self.config["version"]}"+".json", "w"),
            ensure_ascii=False, 
            indent=4
            )