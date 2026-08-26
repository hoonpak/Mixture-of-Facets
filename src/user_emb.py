import torch
import numpy as np

from tqdm import tqdm

from data import *

def create_classification_movies_prompt_for_embs(inp, tag, max_length, tokenizer):
    needed_part_len = len(tokenizer(f'The tag for the movie: " " is "{tag}" ', add_special_tokens=False)['input_ids'])
    tokens = tokenizer(inp, max_length=max_length - needed_part_len - 2, truncation=True, add_special_tokens=False)
    new_text = tokenizer.decode([tokens['input_ids']], skip_special_tokens=True)
    prompt = f'The tag for the movie: "{new_text}" is "{tag}" '
    return prompt

def lamp2_make_item_embeddings(model, tokenizer, batch_size, tmp_dataset, output_path, is_test):
    model = model.to("cuda:0")
    model.eval()

    ids = []
    descriptions = []
    description_embs = []
    input_data_list = []

    for sample in tmp_dataset:
        profile_list = [x for x in sample['profile']]
        user_id = sample['id']
        
        if not is_test:
            profile_list.append({
                'tag': sample['output'],
                'description': extract_after_description(sample['input']),
                'id': sample['id']
            })
            
        unique_idx = 0
        for item in profile_list:
            input_data_list.append(create_classification_movies_prompt_for_embs(item['description'], item['tag'], 512, tokenizer))
            descriptions.append(item['description'])
            ids.append(f"{user_id}_{unique_idx}")
            unique_idx += 1

    for batch_start_point in tqdm(
        range(0, len(input_data_list), batch_size),
        total=len(input_data_list) // batch_size,
        dynamic_ncols=True, ncols=None, leave=True
        ):
        
        batch = input_data_list[batch_start_point:batch_start_point + batch_size]

        input_batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        input_batch = {k: v.to("cuda:0") for k, v in input_batch.items()}

        with torch.inference_mode():
            outputs = model(**input_batch)

        cls_emb = outputs[0][:, 0, :].detach().cpu()  # (batch, hidden)    
        description_embs.append(cls_emb)

    description_embs = torch.cat(description_embs, dim=0)
    assert description_embs.shape[0] == len(ids) == len(descriptions)

    torch.save({
        "ids": ids,
        "texts": descriptions,
        "embeddings": description_embs
    }, output_path)

def create_classification_reviews_prompt_for_embs(inp, score, max_length, tokenizer):
    needed_part_len = len(tokenizer(f'{score} is the score for " " ', add_special_tokens=False)['input_ids'])
    tokens = tokenizer(inp, max_length=max_length - needed_part_len - 2, add_special_tokens=False, truncation=True)
    new_text = tokenizer.decode([tokens['input_ids']], skip_special_tokens=True)
    prompt = f'{score} is the score for "{new_text}" '
    return prompt

def lamp3_make_item_embeddings(model, tokenizer, batch_size, tmp_dataset, output_path, is_test):
    model = model.to("cuda:0")
    model.eval()

    ids = []
    texts = []
    text_embs = []
    input_data_list = []

    for sample in tmp_dataset:
        profile_list = [x for x in sample['profile']]
        user_id = sample['id']
        
        if not is_test:
            profile_list.append({
                'score': sample['output'],
                'text': extract_after_review(sample['input']),
                'id': sample['id']
            })
            
        unique_idx = 0
        for item in profile_list:
            input_data_list.append(create_classification_reviews_prompt_for_embs(item['text'], item['score'], 512, tokenizer))
            texts.append(item['text'])
            ids.append(f"{user_id}_{unique_idx}")
            unique_idx += 1

    for batch_start_point in tqdm(
        range(0, len(input_data_list), batch_size),
        total=len(input_data_list) // batch_size,
        dynamic_ncols=True, ncols=None, leave=True
        ):
        
        batch = input_data_list[batch_start_point:batch_start_point + batch_size]

        input_batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        input_batch = {k: v.to("cuda:0") for k, v in input_batch.items()}

        with torch.inference_mode():
            outputs = model(**input_batch)

        cls_emb = outputs[0][:, 0, :].detach().cpu()  # (batch, hidden)    
        text_embs.append(cls_emb)

    text_embs = torch.cat(text_embs, dim=0)
    assert text_embs.shape[0] == len(ids) == len(texts)

    torch.save({
        "ids": ids,
        "texts": texts,
        "embeddings": text_embs
    }, output_path)

def create_generation_news_prompt_for_embs(inp, title, max_length, tokenizer):
    needed_part_len = len(tokenizer(f'"{title}" is the title for " " ')['input_ids'])
    tokens = tokenizer(inp, max_length=max_length - needed_part_len - 2, add_special_tokens=False, truncation=True)
    new_text = tokenizer.decode([tokens['input_ids']], skip_special_tokens=True)
    prompt = f'"{title}" is the title for "{new_text}" '
    return prompt

def lamp4_make_item_embeddings(model, tokenizer, batch_size, tmp_dataset, output_path, is_test):
    model = model.to("cuda:0")
    model.eval()

    ids = []
    texts = []
    text_embs = []
    input_data_list = []

    for sample in tmp_dataset:
        profile_list = [x for x in sample['profile']]
        user_id = sample['id']
        
        if not is_test:
            profile_list.append({
                'title': sample['output'],
                'text': extract_after_article(sample['input']),
                'id': sample['id']
            })
            
        unique_idx = 0
        for item in profile_list:
            input_data_list.append(create_generation_news_prompt_for_embs(item['text'], item['title'], 512, tokenizer))
            texts.append(item['text'])
            ids.append(f"{user_id}_{unique_idx}")
            unique_idx += 1

    for batch_start_point in tqdm(
        range(0, len(input_data_list), batch_size),
        total=len(input_data_list) // batch_size,
        dynamic_ncols=True, ncols=None, leave=True
        ):
        
        batch = input_data_list[batch_start_point:batch_start_point + batch_size]

        input_batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        input_batch = {k: v.to("cuda:0") for k, v in input_batch.items()}

        with torch.inference_mode():
            outputs = model(**input_batch)

        cls_emb = outputs[0][:, 0, :].detach().cpu()  # (batch, hidden)    
        text_embs.append(cls_emb)

    text_embs = torch.cat(text_embs, dim=0)
    assert text_embs.shape[0] == len(ids) == len(texts)

    torch.save({
        "ids": ids,
        "texts": texts,
        "embeddings": text_embs
    }, output_path)

def create_generation_paper_prompt_for_embs(inp, title, max_length, tokenizer):
    needed_part_len = len(tokenizer(f'"{title}" is a title " " ')['input_ids'])
    tokens = tokenizer(inp, max_length=max_length - needed_part_len - 2, add_special_tokens=False, truncation=True)
    new_text = tokenizer.decode([tokens['input_ids']], skip_special_tokens=True)
    prompt = f'"{title}" is the title for "{new_text}" '
    return prompt

def lamp5_make_item_embeddings(model, tokenizer, batch_size, tmp_dataset, output_path, is_test):
    model = model.to("cuda:0")
    model.eval()

    ids = []
    texts = []
    text_embs = []
    input_data_list = []

    for sample in tmp_dataset:
        profile_list = [x for x in sample['profile']]
        user_id = sample['id']
        
        if not is_test:
            profile_list.append({
                'title': sample['output'],
                'abstract': extract_after_paper(sample['input']),
                'id': sample['id']
            })
            
        unique_idx = 0
        for item in profile_list:
            input_data_list.append(create_generation_paper_prompt_for_embs(item['abstract'], item['title'], 512, tokenizer))
            texts.append(item['abstract'])
            ids.append(f"{user_id}_{unique_idx}")
            unique_idx += 1

    for batch_start_point in tqdm(
        range(0, len(input_data_list), batch_size),
        total=len(input_data_list) // batch_size,
        dynamic_ncols=True, ncols=None, leave=True
        ):
        
        batch = input_data_list[batch_start_point:batch_start_point + batch_size]

        input_batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        input_batch = {k: v.to("cuda:0") for k, v in input_batch.items()}

        with torch.inference_mode():
            outputs = model(**input_batch)

        cls_emb = outputs[0][:, 0, :].detach().cpu()  # (batch, hidden)    
        text_embs.append(cls_emb)

    text_embs = torch.cat(text_embs, dim=0)
    assert text_embs.shape[0] == len(ids) == len(texts)

    torch.save({
        "ids": ids,
        "texts": texts,
        "embeddings": text_embs
    }, output_path)

def make_user_item_list_dict(item_embs):
    user_data_index_dict = {}
    for idx, data_id in enumerate(item_embs['ids']):
        user_id = data_id.split("_")[0]
        if user_id not in user_data_index_dict:
            user_data_index_dict[user_id] = []
        user_data_index_dict[user_id].append(idx)
        
    user_emb_ls = []
    for user_id, user_data_list in user_data_index_dict.items():
        user_emb_ls.append({
            'id': user_id,
            'user_emb_tensor' : item_embs['embeddings'][user_data_list],
            # 'user_text_list' : np.array(item_embs['texts'])[user_data_list].tolist()
            })
    return user_emb_ls