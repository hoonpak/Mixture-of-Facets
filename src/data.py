import json
from rank_bm25 import BM25Okapi

def load_json_data(cur_path):
    return json.load(open(cur_path, "r"))

def get_prompt_function(data_type):
    if data_type == "LaMP2":
        def create_classification_movies_prompt(inp, profile, max_length, tokenizer):
            per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
            saved_tokens = 0
            prompts = []
            for p in profile:
                needed_part_len = len(tokenizer(f'The tag for the movie: " " is "{p["tag"]}" ')['input_ids'])
                tokens = tokenizer(p["description"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
                saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
                new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
                prompt = f'The tag for the movie: "{new_text}" is "{p["tag"]}" '
                prompts.append(prompt)
            return f'{", and ".join(prompts)}. {inp}'
        return create_classification_movies_prompt
    elif data_type == "LaMP3":
        def create_classification_review_prompt(inp, profile, max_length, tokenizer):
            per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
            saved_tokens = 0
            prompts = []
            for p in profile:
                needed_part_len = len(tokenizer(f'{p["score"]} is the score for " " ')['input_ids'])
                tokens = tokenizer(p["text"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
                saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
                new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
                prompt = f'{p["score"]} is the score for "{new_text}" '
                prompts.append(prompt)
            return f'{", and ".join(prompts)}. {inp}'
        return create_classification_review_prompt
    elif data_type == "LaMP4":
        def create_generation_news_prompt(inp, profile, max_length, tokenizer):
            per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
            saved_tokens = 0
            prompts = []
            for p in profile:
                needed_part_len = len(tokenizer(f'"{p["title"]}" is the title for " " ')['input_ids'])
                tokens = tokenizer(p["text"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
                saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
                new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
                prompt = f'"{p["title"]}" is the title for "{new_text}" '
                prompts.append(prompt)
            return f'{", and ".join(prompts)}. {inp}'
        return create_generation_news_prompt
    elif data_type == "LaMP5":
        def create_generation_paper_prompt(inp, profile, max_length, tokenizer):
            per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1) - len(tokenizer("Following the given patterns")['input_ids'])) // len(profile)
            saved_tokens = 0
            prompts = []
            for p in profile:
                needed_part_len = len(tokenizer(f'"{p["title"]}" is a title " " ')['input_ids'])
                tokens = tokenizer(p["abstract"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
                saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
                new_asbtract = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
                prompt = f'"{p["title"]}" is a title for "{new_asbtract}" '
                prompts.append(prompt)
            return f'{", and ".join(prompts)}. Following the given patterns {inp}'
        return create_generation_paper_prompt
    else:
        raise ValueError(f"Invalid data_type: {data_type}")

def get_prompt_function_for_reranker(data_type):
    if data_type == "LaMP2":
        def create_classification_movies_prompt(inp, profile, max_length, tokenizer):
            per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
            saved_tokens = 0
            prompts = []
            for p in profile:
                needed_part_len = len(tokenizer(f'The tag for the movie: " " is "{p["augment_label"]}" ')['input_ids'])
                tokens = tokenizer(p["augment_query"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
                saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
                new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
                prompt = f'The tag for the movie: "{new_text}" is "{p["augment_label"]}" '
                prompts.append(prompt)
            return f'{", and ".join(prompts)}. {inp}'
        return create_classification_movies_prompt
    elif data_type == "LaMP3":
        def create_classification_review_prompt(inp, profile, max_length, tokenizer):
            per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
            saved_tokens = 0
            prompts = []
            for p in profile:
                needed_part_len = len(tokenizer(f'{p["augment_label"]} is the score for " " ')['input_ids'])
                tokens = tokenizer(p["augment_query"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
                saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
                new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
                prompt = f'{p["augment_label"]} is the score for "{new_text}" '
                prompts.append(prompt)
            return f'{", and ".join(prompts)}. {inp}'
        return create_classification_review_prompt
    elif data_type == "LaMP4":
        def create_generation_news_prompt(inp, profile, max_length, tokenizer):
            per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
            saved_tokens = 0
            prompts = []
            for p in profile:
                needed_part_len = len(tokenizer(f'"{p["augment_label"]}" is the title for " " ')['input_ids'])
                tokens = tokenizer(p["augment_query"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
                saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
                new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
                prompt = f'"{p["augment_label"]}" is the title for "{new_text}" '
                prompts.append(prompt)
            return f'{", and ".join(prompts)}. {inp}'
        return create_generation_news_prompt    
    elif data_type == "LaMP5":
        def create_generation_paper_prompt(inp, profile, max_length, tokenizer):
            per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1) - len(tokenizer("Following the given patterns")['input_ids'])) // len(profile)
            saved_tokens = 0
            prompts = []
            for p in profile:
                needed_part_len = len(tokenizer(f'"{p["augment_label"]}" is a title " " ')['input_ids'])
                tokens = tokenizer(p["augment_query"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
                saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
                new_asbtract = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
                prompt = f'"{p["augment_label"]}" is a title for "{new_asbtract}" '
                prompts.append(prompt)
            return f'{", and ".join(prompts)}. Following the given patterns {inp}'
        return create_generation_paper_prompt
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
    
def get_prompt_function_for_adapter(data_type):
    if data_type == "LaMP2":
        def create_classification_movies_prompt(inp, profile, max_length, tokenizer):
            per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
            saved_tokens = 0
            prompts = []
            for p in profile:
                needed_part_len = len(tokenizer(f'The tag for the movie: " " is "{p["augment_label"]}" ')['input_ids'])
                tokens = tokenizer(p["augment_query"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
                saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
                new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
                prompt = f'The tag for the movie: "{new_text}" is "{p["augment_label"]}" '
                prompts.append(prompt)
            return f'{", and ".join(prompts)}. {inp}'
        return create_classification_movies_prompt
    elif data_type == "LaMP3":
        def create_classification_review_prompt(inp, profile, max_length, tokenizer):
            per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
            saved_tokens = 0
            prompts = []
            for p in profile:
                needed_part_len = len(tokenizer(f'{p["augment_label"]} is the score for " " ')['input_ids'])
                tokens = tokenizer(p["augment_query"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
                saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
                new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
                prompt = f'{p["augment_label"]} is the score for "{new_text}" '
                prompts.append(prompt)
            return f'{", and ".join(prompts)}. {inp}'
        return create_classification_review_prompt
    elif data_type == "LaMP4":
        def create_generation_news_prompt(inp, profile, max_length, tokenizer):
            per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
            saved_tokens = 0
            prompts = []
            for p in profile:
                needed_part_len = len(tokenizer(f'"{p["augment_label"]}" is the title for " " ')['input_ids'])
                tokens = tokenizer(p["augment_query"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
                saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
                new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
                prompt = f'"{p["augment_label"]}" is the title for "{new_text}" '
                prompts.append(prompt)
            return f'{", and ".join(prompts)}. {inp}'
        return create_generation_news_prompt
    elif data_type == "LaMP5":
        def create_generation_paper_prompt(inp, profile, max_length, tokenizer):
            per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1) - len(tokenizer("Following the given patterns")['input_ids'])) // len(profile)
            saved_tokens = 0
            prompts = []
            for p in profile:
                needed_part_len = len(tokenizer(f'"{p["augment_label"]}" is a title " " ')['input_ids'])
                tokens = tokenizer(p["augment_query"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
                saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
                new_asbtract = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
                prompt = f'"{p["augment_label"]}" is a title for "{new_asbtract}" '
                prompts.append(prompt)
            return f'{", and ".join(prompts)}. Following the given patterns {inp}'
        return create_generation_paper_prompt
    else:
        raise ValueError(f"Invalid data_type: {data_type}")

def extract_after_description(input_query):
    pivot_index = input_query.find("description:")
    if pivot_index == -1:
        return None
    return input_query[pivot_index + len("description:"):].strip()

def lamp2_retrieve_top_k_with_bm25(profile, query, top_k):
    filtered_profile = [item for item in profile if (item['description'] != query) and (item['description'] != "")]
    if len(filtered_profile) == 0:
        return []
    else:
        tokenized_documents = [item['description'].split() for item in filtered_profile]
        bm25 = BM25Okapi(tokenized_documents)
        tokenized_query = query.split()
        return bm25.get_top_n(tokenized_query, filtered_profile, n=top_k)

def extract_after_review(input_string):
    article_index = input_string.find('review:')
    if article_index == -1:
        return None
    return input_string[article_index + len('review:'):].strip()

def lamp3_retrieve_top_k_with_bm25(profile, query, k):
    filtered_profile = [item for item in profile if (item['text'] != query) and (item['text'] != "")]
    if len(filtered_profile) == 0:
        return []
    else:
        tokenized_documents = [item['text'].split() for item in filtered_profile]
        bm25 = BM25Okapi(tokenized_documents)
        tokenized_query = query.split()
        return bm25.get_top_n(tokenized_query, filtered_profile, n=k)
    
def extract_after_article(input_string):
    article_index = input_string.find('article:')
    if article_index == -1:
        return None
    return input_string[article_index + len('article:'):].strip()

def lamp4_retrieve_top_k_with_bm25(profile, query, k):
    filtered_profile = [item for item in profile if (item['text'] != query) and (item['text'] != "")]
    if len(filtered_profile) == 0:
        return []
    else:
        tokenized_documents = [item['text'].split() for item in filtered_profile]
        bm25 = BM25Okapi(tokenized_documents)
        tokenized_query = query.split()
        return bm25.get_top_n(tokenized_query, filtered_profile, n=k)

def extract_after_paper(input_string):
    article_index = input_string.find('paper:')
    if article_index == -1:
        return None
    return input_string[article_index + len('paper:'):].strip()

def lamp5_retrieve_top_k_with_bm25(profile, query, k):
    filtered_profile = [item for item in profile if (item['abstract'] != query) and (item['abstract'] != "")]
    if len(filtered_profile) == 0:
        return []
    else:
        tokenized_documents = [item['abstract'].split() for item in filtered_profile]
        bm25 = BM25Okapi(tokenized_documents)
        tokenized_query = query.split()
        return bm25.get_top_n(tokenized_query, filtered_profile, n=k)
    
def postprocess_output(pred):
    pred = pred.lower().strip()
    if '"' in pred:
        pred = pred.replace('"', "")
    if '[' in pred:
        pred = pred.replace("[", "")
    if ']' in pred:
        pred = pred.replace("]", "")
    if ',' in pred:
        pred = pred.split(',')[0]
    if '!' in pred:
        pred = pred.split('!')[0]
    return pred

def load_open_ai_api_response(response_file_path):
    with open(response_file_path, "r") as file:
        api_resonses = []
        for line in file.readlines():
            tmp_dict = json.loads(line)
            response = tmp_dict["response"]["body"]["choices"][0]["message"]["content"]
            api_resonses.append({
                'id' : tmp_dict['custom_id'],
                'response' : response
            })
    return api_resonses

def load_open_ai_api_multi_response(response_file_path):
    with open(response_file_path, "r") as file:
        api_resonses = []
        for line in file.readlines():
            tmp_dict = json.loads(line)
            responses = [sample["message"]["content"] for sample in tmp_dict["response"]["body"]["choices"]] 
            api_resonses.append({
                'id' : tmp_dict['custom_id'],
                'responses' : responses
            })
    return api_resonses