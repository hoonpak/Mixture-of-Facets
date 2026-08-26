import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")
import os
import json
import torch
import random
import numpy as np

from umap import UMAP
from sklearn.cluster import KMeans

from user_emb import *
from data import *

def get_kmeans_cluster(user_info_embs, cluster_num):
    X_in = user_info_embs['user_emb_tensor'].numpy()
    if len(X_in) > cluster_num:
        umap_model = UMAP(n_neighbors=min(15, len(X_in) - 1), n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        reducted_embs = umap_model.fit_transform(X_in)
        kmeans = KMeans(n_clusters=cluster_num, random_state=42, n_init="auto").fit(reducted_embs)
        user_info_embs['cluster_label'] = kmeans.labels_
    else:
        user_info_embs['cluster_label'] = np.arange(len(X_in))

def lamp2_get_selected_history(user_info_embs, query, profile):
    return_ls = []
    for cluster_id in range(max(user_info_embs['cluster_label'])+1):
        cluster_indexs = np.where(user_info_embs['cluster_label'] == cluster_id)[0]
        if len(cluster_indexs) == 1:
            if profile[cluster_indexs[0]]['description'] != query:
                retrieved_doc = [profile[cluster_indexs[0]]]
            else:
                retrieved_doc = []
        else:
            cluster_profile = [profile[idx] for idx in cluster_indexs]
            retrieved_doc = lamp2_retrieve_top_k_with_bm25(cluster_profile, query, 1)
        return_ls += retrieved_doc
    return return_ls

def lamp2_get_cluster_based_reranker_augment_train_dataset(user_embs_path, cluster_num, random_query_num, currnet_dataset, save_path):
    user_embs = torch.load(user_embs_path)
    user_emb_ls = make_user_item_list_dict(user_embs)
    
    id_to_key_dict = {}
    for user in currnet_dataset:
        id_to_key_dict[user['id']] = user
        
    new_list = []
    for idx in tqdm(
        range(len(user_emb_ls)), total=len(user_emb_ls),
        dynamic_ncols=True, ncols=None, leave=True
        ):
        
        user_info_embs = user_emb_ls[idx]
        get_kmeans_cluster(user_info_embs, cluster_num)
        
        query = id_to_key_dict[user_info_embs['id']]['input'] 
        label = id_to_key_dict[user_info_embs['id']]['output'] 
        profile = [x for x in id_to_key_dict[user_info_embs['id']]['profile']] 
        
        if len(profile) > random_query_num:
            randomized_query = random.sample(profile, k=random_query_num) 
        else:
            randomized_query = [x for x in profile]
        
        profile.append({
            'tag': label,
            'description': extract_after_description(query),
            })
        query_ls = [extract_after_description(query)] + [que['description'] for que in randomized_query]
        output_ls = [label] + [que['tag'] for que in randomized_query]
        
        assert user_info_embs['user_text_list'][-1] == profile[-1]['description']
        assert len(user_info_embs['user_text_list']) == len(profile)
        
        iidx = 0
        for query, label in zip(query_ls,output_ls):
            if len(profile) > cluster_num:
                retrieved_ls = lamp2_get_selected_history(user_info_embs, query, profile)
            else:
                retrieved_ls = [item for item in profile if item['description'] != query]
                
            for item in retrieved_ls:
                if item['description'] != query:
                    assert item['description'] != query
                    new_list.append({"id": f"{user_info_embs['id']}_{iidx}",
                                    "augment_query": item['description'],
                                    "augment_label": item['tag'],
                                    "origin_query": query,
                                    "origin_label": label
                                    })
                    iidx += 1
                else:
                    pass

    with open(os.path.join(save_path, "train.json"), "w") as file:
        json.dump(new_list, file)

def lamp2_get_cluster_based_reranker_augment_fit_eval_dataset(user_embs_path, cluster_num, random_query_num, currnet_dataset, save_path):
    user_embs = torch.load(user_embs_path)
    user_emb_ls = make_user_item_list_dict(user_embs)
    
    id_to_key_dict = {}
    for user in currnet_dataset:
        id_to_key_dict[user['id']] = user
        
    new_eval_list = []
    new_fit_list = []

    for idx in tqdm(
        range(len(user_emb_ls)), total=len(user_emb_ls),
        dynamic_ncols=True, ncols=None, leave=True
        ):
        
        user_info_embs = user_emb_ls[idx]
        get_kmeans_cluster(user_info_embs, cluster_num)
        
        query = id_to_key_dict[user_info_embs['id']]['input'] 
        label = id_to_key_dict[user_info_embs['id']]['output'] 
        profile = [x for x in id_to_key_dict[user_info_embs['id']]['profile']]
        
        if len(profile) > random_query_num:
            randomized_query = random.sample(profile, k=random_query_num) 
        else:
            randomized_query = [x for x in profile]
        
        query_ls = [extract_after_description(query)] + [que['description'] for que in randomized_query]
        output_ls = [label] + [que['tag'] for que in randomized_query]
        
        assert user_info_embs['user_text_list'][-1] == profile[-1]['description']
        assert len(user_info_embs['user_text_list']) == len(profile)
        
        iidx = 0
        for query_idx, (query, label) in enumerate(zip(query_ls,output_ls)):
            if len(profile) > cluster_num:
                retrieved_ls = lamp2_get_selected_history(user_info_embs, query, profile)
            else:
                retrieved_ls = [item for item in profile if item['description'] != query]
            
            for item in retrieved_ls:
                if (item['description'] != query) and (query_idx == 0):
                    assert item['description'] != query
                    new_eval_list.append({"id": f"{user_info_embs['id']}_{iidx}",
                                    "augment_query": item['description'],
                                    "augment_label": item['tag'],
                                    "origin_query": query,
                                    "origin_label": label
                                    })
                    iidx += 1
                elif (item['description'] != query) and (query_idx != 0):
                    assert item['description'] != query
                    new_fit_list.append({"id": f"{user_info_embs['id']}_{iidx}",
                                    "augment_query": item['description'],
                                    "augment_label": item['tag'],
                                    "origin_query": query,
                                    "origin_label": label
                                    })
                    iidx += 1
                else:
                    pass

    with open(os.path.join(save_path, "fit_history.json"), "w") as file:
        json.dump(new_fit_list, file)
    with open(os.path.join(save_path, "eval.json"), "w") as file:
        json.dump(new_eval_list, file)
        
def lamp3_get_selected_history(user_info_embs, query, profile):
    return_ls = []
    for cluster_id in range(max(user_info_embs['cluster_label'])+1):
        cluster_indexs = np.where(user_info_embs['cluster_label'] == cluster_id)[0]
        if len(cluster_indexs) == 1:
            if profile[cluster_indexs[0]]['text'] != query: 
                retrieved_doc = [profile[cluster_indexs[0]]]
            else:
                retrieved_doc = []
        else:
            cluster_profile = [profile[idx] for idx in cluster_indexs]
            retrieved_doc = lamp3_retrieve_top_k_with_bm25(cluster_profile, query, 1)
        return_ls += retrieved_doc
    return return_ls

def lamp3_get_cluster_based_reranker_augment_train_dataset(user_embs_path, cluster_num, random_query_num, currnet_dataset, save_path):
    user_embs = torch.load(user_embs_path)
    user_emb_ls = make_user_item_list_dict(user_embs)
    
    id_to_key_dict = {}
    for user in currnet_dataset:
        id_to_key_dict[user['id']] = user
        
    new_list = []
    for idx in tqdm(
        range(len(user_emb_ls)), total=len(user_emb_ls),
        dynamic_ncols=True, ncols=None, leave=True
        ):
        
        user_info_embs = user_emb_ls[idx]
        get_kmeans_cluster(user_info_embs, cluster_num)
        
        query = id_to_key_dict[user_info_embs['id']]['input'] 
        label = id_to_key_dict[user_info_embs['id']]['output'] 
        profile = [x for x in id_to_key_dict[user_info_embs['id']]['profile']]
        
        if len(profile) > random_query_num:
            randomized_query = random.sample(profile, k=random_query_num) 
        else:
            randomized_query = [x for x in profile]
        
        profile.append({
            'score': label,
            'text': extract_after_review(query),
            })
        query_ls = [extract_after_review(query)] + [que['text'] for que in randomized_query]
        output_ls = [label] + [que['score'] for que in randomized_query]
        
        assert user_info_embs['user_text_list'][-1] == profile[-1]['text']
        assert len(user_info_embs['user_text_list']) == len(profile)
        
        iidx = 0
        for query, label in zip(query_ls,output_ls):
            if len(profile) > cluster_num:
                retrieved_ls = lamp3_get_selected_history(user_info_embs, query, profile)
            else:
                retrieved_ls = [item for item in profile if item['text'] != query]
                
            for item in retrieved_ls:
                if item['text'] != query:
                    assert item['text'] != query
                    new_list.append({"id": f"{user_info_embs['id']}_{iidx}",
                                    "augment_query": item['text'],
                                    "augment_label": item['score'],
                                    "origin_query": query,
                                    "origin_label": label
                                    })
                    iidx += 1
                else:
                    pass

    with open(os.path.join(save_path, "train.json"), "w") as file:
        json.dump(new_list, file)

def lamp3_get_cluster_based_reranker_augment_fit_eval_dataset(user_embs_path, cluster_num, random_query_num, currnet_dataset, save_path):
    user_embs = torch.load(user_embs_path)
    user_emb_ls = make_user_item_list_dict(user_embs)
    
    id_to_key_dict = {}
    for user in currnet_dataset:
        id_to_key_dict[user['id']] = user
        
    new_eval_list = []
    new_fit_list = []

    for idx in tqdm(
        range(len(user_emb_ls)), total=len(user_emb_ls),
        dynamic_ncols=True, ncols=None, leave=True
        ):
        
        user_info_embs = user_emb_ls[idx]
        get_kmeans_cluster(user_info_embs, cluster_num)
        
        query = id_to_key_dict[user_info_embs['id']]['input'] 
        label = id_to_key_dict[user_info_embs['id']]['output'] 
        profile = [x for x in id_to_key_dict[user_info_embs['id']]['profile']]
        
        if len(profile) > random_query_num:
            randomized_query = random.sample(profile, k=random_query_num) 
        else:
            randomized_query = [x for x in profile]
        
        query_ls = [extract_after_review(query)] + [que['text'] for que in randomized_query]
        output_ls = [label] + [que['score'] for que in randomized_query]
        
        assert user_info_embs['user_text_list'][-1] == profile[-1]['text']
        assert len(user_info_embs['user_text_list']) == len(profile)
        
        iidx = 0
        for query_idx, (query, label) in enumerate(zip(query_ls,output_ls)):
            if len(profile) > cluster_num:
                retrieved_ls = lamp3_get_selected_history(user_info_embs, query, profile)
            else:
                retrieved_ls = [item for item in profile if item['text'] != query]
            
            for item in retrieved_ls:
                if (item['text'] != query) and (query_idx == 0):
                    assert item['text'] != query
                    new_eval_list.append({"id": f"{user_info_embs['id']}_{iidx}",
                                    "augment_query": item['text'],
                                    "augment_label": item['score'],
                                    "origin_query": query,
                                    "origin_label": label
                                    })
                    iidx += 1
                elif (item['text'] != query) and (query_idx != 0):
                    assert item['text'] != query
                    new_fit_list.append({"id": f"{user_info_embs['id']}_{iidx}",
                                    "augment_query": item['text'],
                                    "augment_label": item['score'],
                                    "origin_query": query,
                                    "origin_label": label
                                    })
                    iidx += 1
                else:
                    pass

    with open(os.path.join(save_path, "fit_history.json"), "w") as file:
        json.dump(new_fit_list, file)
    with open(os.path.join(save_path, "eval.json"), "w") as file:
        json.dump(new_eval_list, file)
        
def lamp4_get_selected_history(user_info_embs, query, profile):
    return_ls = []
    for cluster_id in range(max(user_info_embs['cluster_label'])+1):
        cluster_indexs = np.where(user_info_embs['cluster_label'] == cluster_id)[0]
        if len(cluster_indexs) == 1:
            if profile[cluster_indexs[0]]['text'] != query: 
                retrieved_doc = [profile[cluster_indexs[0]]]
            else:
                retrieved_doc = []
        else:
            cluster_profile = [profile[idx] for idx in cluster_indexs]
            retrieved_doc = lamp4_retrieve_top_k_with_bm25(cluster_profile, query, 1)
        return_ls += retrieved_doc
    return return_ls

def lamp4_get_cluster_based_reranker_augment_train_dataset(user_embs_path, cluster_num, random_query_num, currnet_dataset, save_path):
    user_embs = torch.load(user_embs_path)
    user_emb_ls = make_user_item_list_dict(user_embs)
    
    id_to_key_dict = {}
    for user in currnet_dataset:
        id_to_key_dict[user['id']] = user
        
    new_list = []
    for idx in tqdm(
        range(len(user_emb_ls)), total=len(user_emb_ls),
        dynamic_ncols=True, ncols=None, leave=True
        ):
        
        user_info_embs = user_emb_ls[idx]
        get_kmeans_cluster(user_info_embs, cluster_num)
        
        query = id_to_key_dict[user_info_embs['id']]['input'] 
        label = id_to_key_dict[user_info_embs['id']]['output'] 
        profile = [x for x in id_to_key_dict[user_info_embs['id']]['profile']]
        
        if len(profile) > random_query_num:
            randomized_query = random.sample(profile, k=random_query_num) 
        else:
            randomized_query = [x for x in profile]
        
        profile.append({
            'title': label,
            'text': extract_after_article(query),
            })
        query_ls = [extract_after_article(query)] + [que['text'] for que in randomized_query]
        output_ls = [label] + [que['title'] for que in randomized_query]
        
        assert user_info_embs['user_text_list'][-1] == profile[-1]['text']
        assert len(user_info_embs['user_text_list']) == len(profile)
        
        iidx = 0
        for query, label in zip(query_ls,output_ls):
            if len(profile) > cluster_num:
                retrieved_ls = lamp4_get_selected_history(user_info_embs, query, profile)
            else:
                retrieved_ls = [item for item in profile if item['text'] != query]
                
            for item in retrieved_ls:
                if item['text'] != query:
                    assert item['text'] != query
                    new_list.append({"id": f"{user_info_embs['id']}_{iidx}",
                                    "augment_query": item['text'],
                                    "augment_label": item['title'],
                                    "origin_query": query,
                                    "origin_label": label
                                    })
                    iidx += 1
                else:
                    pass

    with open(os.path.join(save_path, "train.json"), "w") as file:
        json.dump(new_list, file)

def lamp4_get_cluster_based_reranker_augment_fit_eval_dataset(user_embs_path, cluster_num, random_query_num, currnet_dataset, save_path):
    user_embs = torch.load(user_embs_path)
    user_emb_ls = make_user_item_list_dict(user_embs)
    
    id_to_key_dict = {}
    for user in currnet_dataset:
        id_to_key_dict[user['id']] = user
        
    new_eval_list = []
    new_fit_list = []

    for idx in tqdm(
        range(len(user_emb_ls)), total=len(user_emb_ls),
        dynamic_ncols=True, ncols=None, leave=True
        ):
        
        user_info_embs = user_emb_ls[idx]
        get_kmeans_cluster(user_info_embs, cluster_num)
        
        query = id_to_key_dict[user_info_embs['id']]['input'] 
        label = id_to_key_dict[user_info_embs['id']]['output'] 
        profile = [x for x in id_to_key_dict[user_info_embs['id']]['profile']]
        
        if len(profile) > random_query_num:
            randomized_query = random.sample(profile, k=random_query_num) 
        else:
            randomized_query = [x for x in profile]
        
        query_ls = [extract_after_article(query)] + [que['text'] for que in randomized_query]
        output_ls = [label] + [que['title'] for que in randomized_query]
        
        assert user_info_embs['user_text_list'][-1] == profile[-1]['text']
        assert len(user_info_embs['user_text_list']) == len(profile)
        
        iidx = 0
        for query_idx, (query, label) in enumerate(zip(query_ls,output_ls)):
            if len(profile) > cluster_num:
                retrieved_ls = lamp4_get_selected_history(user_info_embs, query, profile)
            else:
                retrieved_ls = [item for item in profile if item['text'] != query]
            
            for item in retrieved_ls:
                if (item['text'] != query) and (query_idx == 0):
                    assert item['text'] != query
                    new_eval_list.append({"id": f"{user_info_embs['id']}_{iidx}",
                                    "augment_query": item['text'],
                                    "augment_label": item['title'],
                                    "origin_query": query,
                                    "origin_label": label
                                    })
                    iidx += 1
                elif (item['text'] != query) and (query_idx != 0):
                    assert item['text'] != query
                    new_fit_list.append({"id": f"{user_info_embs['id']}_{iidx}",
                                    "augment_query": item['text'],
                                    "augment_label": item['title'],
                                    "origin_query": query,
                                    "origin_label": label
                                    })
                    iidx += 1
                else:
                    pass

    with open(os.path.join(save_path, "fit_history.json"), "w") as file:
        json.dump(new_fit_list, file)
    with open(os.path.join(save_path, "eval.json"), "w") as file:
        json.dump(new_eval_list, file)

def lamp5_get_selected_history(user_info_embs, query, profile):
    return_ls = []
    for cluster_id in range(max(user_info_embs['cluster_label'])+1):
        cluster_indexs = np.where(user_info_embs['cluster_label'] == cluster_id)[0]
        if len(cluster_indexs) == 1:
            if profile[cluster_indexs[0]]['abstract'] != query: 
                retrieved_doc = [profile[cluster_indexs[0]]]
            else:
                retrieved_doc = []
        else:
            cluster_profile = [profile[idx] for idx in cluster_indexs]
            retrieved_doc = lamp5_retrieve_top_k_with_bm25(cluster_profile, query, 1)
        return_ls += retrieved_doc
    return return_ls

def lamp5_get_cluster_based_reranker_augment_train_dataset(user_embs_path, cluster_num, random_query_num, currnet_dataset, save_path):
    user_embs = torch.load(user_embs_path)
    user_emb_ls = make_user_item_list_dict(user_embs)
    
    id_to_key_dict = {}
    for user in currnet_dataset:
        id_to_key_dict[user['id']] = user
        
    new_list = []
    for idx in tqdm(
        range(len(user_emb_ls)), total=len(user_emb_ls),
        dynamic_ncols=True, ncols=None, leave=True
        ):
        
        user_info_embs = user_emb_ls[idx]
        get_kmeans_cluster(user_info_embs, cluster_num)
        
        query = id_to_key_dict[user_info_embs['id']]['input'] 
        label = id_to_key_dict[user_info_embs['id']]['output'] 
        profile = [x for x in id_to_key_dict[user_info_embs['id']]['profile']]
        
        if len(profile) > random_query_num:
            randomized_query = random.sample(profile, k=random_query_num) 
        else:
            randomized_query = [x for x in profile]
        
        profile.append({
            'title': label,
            'abstract': extract_after_paper(query),
            })
        query_ls = [extract_after_paper(query)] + [que['abstract'] for que in randomized_query]
        output_ls = [label] + [que['title'] for que in randomized_query]
        
        assert user_info_embs['user_text_list'][-1] == profile[-1]['abstract']
        assert len(user_info_embs['user_text_list']) == len(profile)
        
        iidx = 0
        for query, label in zip(query_ls,output_ls):
            if len(profile) > cluster_num:
                retrieved_ls = lamp5_get_selected_history(user_info_embs, query, profile)
            else:
                retrieved_ls = [item for item in profile if item['abstract'] != query]
                
            for item in retrieved_ls:
                if item['abstract'] != query:
                    assert item['abstract'] != query
                    new_list.append({"id": f"{user_info_embs['id']}_{iidx}",
                                    "augment_query": item['abstract'],
                                    "augment_label": item['title'],
                                    "origin_query": query,
                                    "origin_label": label
                                    })
                    iidx += 1
                else:
                    pass

    with open(os.path.join(save_path, "train.json"), "w") as file:
        json.dump(new_list, file)

def lamp5_get_cluster_based_reranker_augment_fit_eval_dataset(user_embs_path, cluster_num, random_query_num, currnet_dataset, save_path):
    user_embs = torch.load(user_embs_path)
    user_emb_ls = make_user_item_list_dict(user_embs)
    
    id_to_key_dict = {}
    for user in currnet_dataset:
        id_to_key_dict[user['id']] = user
        
    new_eval_list = []
    new_fit_list = []

    for idx in tqdm(
        range(len(user_emb_ls)), total=len(user_emb_ls),
        dynamic_ncols=True, ncols=None, leave=True
        ):
        
        user_info_embs = user_emb_ls[idx]
        get_kmeans_cluster(user_info_embs, cluster_num)
        
        query = id_to_key_dict[user_info_embs['id']]['input'] 
        label = id_to_key_dict[user_info_embs['id']]['output'] 
        profile = [x for x in id_to_key_dict[user_info_embs['id']]['profile']]
        
        if len(profile) > random_query_num:
            randomized_query = random.sample(profile, k=random_query_num) 
        else:
            randomized_query = [x for x in profile]
        
        query_ls = [extract_after_paper(query)] + [que['abstract'] for que in randomized_query]
        output_ls = [label] + [que['title'] for que in randomized_query]
        
        assert user_info_embs['user_text_list'][-1] == profile[-1]['abstract']
        assert len(user_info_embs['user_text_list']) == len(profile)
        
        iidx = 0
        for query_idx, (query, label) in enumerate(zip(query_ls,output_ls)):
            if len(profile) > cluster_num:
                retrieved_ls = lamp5_get_selected_history(user_info_embs, query, profile)
            else:
                retrieved_ls = [item for item in profile if item['abstract'] != query]
            
            for item in retrieved_ls:
                if (item['abstract'] != query) and (query_idx == 0):
                    assert item['abstract'] != query
                    new_eval_list.append({"id": f"{user_info_embs['id']}_{iidx}",
                                    "augment_query": item['abstract'],
                                    "augment_label": item['title'],
                                    "origin_query": query,
                                    "origin_label": label
                                    })
                    iidx += 1
                elif (item['abstract'] != query) and (query_idx != 0):
                    assert item['abstract'] != query
                    new_fit_list.append({"id": f"{user_info_embs['id']}_{iidx}",
                                    "augment_query": item['abstract'],
                                    "augment_label": item['title'],
                                    "origin_query": query,
                                    "origin_label": label
                                    })
                    iidx += 1
                else:
                    pass

    with open(os.path.join(save_path, "fit_history.json"), "w") as file:
        json.dump(new_fit_list, file)
    with open(os.path.join(save_path, "eval.json"), "w") as file:
        json.dump(new_eval_list, file)