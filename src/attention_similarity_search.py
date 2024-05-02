# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:50:07 2024

@author: Mico
"""

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
from skimage import io
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cos_dist
from skimage.measure import block_reduce
from  scipy.ndimage import convolve
from skimage.measure import label, regionprops
from sklearn.metrics.pairwise import cosine_distances
import cv2


def get_path_embed(image_filename):
    with h5py.File(hdf5_path, 'r') as h5f:
        patch_embed = h5f[f'{image_filename}/features'][()]
    return patch_embed

def get_attentions(head_patch_tokens, head_cls_token, threshold=0.3, normalize=False):
    if normalize:
        head_patch_tokens = head_patch_tokens / np.linalg.norm(head_patch_tokens, axis=-1).reshape(-1,1)
        head_cls_token = head_cls_token / np.linalg.norm(head_cls_token, axis=-1).reshape(-1,1)
    attentions = np.squeeze(np.matmul(head_patch_tokens, np.squeeze(head_cls_token).T))

    idx = np.argsort(attentions)
    val = attentions[idx]
    val /= np.sum(val)
    cumval = np.cumsum(val)

    th_attn = cumval > (1 - threshold)
    th_attn = th_attn[np.argsort(idx)]
    attentions[np.logical_not(th_attn)] = attentions.min()
    return attentions

def get_mean_attention(query_cls_token, patch_embed, num_heads, feature_dim, threshold=0.3):
    head_dim = feature_dim // num_heads

    query_att_per_head = []
    new_grid_shape = patch_embed.shape[0:2]
    patch_tokens = patch_embed.reshape(-1, feature_dim)
    for head in range(0, num_heads):
        query_attentions = get_attentions(patch_tokens[:, head*head_dim:(head+1)*head_dim],
                                          query_cls_token[head*head_dim:(head+1)*head_dim,:],
                                          threshold=threshold,
                                          normalize=False)
        attention_map = query_attentions.reshape(new_grid_shape)
        query_att_per_head.append(attention_map)
    query_att_per_head = np.dstack(query_att_per_head)
    query_mean_attn = query_att_per_head.mean(axis=-1)

    return query_mean_attn


def compute_average_precision(ranking):
    ids = np.arange(1, len(ranking)+1)
    recall = np.cumsum(ranking) * ranking
    valid_ids = recall != 0
    ids = ids[valid_ids]
    recall = recall[valid_ids]
    precision = (recall / ids)
    recall = recall / np.max(recall)
    AP = np.mean(precision)
    return AP, precision, recall

def interpolate_pr(precision, recall, bins=11):
    recall_interpolated = np.linspace(0, 1, bins)
    precision_interpolated = np.interp(recall_interpolated, recall, precision)

    for i in range(0, len(recall_interpolated)):
        precision_interpolated[-i] = np.max(precision_interpolated[-i:])
    return precision_interpolated, recall_interpolated


def display_search_results(query_image, doc_images, title=None):
    top_k = len(doc_images)
    fig, axs = plt.subplots(1, top_k+2, figsize=(16, 8))
    axs[0].imshow(query_image)
    axs[0].axis('off')
    axs[1].imshow(query_image*0+255)
    axs[1].axis('off')
    for i in range(0, top_k):
        axs[i+2].imshow(doc_images[i])
        axs[i+2].axis('off')
    if title:
        plt.title(title)
    plt.show()

def normalize(data):
    norm = np.sqrt(np.sum(np.square(data), axis = 1))
    norm = np.expand_dims(norm, 0)  
    data = data / np.transpose(norm)
    return data

def square_root_norm(data) :
    return normalize(np.sign(data)*np.sqrt(np.abs(data))) 

dataset_path = r'D:\datasets\medieval_images\DocExplore_images'
images_path = os.path.join('..', 'data', 'images')
queries_folder = r'D:\datasets\medieval_images\evaluation_kit_v2\DocExplore_queries_web' 

df_queries_path = os.path.join('..', 'data', 'df_queries.parquet')
queries_features_path = os.path.join('..', 'data', 'queries_cls_tokens.npy')
df_path = os.path.join('..', 'data', 'df_dataset.parquet')
hdf5_path = os.path.join('..', 'data', 'docs_patch_embed.hdf5')

df_queries = pd.read_parquet(df_queries_path)
df_docs = pd.read_parquet(df_path)

cls_tokens = np.load(queries_features_path)

num_heads = 1
feature_dim = 384
num_pools = 6
num_patch_pools = 1

use_multiscale_patches = True
display_images = False
display_rank = False
APs = []
precisions = []
df_queries.reset_index(inplace=True, names='query_id')
#df_queries = df_queries[df_queries['size_cat'] == 'big']
#df_queries = df_queries[df_queries['size_cat'] != 'big']
#df_queries = df_queries[df_queries['ar_cat'] == 'square']
#df_queries['c'] = df_queries['class']
#df_queries = df_queries.groupby('c').first()
#df_queries = df_queries[df_queries['class'] == 'henri_d']
df_queries = df_queries.set_index('query_id')

df_queries.sort_index(inplace=True)
evaluation_results = []

for q_idx, q_row in tqdm(df_queries.iterrows(), total=df_queries.shape[0]):
    query_img = io.imread(os.path.join(queries_folder, q_row['img_path']))

    query_cls_token = cls_tokens[q_idx, :].reshape(-1, 1)
    scores = []
    if display_images:
        io.imshow(query_img)
        io.show()
    for index, row in df_docs.iterrows():
        patch_embed = get_path_embed(row['img_path'])
        if use_multiscale_patches:
            multiscale_patch_embeds = []
            for i in range(0, num_patch_pools):
                if min(patch_embed.shape) > 3:
                    if i > 0:
                        patch_embed = block_reduce(patch_embed, (2, 2, 1), np.mean)
                        #kernel = np.ones((3, 3, 1)) / 9
                        #patch_embed = convolve(patch_embed, kernel)[::2, ::2,:]
                multiscale_patch_embeds.append(patch_embed.reshape(-1, feature_dim))
            multiscale_patch_embeds = np.vstack(multiscale_patch_embeds)
            distances = np.linalg.norm(multiscale_patch_embeds - query_cls_token.T, axis=1)
            #distances = cosine_distances(multiscale_patch_embeds, query_cls_token.T)
            score = min(np.squeeze(distances))
            scores.append(score)
        else:
            mean_attention = get_mean_attention(query_cls_token, patch_embed,
                                                num_heads, feature_dim,
                                                threshold=0.2)
            masked_attn = mean_attention[mean_attention > 0]
            if len(masked_attn) > 1:
                #display_images = (row['doc_id'] in q_row['documents']) and len(masked_attn) > (query_img.shape[0]//32)**2
                #display_images = display_images and (masked_attn.max() > 100)
    
                th = max(np.percentile(masked_attn, 10), 0)
                multiscale_maps = []
                map_avg_pool = mean_attention
                map_shape = mean_attention.shape
                proposal_tokens = []
                boxes = []
                for i in range(0, num_pools):
                    if min(map_avg_pool.shape) > 3:
                        if i > 0:
                            map_avg_pool = block_reduce(map_avg_pool, (2, 2), np.mean)
                        multiscale_maps.append(map_avg_pool)
                        labeled_attn = label(map_avg_pool > th)
                        regions = regionprops(labeled_attn)
                        scale_x = map_shape[1] / map_avg_pool.shape[1]
                        scale_y = map_shape[0] / map_avg_pool.shape[0]
                        for props in regions:
                            ymin, xmin, ymax, xmax = props.bbox
                            ymin, ymax = int(ymin * scale_y), int(ymax * scale_y)
                            xmin, xmax = int(xmin * scale_x), int(xmax * scale_x)
                            boxes.append([[xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]])
                            proposal_token = patch_embed[ymin:ymax, xmin:xmax, :].reshape(-1, feature_dim).mean(axis=0)
                            proposal_tokens.append(proposal_token)
                score = 1
                if len(proposal_tokens) > 0:
                    proposal_tokens = np.stack(proposal_tokens)
                    distances = cosine_distances(proposal_tokens, query_cls_token.T)
                    score = distances.min()
                    display_images = False#score < min([0.7]+scores)#0.4
                    if display_images:
                        doc_img = io.imread(os.path.join(images_path, row['img_path']))
                        scale_x = doc_img.shape[1] / mean_attention.shape[1]
                        scale_y = doc_img.shape[0] / mean_attention.shape[0]
                        for box_i, box in enumerate(boxes):
                            xs = np.clip(np.array(box[0]) * scale_x, 0, doc_img.shape[0]).astype(int)
                            ys = np.clip(np.array(box[1]) * scale_y, 0, doc_img.shape[1]).astype(int)
                            color = (0,0,255)
                            bbox_text = f"{int((1 - distances[box_i][0]) * 100)}%"
                            ymin, xmin, ymax, xmax = ys[0], xs[0], ys[2], xs[1]
    
                            doc_img = cv2.rectangle(doc_img,
                                                    (xmin, ymin), (xmax, ymax),
                                                    color, 2)
                            y_mid = ymax-(ymax-ymin)//2
                            x_mid = xmax-(xmax-xmin)//2
                            doc_img = cv2.putText(doc_img, bbox_text, (x_mid, y_mid),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        io.imshow(doc_img)
                        plt.title(f'score: {score}')
                        io.show()
                    scores.append(score)
    scores = np.array(scores)
    idx_sorted = np.argsort(scores)

    gt_list = list(q_row['documents'])
    doc_names = df_docs['doc_id'][idx_sorted]
    doc_names = doc_names.drop_duplicates(keep='first')
    ranking = 1*doc_names.isin(gt_list).values
    
    results_str = q_row['filename'].split('.')[0] +': '+ ' '.join(doc_names.to_list())
    evaluation_results.append(results_str)
    top_k = [fn + '.jpg' for fn in doc_names.to_list()[:10]]
    doc_images = [io.imread(os.path.join(dataset_path, fn)) for fn in top_k]

    AP, precision, recall = compute_average_precision(ranking)
    precision, recall = interpolate_pr(precision, recall)
    display_search_results(query_img, doc_images, title=f'AP: {AP}')
    APs.append(AP)
    precisions.append(precision)

df_queries['AP'] = APs

df_queries.to_parquet(f'num_heads_{num_heads}_feature_dim_{feature_dim}_use_multiscale_patches_{use_multiscale_patches}.parquet')

print(f"mAP: {df_queries['AP'].mean()}")

print(df_queries.groupby(['size_cat', 'ar_cat'])['AP'].mean())
print(df_queries.groupby(['class'])['AP'].mean())


with open('results.txt', 'w') as fp:
    fp.write('\r'.join(evaluation_results))

print(f"mAP: {df_queries['retrieval'].mean()}")

print(df_queries.groupby(['size_cat', 'ar_cat'])['retrieval'].mean())
print(df_queries.groupby(['class'])['retrieval'].mean())


print(f"mAP: {df_queries['mAP'].mean()}")
print(df_queries.groupby(['size_cat', 'ar_cat'])['mAP'].mean())
print(df_queries.groupby(['class'])['mAP'].mean())
