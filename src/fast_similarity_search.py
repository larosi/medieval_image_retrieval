# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:24:35 2024

@author: Mico
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage import io
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from sklearn.decomposition import PCA


def get_path_embed(hdf5_path, image_filename):
    with h5py.File(hdf5_path, 'r') as h5f:
        patch_embed = h5f[f'{image_filename}/features'][()]
    return patch_embed


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
    
def min_max_scale(data):
    min_v = data.min()
    max_v = data.max()
    range_v = max_v - min_v
    if range_v != 0:
        data = (data - min_v) / range_v
    return data

dataset_path = r'D:\datasets\medieval_images\DocExplore_images'
images_path = os.path.join('..', 'data', 'images')
queries_folder = r'D:\datasets\medieval_images\evaluation_kit_v2\DocExplore_queries_web' 

df_queries_path = os.path.join('..', 'data', 'df_queries.parquet')
queries_features_path = os.path.join('..', 'data', 'queries_cls_tokens.npy')
df_path = os.path.join('..', 'data', 'df_dataset.parquet')
#hdf5_path = os.path.join('..', 'data', 'docs_patch_embed_base.hdf5')
hdf5_path = os.path.join('..', 'data', 'docs_patch_embed_base.hdf5')

df_queries = pd.read_parquet(df_queries_path)
df_docs = pd.read_parquet(df_path)



# dino small
backbone_size = 'base'

if backbone_size == 'small':
    num_heads = 3
    feature_dim = 384

if backbone_size == 'base':
    num_heads = 12
    feature_dim = 768


num_patch_pools = 3
use_heads = True
num_heads = num_heads if use_heads else 1
head_dim = feature_dim // num_heads
show_results = False
distance_type = 'cos' # 'cos' 'l2' 'prod'

df_queries.reset_index(inplace=True, names='query_id')
df_queries = df_queries.set_index('query_id')

df_queries.sort_index(inplace=True)
evaluation_results = []
eps = 1e-10

#kernel_type = 'square' #['horizontal', 'square', 'vertical', 'horizontal']
for kernel_type in ['horizontal', 'vertical', 'square']:
    cls_tokens = np.load(queries_features_path)
    if kernel_type == 'square':
        kernel = np.ones((3, 3, 1)) / 9
        stride = (1, 1)
    elif kernel_type == 'vertical':
        kernel = np.ones((3, 1, 1)) / 3
        stride = (1, 1)
    elif kernel_type =='horizontal':
        kernel = np.ones((1, 3, 1)) / 3
        stride = (1, 1)

    scale_levels = []
    doc_filenames = []
    kernel_orientation = []
    all_multiscale_patch_embeds = []


    for index, row in tqdm(df_docs.iterrows(), total=df_docs.shape[0]):
        patch_embed = get_path_embed(hdf5_path, row['img_path'])
        multiscale_patch_embeds = []
        if show_results:
            pca = PCA(n_components=4)
            pca.fit(patch_embed.reshape(-1, feature_dim))
            doc_img = io.imread(os.path.join(dataset_path, row['img_path'])) 
            io.imshow(doc_img)
            io.show()
        for i in range(1, num_patch_pools):
            if min(patch_embed.shape) > 3:
                if i > 0:
                    patch_embed = convolve(patch_embed, kernel)[::stride[0], ::stride[1],:]
            else:
                break
            if show_results:
       
                patches_rgb = pca.transform(patch_embed.reshape(-1, feature_dim)).reshape(patch_embed.shape[0:2] + (4,))
                patches_rgb = min_max_scale(patches_rgb)[:,:,1:4]
                io.imshow(patches_rgb)
                io.show()
                print(patches_rgb.shape)
            multiscale_patch_embeds.append(patch_embed.reshape(-1, feature_dim))
            num_embedings = multiscale_patch_embeds[-1].shape[0]
            scale_levels += [i] * num_embedings
            doc_filenames += [row['img_path']] * num_embedings
        multiscale_patch_embeds = np.vstack(multiscale_patch_embeds)
        all_multiscale_patch_embeds.append(multiscale_patch_embeds)
    all_multiscale_patch_embeds = np.vstack(all_multiscale_patch_embeds)

    df_multiscale = pd.DataFrame()
    df_multiscale['img_path'] = doc_filenames
    df_multiscale['scale_level'] = scale_levels
    df_multiscale.reset_index(drop=False, inplace=True, names='patch_id')

    path2doc_id = dict(df_docs[['img_path', 'doc_id']].values)
    df_multiscale['doc_id'] = df_multiscale['img_path'].apply(lambda x: path2doc_id[x])
    


    if distance_type == 'cos':
        for head_i in range(0, num_heads):
            start = head_i*head_dim
            end = (head_i+1)*head_dim
            cls_tokens[:,start:end] = cls_tokens[:,start:end] / np.linalg.norm(cls_tokens[:,start:end] , axis=-1).reshape(-1, 1)
            all_multiscale_patch_embeds[:,start:end]  = all_multiscale_patch_embeds[:,start:end] / np.linalg.norm(all_multiscale_patch_embeds[:,start:end], axis=-1).reshape(-1, 1)
        num_distances = all_multiscale_patch_embeds.shape[0]

    #df_queries = df_queries[df_queries['ar_cat'] != 'square']
    #df_queries = df_queries[df_queries['size_cat'] != 'big']
    #df_queries = df_queries[df_queries['class'] == 'triple_sep']
    evaluation_results = []
    for q_idx, q_row in tqdm(df_queries.iterrows(), total=df_queries.shape[0]):
        distances = np.zeros((num_distances, ))
        for head_i in range(0, num_heads):
            start = head_i*head_dim
            end = (head_i+1)*head_dim
            if distance_type == 'cos':
                distances += (1 - np.squeeze(np.matmul(all_multiscale_patch_embeds[:, start:end], cls_tokens[:, start:end][q_idx:q_idx+1,:].T))) / num_heads
            elif distance_type == 'prod':
                distances += (-np.squeeze(np.matmul(all_multiscale_patch_embeds[:, start:end], cls_tokens[:, start:end][q_idx:q_idx+1,:].T))) / num_heads
            elif distance_type == 'l2':
                distances += (np.linalg.norm(all_multiscale_patch_embeds[:, start:end] - cls_tokens[:, start:end][q_idx:q_idx+1,:], axis=1)) / num_heads

        idx_sorted = np.argsort(np.squeeze(distances))
    
        doc_names = df_multiscale['doc_id'][idx_sorted]
        doc_names = doc_names.drop_duplicates(keep='first')
    
        results_str = q_row['filename'].split('.')[0] +': '+ ' '.join(doc_names.to_list())
        evaluation_results.append(results_str)
        if show_results:
            n_to_show = 15
            top_k = [fn + '.jpg' for fn in doc_names.to_list()[:n_to_show]]
    
            query_img = io.imread(os.path.join(queries_folder, q_row['img_path']))
            doc_images = [io.imread(os.path.join(dataset_path, fn)) for fn in top_k]
            display_search_results(query_img, doc_images)
    
    with open(f'results_query_upscaled_no_pool_kernel_{kernel_type}_stride_{stride[0]}-{{stride[1]}}_{backbone_size}_meantokens_heads_{num_heads}_dist_{distance_type}_no_padding.txt', 'w') as fp:
        fp.write('\r'.join(evaluation_results))
    
    """
    print(f"mAP: {df_queries['mAP'].mean()}")
    
    print(df_queries.groupby(['size_cat', 'ar_cat'])['mAP'].mean())
    print(df_queries.groupby(['class'])['mAP'].mean())
    """
