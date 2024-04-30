# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 09:34:02 2024

@author: Mico
"""

import os
import numpy as np
from skimage import io
from skimage.transform import resize
from tqdm import tqdm
import pandas as pd

def load_txt(txt_path):
    with open(txt_path, 'r') as fp:
        data = fp.read()
    return data

def create_queries_df(queries_folder, gt_path):
    gt = load_txt(gt_path)
    gt = gt.split('\n')
    query2docs = {}
    for line in gt:
        if line !='':
            query_fn, query_gt = line.split(':')
            query_gt = query_gt.split('\t')[1:]
            query2docs[query_fn] = query_gt
    
    df_queries = {'img_path': [], 'class': [], 'query_id': [], 'documents': []}
    for class_crop in os.listdir(queries_folder):
        class_dir = os.path.join(queries_folder, class_crop)
        for crop_fn in os.listdir(class_dir):
            query_id = crop_fn.split('.')[0]
            df_queries['img_path'].append(os.path.join(class_crop, crop_fn)) 
            df_queries['class'].append(class_crop)
            df_queries['query_id'].append(query_id)
            df_queries['documents'].append(query2docs[query_id])
            
    df_queries = pd.DataFrame.from_dict(df_queries)
    return df_queries

def merge_df_queries(queries_folder, gt_path):
    df_queries2 = create_queries_df(queries_folder, gt_path)
    df_queries = pd.read_pickle(os.path.join('..', 'data','queries_df.pkl'))
    df_queries2.sort_values('query_id', inplace=True)
    df_queries2.reset_index(drop=True, inplace=True)
    df_queries.sort_values('filename', inplace=True)
    df_queries.reset_index(drop=True, inplace=True)
    
    df_queries['documents'] = df_queries2['documents']
    df_queries['img_path'] = df_queries2['img_path']
    return df_queries

dataset_path = r'D:\datasets\medieval_images\DocExplore_images'
queries_folder = r'D:\datasets\medieval_images\evaluation_kit_v2\DocExplore_queries_web' 
gt_path = r'D:\datasets\medieval_images\evaluation_kit_v2\im_example.txt'
df_queries = merge_df_queries(queries_folder, gt_path)
df_queries.to_parquet(os.path.join('..', 'data', 'df_queries.parquet'))

create_queries_df(queries_folder, gt_path)

image_filenames = os.listdir(dataset_path)

df_documents = {'img_path': [], 'height': [], 'width': [],
                'rgb_mean': [], 'rgb_std': []}

for img_filename in tqdm(image_filenames):
    img_path = os.path.join(dataset_path, img_filename)
    img = io.imread(img_path)
    h, w, ch = img.shape

    img_reshaped = img.reshape(-1, ch) / 255.0
    rgb_mean = list(np.round(img_reshaped.mean(axis=0), 3))
    rgb_std = list(np.round(img_reshaped.std(axis=0), 3))

    df_documents['img_path'].append(img_filename)
    df_documents['height'].append(h)
    df_documents['width'].append(w)
    df_documents['rgb_mean'].append(rgb_mean)
    df_documents['rgb_std'].append(rgb_std)
df_documents = pd.DataFrame(df_documents)

df_documents.to_parquet(os.path.join('..', 'data', 'df_docs.parquet'))
