# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:33:17 2024

@author: Mico
"""


import pandas as pd
from PIL import Image
import numpy as np
import torch
import h5py
from torchvision import transforms
from tqdm import tqdm 
from skimage import io
from skimage.transform import resize
import os

from extract_patch_embeddings import load_model

def pad_query(query):
    if query.dtype == np.dtype('uint8'):
        query = query /255.0
    h, w = query.shape[:2]
    long_side = max([h, w])
    pad_h = ((long_side-h) // 2)
    pad_w = ((long_side-w) // 2)
    pad_h += int(long_side*0.1)
    pad_w += int(long_side*0.1)
    padding_width = ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
    constant_value = 0.6947403666666667 #(0.75811552, 0.7067092 , 0.61939638)
    query = np.pad(query, pad_width=padding_width, mode='constant', constant_values=constant_value)
    return query

def to_uint8(query):
    query = (255*query).astype(np.uint8)
    return query

queries_folder = r'D:\datasets\medieval_images\evaluation_kit_v2\DocExplore_queries_web' 
df_path = os.path.join('..', 'data', 'df_queries.parquet')
out_features_path = os.path.join('..', 'data', 'queries_cls_tokens.npy')
df = pd.read_parquet(df_path)

model = load_model(backbone_size='base', use_registers=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

trans = transforms.Compose([transforms.Resize(224),
                            transforms.CenterCrop((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)])

trans_big = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

features = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    query = io.imread(os.path.join(queries_folder, row['img_path']))
    is_big = True #row['size_cat'] != 'small'
    if is_big:
        query = resize(query, (((query.shape[0])//14 + 1)*14, ((query.shape[1])//14 + 1)*14))
    else:    
        upscale_factor = 1
        query = resize(query, (((query.shape[0]*upscale_factor)//14 + 1)*14, ((query.shape[1]*upscale_factor)//14 + 1)*14))
        #query = pad_query(query)
    #io.imshow(query)
    #io.show()
    query = to_uint8(query)

    img = Image.fromarray(query)
    img = trans_big(img)
    """
    if is_big:
        img = trans_big(img)
    else:
        img = trans(img)
    """
    img = torch.unsqueeze(img, dim=0).to(device)

    #cls_token = model(img).cpu().detach().numpy()
    #['x_prenorm'][:,1+4:,:]
    patch_tokens = torch.squeeze(model.forward_features(img)['x_norm_patchtokens']).cpu().detach().numpy()
    cls_token = patch_tokens.mean(axis=0)
    features.append(cls_token)
features = np.vstack(features)
np.save(out_features_path, features)
