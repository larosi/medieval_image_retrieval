# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:53:16 2024

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
import os


def load_model(backbone_size='small', use_registers=False):
    backbone_archs = {"small": "vits14",
                      "base": "vitb14",
                      "large": "vitl14",
                      "giant": "vitg14"}

    backbone_arch = backbone_archs[backbone_size]
    reg = '_reg' if use_registers else ''
    backbone_name = f"dinov2_{backbone_arch}{reg}"
    try:
        model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    except RuntimeError as e:
        if 'Cannot find callable' in e.message:
            model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name, force_reload=True)
        raise
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


def tokens_from_windows(windows_tokens, h_new, w_new, grid_shape, windows_size, stride):
    overlap_y = stride / windows_size[0]
    overlap_x = stride / windows_size[1] 
    num_rows = (h_new - windows_size[0]) // stride + 1
    num_cols = (w_new - windows_size[1]) // stride + 1
    
    y_step = int(grid_shape[0]*overlap_y)
    x_step = int(grid_shape[1]*overlap_x)
    
    new_grid_shape = (int((h_new // windows_size[0]) * grid_shape[0] ),
                      int((w_new // windows_size[1]) * grid_shape[1]),
                      windows_tokens[0].shape[-1])
    fmap_shape = (grid_shape[0],
                  grid_shape[1],
                  windows_tokens[0].shape[-1])
    reconstructed_tokens = np.zeros(new_grid_shape, dtype=windows_tokens[0].dtype)
    window_count = 0

    for y in range(0, num_rows):
        for x in range(0, num_cols):
            tokens = windows_tokens[window_count].reshape(fmap_shape)
            current_token = reconstructed_tokens[y*y_step:y*y_step+grid_shape[0], x*x_step:x*x_step+grid_shape[1], :]
            avg_token = (current_token + tokens) /2
            reconstructed_tokens[y*y_step:y*y_step+grid_shape[0], x*x_step:x*x_step+grid_shape[1], :] = avg_token
            window_count += 1
            #io.imshow(tokens[:,:,0:3])
            #io.show()
            #io.imshow(reconstructed_tokens[:,:,0:3])
            #io.show()
    return reconstructed_tokens

def sliding_window(image, windows_size, stride):
    height, width = image.shape[:2]
    num_rows = (height - windows_size[0]) // stride + 1
    num_cols = (width - windows_size[1]) // stride + 1
    patches = []
    for y in range(0, num_rows * stride, stride):
        for x in range(0, num_cols * stride, stride):
            patch = image[y:y+windows_size[0], x:x+windows_size[1],:]
            patches.append(patch)
    return np.array(patches)

def add_padding(img_rgb, windows_size):
    h, w = img_rgb.shape[0:2]
    pad_h = int(windows_size[0] * np.ceil(h / windows_size[0])) - h
    pad_w = int(windows_size[1] * np.ceil(w / windows_size[1])) - w
    padding_width = ((0, pad_h), (0, pad_w), (0, 0))
    constant_value = 0 #(0.75811552, 0.7067092 , 0.61939638)
    img_rgb = np.pad(img_rgb, pad_width=padding_width,
                   mode='constant', 
                   constant_values=constant_value)
    return img_rgb

def save_features(dataset_filename, image_filename, features):
    with h5py.File(dataset_filename, 'a') as h5f:
        if image_filename in h5f:
            del h5f[image_filename]
        dataset_group = h5f.create_group(image_filename)
        dataset_group.create_dataset('features', data=features)


if __name__ == '__main__':
    images_dir = os.path.join('..', 'data', 'images')
    df_path = os.path.join('..', 'data', 'df_dataset.parquet')
    hdf5_path = os.path.join('..', 'data', 'docs_patch_embed.hdf5')
    df = pd.read_parquet(df_path)

    ratio = (2, 2)
    windows_size = (224*ratio[0], 224*ratio[1])
    grid_shape = (windows_size[0]//14, windows_size[1]//14)
    stride = windows_size[0]
    model = load_model(backbone_size='small', use_registers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    trans = transforms.Compose([transforms.Resize(windows_size[0]),
                                transforms.CenterCrop(windows_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_rgb = io.imread(os.path.join(images_dir, row['img_path']))
        h_no_pad, w_no_pad = img_rgb.shape[:2]
        img_rgb = add_padding(img_rgb, windows_size)

        h_new, w_new = img_rgb.shape[:2]

        h_remove = (h_new - h_no_pad) // 14
        w_remove = (w_new - w_no_pad) // 14

        windows = sliding_window(img_rgb, windows_size, stride)
        windows_tokens = []

        with torch.no_grad():
            for window_i in range(0, windows.shape[0]):
                img = Image.fromarray(windows[window_i])
                img = trans(img)
                img = torch.unsqueeze(img, dim=0).to(device)

                features = model.forward_features(img)
                patch_tokens = torch.squeeze(features['x_norm_patchtokens']).cpu().detach().numpy()
                windows_tokens.append(patch_tokens)
        reconstructed_tokens = tokens_from_windows(windows_tokens,
                                                   h_new, w_new,
                                                   grid_shape,
                                                   windows_size,
                                                   stride)
        tokens_h, tokens_w = reconstructed_tokens.shape[0:2]
        reconstructed_tokens = reconstructed_tokens[:tokens_h-h_remove, :tokens_w-w_remove, :]
        save_features(hdf5_path, row['img_path'], reconstructed_tokens)
