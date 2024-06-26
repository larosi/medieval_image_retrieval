# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:32:21 2024

@author: Mico
"""

import os
import numpy as np
from skimage import io
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import medfilt
from skimage.measure import block_reduce
from scipy.ndimage import binary_closing
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

def sliding_window(image, patch_size, stride):
    height, width = image.shape[:2]
    num_rows = (height - patch_size[0]) // stride + 1
    num_cols = (width - patch_size[1]) // stride + 1
    patches = []
    for y in range(0, num_rows * stride, stride):
        for x in range(0, num_cols * stride, stride):
            patch = image[y:y+patch_size[0], x:x+patch_size[1]]
            patches.append(patch)
    return np.array(patches)

def crop_borders(img, percentaje=0.02):
    h, w = img.shape[0:2]
    cropy = int(h - 2*percentaje*h)
    cropx = int(w - 2*percentaje*w)
    img_crop = center_crop(img, cropx, cropy)
    return img_crop

def center_crop(img, cropx, cropy):
    y, x = img.shape[0:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]

dataset_path = r'D:\datasets\medieval_images\DocExplore_images'
df = pd.read_parquet(os.path.join('..', 'data', 'df_docs.parquet'))

df['area'] = df['height'] * df['width']
df['aspect'] = df['height']/df['width']

df['outlier'] = np.logical_or(df['aspect'] > np.percentile(df['aspect'], 0.8)*2,
                              df['area'] < np.percentile(df['area'], 0.3))

df = df[~df['outlier']].reset_index(drop=True)

sns.scatterplot(data=df, y='height', x='width')
plt.show()


sns.scatterplot(df, y='area', x='aspect')
plt.show()


df['orientation'] = np.where(df['aspect'] > 1,
                             'portait', 'landscape')

df['size'] = np.where(np.sqrt(df['area']) >= 2*np.percentile(np.sqrt(df['area']), 0.8),
                      'large', 'small')

plt.hist(np.sqrt(df[df['size'] == 'small']['area']))
plt.show()

plt.hist(np.sqrt(df[df['size'] == 'large']['area']))
plt.show()

plt.hist(df[df['size'] == 'small']['aspect'])
plt.show()

sns.scatterplot(data=df[df['orientation'] == 'portait'], y='height', x='width')
plt.show()

new_data = {'xmin': [], 'xmax': [], 'ymin': [], 'ymax': [], 'laplacian_var': []}

from skimage.filters import laplace
display_images = False
display_crop = False
crop_borders_percentaje = 0.02
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    img_path = os.path.join(dataset_path, row['img_path'])
    img = io.imread(img_path)
    img = crop_borders(img, percentaje=crop_borders_percentaje)
    img_laplacian = rgb2gray(img)
    img_laplacian = resize(img_laplacian, output_shape=(img.shape[0]/2, img.shape[1]/2))
    
    img_laplacian = np.abs(laplace(img_laplacian))
    if display_images:
        io.imshow(img)
        io.show()
        io.imshow(img_laplacian)
        io.show()
    
    #img_laplacian = img_laplacian.max(axis=-1)
    for _ in range(0, 4):
        img_laplacian = block_reduce(img_laplacian, (2, 2), np.max)
    img_laplacian = medfilt(img_laplacian)
    th = threshold_otsu(img_laplacian)
    mask = img_laplacian > min(th, 0.3)
    mask = binary_closing(mask)
    if display_images:
        io.imshow(mask)
        io.show()
        io.imshow(img_laplacian)
        io.show()
    ys, xs = np.where(mask)
    scale_x = img.shape[0] / mask.shape[0] 
    scale_y = img.shape[0] / mask.shape[0]
    h, w = img.shape[0:2]
    ymin = max(0, int(min(ys) * scale_y - h*0.05))
    ymax = min(h, int(max(ys) * scale_y + h*0.1))

    xmin = max(0, int(min(xs) * scale_x - w*0.05))
    xmax = min(w, int(max(xs) * scale_x + w*0.1))

    img_crop = img[ymin:ymax, xmin:xmax, :]
    laplacian_var = laplace(img_crop).var()
    if display_crop:
        io.imshow(img)
        io.show()
        io.imshow(img_crop)
        io.show()
    for col in new_data.keys():
        new_data[col].append(locals()[col])

for col in new_data.keys():
    df[col] = new_data[col]

df['laplacian_var'].hist()
plt.show()
assert -1 == 0
empty_images_th = np.percentile(df['laplacian_var'], 7)
for index, row in tqdm(df[df['laplacian_var'] < empty_images_th].iterrows(), total=df.shape[0]):
    img_path = os.path.join(dataset_path, row['img_path'])
    img = io.imread(img_path)
    img = crop_borders(img, percentaje=0.02)
    io.imshow(img)
    plt.title(f"{row['laplacian_var']}")
    io.show()

df = df[df['laplacian_var'] >= empty_images_th]


output_folder = os.path.join('..', 'data','images')

df['doc_id'] = df['img_path'].str.split('.').str[0]
new_rows = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    img_path = os.path.join(dataset_path, row['img_path'])
    img = io.imread(img_path)
    img = crop_borders(img, percentaje=crop_borders_percentaje)
    xmin, xmax, ymin, ymax = row[['xmin', 'xmax', 'ymin', 'ymax']]
    img = img[ymin:ymax, xmin:xmax, :]
    h, w = img.shape[0:2]

    if row['size'] == 'small':
        w_new = 224*2
        h_new = int((14*(h*w_new / w)//14))
        img = resize(img, (h_new, w_new), order=3, preserve_range=True).astype(np.uint8)
        out_path = os.path.join(output_folder, row['img_path'])
        io.imsave(out_path, img)
        new_rows.append(row)
    else:
        img = crop_borders(img, percentaje=0.02)
        patch_size = (w//3, w//3)
        stride = int(0.5*patch_size[0])
        windows = sliding_window(img, patch_size, stride)
        for i in range(0, windows.shape[0]):
            img_win = windows[i, :, :, :]
            h, w = img_win.shape[0:2]
            w_new = 224*2
            h_new = int((14*(h*w_new / w)//14))
            img_win = resize(img_win, (h_new, w_new), order=3, preserve_range=True).astype(np.uint8)

            out_fn = f"{row['doc_id']}-{str(i).zfill(3)}.jpg"
            out_path = os.path.join(output_folder, out_fn)
            io.imsave(out_path, img_win)
            row_copy = row.copy()
            row_copy['img_path'] = out_fn
            new_rows.append(row_copy)
df = pd.DataFrame(new_rows).reset_index(drop=True)
df.to_parquet(os.path.join('..', 'data', 'df_dataset.parquet'))
"""
fg_images = os.listdir(output_folder)
df = df[df['img_path'].isin(fg_images)]
df.reset_index(drop=True, inplace=True)
"""