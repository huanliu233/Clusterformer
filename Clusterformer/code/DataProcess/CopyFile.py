#!/bin/env python3
from distutils.file_util import copy_file
import os
from shutil import copyfile
from tqdm import *


def CopyFile(ori_path,drc_path,ori_list,drc_list):
    pbar = tqdm(total=len(ori_list))
    for i in range(len(ori_list)):
        ori = ori_path + ori_list[i]
        drc = drc_path + drc_list[i]
        copyfile(ori,drc)
        pbar.update(1)
    pbar.close()

ori_file_path = 'd:/switzerland/data/v5/train/label/'
drc_file_path = 'd:/switzerland/data/v5/train/label_new/'
os.makedirs(drc_file_path,exist_ok=True)
img_type = '.tif'

ori_list = []
drc_list = []
list = os.listdir(ori_file_path)
for name in list:
    if name[-len(img_type):] != img_type:
        continue
    if 're' in name:
        ori_list.append(name)
        new_name = name.replace('re','')
        drc_list.append(new_name)
    else:
        ori_list.append(name)
        drc_list.append(name)
CopyFile(ori_file_path,drc_file_path,ori_list,drc_list)

# ori_file_path = 'd:/switzerland/data/v5/big_train_clip/'
# drc_file_path = 'd:/switzerland/data/v5/train/label_ori/'
# os.makedirs(drc_file_path,exist_ok=True)
# img_type = '.tif'

# ori_list = []
# drc_list = []
# list = os.listdir(ori_file_path)
# for name in list:
#     if name[-len(img_type):] != img_type:
#         continue
#     if '3m' in name:
#         ori_list.append(name)
#         new_name = name.replace('_3m','')
#         drc_list.append(new_name)
# CopyFile(ori_file_path,drc_file_path,ori_list,drc_list)

# ori_file_path = 'd:/switzerland/data/v5/big_train_clip/'
# drc_file_path = 'd:/switzerland/data/v5/train/label_ori/'
# os.makedirs(drc_file_path, exist_ok=True)
# img_type = '.tif'

# ori_file_path = 'd:/switzerland/data/v5/big_val_clip/'
# drc_file_path = 'd:/switzerland/data/v5/val/map_ori/'
# os.makedirs(drc_file_path,exist_ok=True)
# img_type = '.tif'
# ori_list = []
# drc_list = []
# list = os.listdir(ori_file_path)
# for name in list:
#     if name[-len(img_type):] != img_type:
#         continue
#     if 'gray' in name:
#         ori_list.append(name)
#         new_name = name.replace('_gray', '')
#         drc_list.append(new_name)
# CopyFile(ori_file_path, drc_file_path, ori_list, drc_list)

# ori_file_path = 'd:/switzerland/data/v5/train/label_ori/'
# drc_file_path = 'd:/switzerland/data/v5/train/label/'
# os.makedirs(drc_file_path, exist_ok=True)
# img_type = '.tif'
# ori_list = []
# drc_list = []
# list = os.listdir(ori_file_path)
# for name in list:
#     if name[-len(img_type):] != img_type:
#         continue
#     ori_list.append(name)
#     new_name = name.replace('.tif', '_tls.tif')
#     drc_list.append(new_name)
# CopyFile(ori_file_path, drc_file_path, ori_list, drc_list)

