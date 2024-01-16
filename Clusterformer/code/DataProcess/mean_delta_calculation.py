#!/bin/env python3
import os
import numpy as np
import cv2
import time
from osgeo import gdal
from tqdm import tqdm
from PIL import Image
# from WriteTif import WriteImage
from Util import ReflectImage,WriteImage

####等待切割的图片
# ori_file_path = 'd:/switzerland/data/2019_LK25_gray/'
# drc_img_path = 'd:/switzerland/data/v2/map/train/'
# ori_file_path = 'd:/switzerland/data/group_dataset/big_val_stdsize/'
# drc_img_path = 'd:/switzerland/data/v5/big_val_clip/'
# ori_file_path = '/home/yinjw/p2p/data/ori_data/高光谱SAR/柏林/data/sar_image/'
# drc_img_path = '/home/yinjw/p2p/data/proc_daa/柏林/sar_image/'

ori_file_path = r"C:\Users\Administrator\Desktop\pillar_tree_segmentation\data\all/" 



image_img_type = '.jpg'
# result_img_band = 1
MeanR = 0
MeanG = 0
MeanB = 0

no_of_pixels = 0
listfile = os.listdir(ori_file_path)
listfile.sort()
for img_name in listfile:
    if img_name[-len(image_img_type):] != image_img_type:
        continue
    image_img_path = ori_file_path + img_name
    image_data = gdal.Open(image_img_path)
    image_wid = image_data.RasterXSize
    image_hei = image_data.RasterYSize
    image_data = image_data.ReadAsArray(0, 0, image_wid, image_hei)
    image_data = image_data.astype(np.float32)
    image_data = image_data.transpose(1, 2, 0)
    no_of_pixels = no_of_pixels + image_wid*image_hei
    MeanR += np.sum(image_data[:,:,0])
    MeanG += np.sum(image_data[:,:,1])
    MeanB += np.sum(image_data[:,:,2])

MeanR = MeanR/no_of_pixels
MeanG = MeanG/no_of_pixels
MeanB = MeanB/no_of_pixels

sigmaR = 0
sigmaG = 0
sigmaB = 0
for img_name in listfile:
    if img_name[-len(image_img_type):] != image_img_type:
        continue
    image_img_path = ori_file_path + img_name
    image_data = gdal.Open(image_img_path)
    image_wid = image_data.RasterXSize
    image_hei = image_data.RasterYSize
    image_data = image_data.ReadAsArray(0, 0, image_wid, image_hei)
    image_data = image_data.astype(np.float32)
    image_data = image_data.transpose(1, 2, 0)

    sigmaR += np.sum((image_data[:,:,0] - MeanR)**2)
    sigmaG += np.sum( (image_data[:,:,1] - MeanG)**2)
    sigmaB += np.sum( (image_data[:,:,2] - MeanB)**2)

sigmaR = np.sqrt(sigmaR/no_of_pixels)
sigmaG = np.sqrt(sigmaG/no_of_pixels)
sigmaB = np.sqrt(sigmaB/no_of_pixels)

with open('mean_sigma.txt', 'a+') as inputfile:
    inputfile.write("MeanR: %02d, MeanG: %04f, MeanB: %04f,\
                    sigmaR: %04f, sigmaG: %04f,sigmaB: %04f'\n'"
                    %(MeanR,MeanG,MeanB,sigmaR,sigmaG,sigmaB))
print('done')