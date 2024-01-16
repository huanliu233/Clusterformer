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

ori_file_path = r"C:\Users\Administrator\Desktop\pillar_tree_segmentation\data\train\map/" 
doc_file_path = r"C:\Users\Administrator\Desktop\pillar_tree_segmentation\data\train\map2/" 

standard_row = 512
standard_col = 512

image_img_type = '.tif'
# result_img_band = 1

listfile = os.listdir(ori_file_path)
listfile.sort()
for img_name in listfile:
    if img_name[-len(image_img_type):] != image_img_type:
        continue
    image_img_path = ori_file_path + img_name
    image_data = gdal.Open(image_img_path)

    image_wid = image_data.RasterXSize
    image_hei = image_data.RasterYSize
    
    band_num = image_data.RasterCount

    image_data = image_data.ReadAsArray(0, 0, image_wid, image_hei)
    # print(image_data.dtype.name)
    # continue
    result_datatype = gdal.GDT_Byte

    # print(image_data.shape)
    # if image_data.shape[1] > image_data.shape[0]:
    image_data = image_data.transpose(1, 2, 0)
    image_data = image_data.astype(np.uint8)
    driver =gdal.GetDriverByName('MEM')
    doc = doc_file_path+img_name
    dataset = driver.Create(doc, image_wid, image_hei, 3, result_datatype)
    for b in range(3):
        new_band = image_data[:, :, b]
        dataset.GetRasterBand(b+1).WriteArray(new_band)

    driver =gdal.GetDriverByName('JPEG')
    label_path = doc.replace('tif','jpg')
    dst_ds = driver.CreateCopy(label_path,dataset)
print('done')