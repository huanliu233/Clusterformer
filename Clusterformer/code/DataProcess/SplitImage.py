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
ori_file_path = 'd:/switzerland/data/group_dataset/big_val_stdsize/'
drc_img_path = 'd:/switzerland/data/v5/big_val_clip/'
os.makedirs(drc_img_path, exist_ok=True)
stride = 800
standard_row = 1000
standard_col = 1000

image_img_type = '.tif'
result_img_band = 1

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
    
    
    

    # result_datatype = gdal.GDT_Float32
    # result_datatype = gdal.GDT_Byte
    # result_datatype = gdal.GDT_UInt16
    result_tiff_width = standard_row
    result_tiff_height = standard_col
    result_tiff_bands = band_num

    image_data = image_data.ReadAsArray(0, 0, image_wid, image_hei)
    # print(image_data.dtype.name)
    # continue
    if 'int8' in image_data.dtype.name:
        result_datatype = gdal.GDT_Byte
    elif 'int16' in image_data.dtype.name:
        result_datatype = gdal.GDT_UInt16
    else:
        result_datatype = gdal.GDT_UInt32
    # print(image_data.shape)
    # if image_data.shape[1] > image_data.shape[0]:
    # image_data = image_data.transpose(1, 2, 0)

    if band_num == 1:
        image_data = image_data[:, :, np.newaxis]
    
    # print(image_data.shape)
    # print(image_wid,image_hei)

    img_num = 0
    current_hei = 0
    while current_hei < image_hei:
        if image_hei-current_hei<(standard_row/2):
            current_hei = current_hei + stride
            continue
        
        image_pad_hei = image_data[current_hei:min(current_hei+standard_row,image_hei), :, :]

        current_wid = 0
        while current_wid < image_wid:
            if image_wid-current_wid < (standard_col/2):
                current_wid = current_wid + stride
                continue
            image_pad = image_pad_hei[:,current_wid:min(current_wid+standard_col,image_wid), :]
            
            Reflect = ReflectImage(image_pad, standard_row, standard_col)
            image_pad = Reflect.operation()

            save_img_name = img_name[:-len(image_img_type)]
            save_img_name = save_img_name+'_'+str(img_num+1)+'.tif'
            img_num = img_num + 1
            # print(save_img_name,image_pad.shape)
        
            WriteImage(drc_img_path+save_img_name,image_pad,datatype=result_datatype,bands=result_tiff_bands)
            current_wid = current_wid+stride
        current_hei = current_hei + stride
