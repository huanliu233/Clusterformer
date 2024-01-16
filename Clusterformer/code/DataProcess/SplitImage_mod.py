#!/bin/env python3
import os
import numpy as np
import cv2
import time
from osgeo import gdal
from tqdm import tqdm
from PIL import Image
# from WriteTif import WriteImage
from .Util import ReflectImage,WriteImage

####等待切割的图片
# ori_file_path = 'd:/switzerland/data/2019_LK25_gray/'
# drc_img_path = 'd:/switzerland/data/v2/map/train/'
# ori_file_path = 'd:/switzerland/data/group_dataset/big_val_stdsize/'
# drc_img_path = 'd:/switzerland/data/v5/big_val_clip/'
# ori_file_path = '/home/yinjw/p2p/data/ori_data/高光谱SAR/柏林/data/sar_image/'
# drc_img_path = '/home/yinjw/p2p/data/proc_daa/柏林/sar_image/'

# ori_file_path = r"C:\Users\Administrator\Desktop\unet\data\ori_test\map/" 
# drc_img_path = r"C:\Users\Administrator\Desktop\unet\data\test\map/"
def caiqie(ori_file_path,image_img_type='.JPG'):
    # os.makedirs(drc_img_path, exist_ok=True)
    
    stride = 500
    standard_row = 512
    standard_col = 512

    # result_img_band = 1

    listfile = os.listdir(ori_file_path)
    listfile.sort()
    pbar = tqdm(total=len(listfile))
    for img_name in listfile:
        if img_name[-len(image_img_type):] != image_img_type:
            continue
        pbar.update(1)
        image_img_path = ori_file_path + img_name
        image_data = gdal.Open(image_img_path)
        image_wid = image_data.RasterXSize
        image_hei = image_data.RasterYSize
        
        band_num = image_data.RasterCount
        

        os.makedirs(ori_file_path +img_name[:-len(image_img_type)] + '/',  exist_ok=True)
        savepath = ori_file_path +img_name[:-len(image_img_type)] + '/'
        # result_datatype = gdal.GDT_Float32
        # result_datatype = gdal.GDT_Byte
        # result_datatype = gdal.GDT_UInt16
        result_tiff_width = standard_row
        result_tiff_height = standard_col
        result_tiff_bands = band_num

        image_data = image_data.ReadAsArray(0, 0, image_wid, image_hei)
        if band_num>3:
            image_data = image_data[:3,:,:]
            result_tiff_bands = band_num - 1
        # print(image_data.dtype.name)
        # continue
        if 'int8' in image_data.dtype.name:
            result_datatype = gdal.GDT_Byte
        elif 'int16' in image_data.dtype.name:
            result_datatype = gdal.GDT_UInt16
        else:
            result_datatype = gdal.GDT_Float32
        # print(image_data.shape)
        # if image_data.shape[1] > image_data.shape[0]:
        image_data = image_data.transpose(1, 2, 0)

        if band_num == 1:
            image_data = image_data[:, :, np.newaxis]
        

        img_num = 0
        current_hei = 0
        while current_hei < image_hei:
            if image_hei-current_hei<standard_row:
                current_hei = image_hei- standard_row

            image_split_row = image_data[current_hei:current_hei+standard_row, :, :]
            # print(image_split.shape)
            current_wid = 0
            while current_wid < image_wid:
                if image_wid-current_wid < standard_col:
                    current_wid = image_wid- standard_col

                image_split_col = image_split_row[:,current_wid:current_wid+standard_col, :]
                # print(image_split.shape)
        
                save_img_name = savepath + str(img_num+1).zfill(4)+'.jpg'
                img_num = img_num + 1
                # print(save_img_name,image_pad.shape)
                # print(image_split.shape)
                
                WriteImage(save_img_name,image_split_col,datatype=result_datatype,bands=result_tiff_bands)
                if current_wid == image_wid- standard_col:
                    break
                else:
                    current_wid = current_wid+stride
            if current_hei == image_hei- standard_row:
                break
            else:
                current_hei = current_hei + stride

    print('done')

