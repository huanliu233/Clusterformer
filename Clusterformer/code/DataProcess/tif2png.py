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

ori_file_path = r"C:\Users\Administrator\Desktop\wenjian/" 
doc_file_path = r"C:\Users\Administrator\Desktop\wenjian/" 



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
    # image_data = image_data.transpose(1, 2, 0)
    image_data = image_data > 0
    image_data = image_data.astype(np.int32)*255
    driver =gdal.GetDriverByName('MEM')
    doc = doc_file_path+img_name
    label_path = doc.replace('tif','png')
    cv2.imwrite(label_path, image_data[0,:,:])
    # dataset = driver.Create(doc, image_wid, image_hei, 1, result_datatype)
    # # for i in range(3):
    # dataset.GetRasterBand(1).WriteArray(image_data[1,:,:])
    # driver =gdal.GetDriverByName('PNG')
    # label_path = doc.replace('tif','png')
    # dst_ds = driver.CreateCopy(label_path,dataset)
print('done')