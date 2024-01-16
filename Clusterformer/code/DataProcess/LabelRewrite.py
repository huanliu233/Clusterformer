import cv2
import os
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import shutil

# img = gdal.Open('/home/gyy/data/switzerland/data/v4/train_label_normalize_1000/2010_poly_259_tls.tif')
# image = img.ReadAsArray(0, 0, 1000, 1000)
# image = image*255
# cv2.imwrite('/home/gyy/data/switzerland/code/DataProcess/temp/x.tif',image)

# exit()


ori_label_path = '/home/gyy/data/switzerland/data/v4/train_label_normalize_1000/'
img_seffix = '.tif'
sum = 0
img_list = os.listdir(ori_label_path)
for image_ori_name in img_list:
    if image_ori_name[-len(img_seffix):].upper() != img_seffix.upper():
        continue
    
    img_path = ori_label_path+image_ori_name
    img = gdal.Open(img_path)
    tiffwid = img.RasterXSize
    tiffhei = img.RasterYSize
    image = img.ReadAsArray(0, 0, tiffwid, tiffhei)
    if image.max()<=1:
        continue
    image[image>1]=1
    cv2.imwrite(ori_label_path+image_ori_name,image)