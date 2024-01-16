import cv2
import os
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import shutil

src_path = 'd:/switzerland/data/ori/1980s_Ortho_gray_2018_1m/'
drc_path = 'd:/switzerland/data/normalization/1980_ortho/'
os.makedirs(drc_path,exist_ok=True)
img_seffix = '.tif'
img_list = os.listdir(src_path)
drc_datatype = gdal.GDT_Byte
for image_ori_name in img_list:
    if image_ori_name[-len(img_seffix):].upper() != img_seffix.upper():
        continue

    img_path = src_path+image_ori_name
    img = gdal.Open(img_path)
    tiffwid = img.RasterXSize
    tiffhei = img.RasterYSize
    band_num = img.RasterCount
    image = img.ReadAsArray(0, 0, tiffwid, tiffhei)
    image[image==65535]=255
        # print(image_ori_name)
    
    result_tiff_datatype = drc_datatype
    result_tiff_width = tiffwid
    result_tiff_height = tiffhei
    result_tiff_bands = band_num
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(drc_path+image_ori_name, result_tiff_width,result_tiff_height, result_tiff_bands, result_tiff_datatype)
    for b in range(result_tiff_bands):
        new_band = image
        dataset.GetRasterBand(b+1).WriteArray(new_band)
    del dataset
    
