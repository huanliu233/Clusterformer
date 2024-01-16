from ctypes import resize
import os
import numpy as np
import cv2
from osgeo import gdal
from Util import ResizeImage,WriteImage

ori_file_path = 'd:/switzerland/data/group_dataset/big_val_stdsize//'  # 源文件夹
drc_file_path = 'd:/switzerland/data/group_dataset/big_val_stdsize//'  # 目标文件夹

img_type = '_3m.tif'
os.makedirs(drc_file_path, exist_ok=True)

listfile = os.listdir(ori_file_path)  # 图片的名称的列表
listfile.sort()
for img_name in listfile:  # 读名称列表
    if img_name[-len(img_type):] != img_type:
        continue  # 筛掉不是.tif结尾的影像
    img_path = ori_file_path + img_name
    img = gdal.Open(img_path)  # 读影像
    tiffwid = img.RasterXSize
    tiffhei = img.RasterYSize
    band_num = img.RasterCount
    result_img_band = band_num
    image = img.ReadAsArray(0, 0, tiffwid, tiffhei)
    gt = img.GetGeoTransform()
    proj = img.GetProjection()
    # print(image.shape)

    map_path = img_path.replace('_3m', '_gray')
    mapimg  =gdal.Open(map_path)
    tw = mapimg.RasterXSize
    th = mapimg.RasterYSize
    if tw!=tiffwid or th!=tiffhei:
        print(map_path)
        
        new_image  = ResizeImage(image,tw,th)
        
        WriteImage(img_path.replace('_3m','_3mre'),new_image,gt = gt,proj = proj)
        # exit()
    # exit()
