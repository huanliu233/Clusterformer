import cv2
import os
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import shutil

#直方图均衡化
class ReflectImage:
    def __init__(self, image, standard_width=1000, standard_height=1000):
        self.image = image
        self.standard_width = standard_width
        self.standard_height = standard_height

    def operation(self):
        # print(self.standard_width-self.image.shape[0])
        # print(self.standard_height-self.image.shape[1])
        # print(self.image.shape)
        if len(self.image.shape)==3:
            image_new = np.pad(self.image, ((0,self.standard_width-self.image.shape[0]), (0,self.standard_height-self.image.shape[1]),(0,0)) , 'reflect')
        if len(self.image.shape)==2:
            image_new = np.pad(self.image, ((
                0, self.standard_width-self.image.shape[0]), (0, self.standard_height-self.image.shape[1])), 'reflect')
        return image_new

# ori_img_path = '/home/gyy/data/switzerland/data/v4/train_map_normalize_1000/'  # 源文件夹
# ori_label_path = '/home/gyy/data/switzerland/data/v4/train_label_normalize_1000/'

# reflect_list_txt = '/home/gyy/data/switzerland/data/v4/reflect_list.txt'

# img_seffix = '.tif'
# standard_height = 1000
# standard_width = 1000
# sum = 5
# n = 0
# img_list = os.listdir(ori_img_path)
# for image_ori_name in img_list:
#     if image_ori_name[-len(img_seffix):].upper() != img_seffix.upper():
#         continue
    
#     img_path = ori_img_path+image_ori_name
#     img = gdal.Open(img_path)
#     tiffwid = img.RasterXSize
#     tiffhei = img.RasterYSize
#     if tiffhei == standard_height and tiffwid == standard_width:
#         continue
#     image = img.ReadAsArray(0, 0, tiffwid, tiffhei)

#     reflect = ReflectImage(image)
#     image_new = reflect.operation()
#     cv2.imwrite(ori_img_path+image_ori_name,image_new)
    
#     img_path = ori_label_path+image_ori_name
#     img = gdal.Open(img_path)
#     image = img.ReadAsArray(0, 0, tiffwid, tiffhei)
#     reflect = ReflectImage(image)
#     image_new = reflect.operation()*255
    
#     cv2.imwrite(ori_label_path+image_ori_name,image_new)
#     with open(reflect_list_txt, 'a') as appender:
#         appender.write(image_ori_name+'/')

# reflect_list_txt.close()

# img = gdal.Open('/home/gyy/data/switzerland/data/v4/train_label_normalize_1000/2019_1174_222_reflect.tif')
# tiffwid = img.RasterXSize
# tiffhei = img.RasterYSize
# image = img.ReadAsArray(0, 0, tiffwid, tiffhei)
# image = image*255
# cv2.imwrite('/home/gyy/data/switzerland/data/v4/train_label_normalize_1000/2019_1174_222_reflect_1.tif',image)
