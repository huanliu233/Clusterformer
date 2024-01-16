import numpy as np
import tifffile as tiff
import os
import cv2
from shutil import copyfile
from tqdm import tqdm
import time
import random
from spectral import *

class Standard():
    def __init__(self):
        self.img_path_map='d:/zhongkexing/sea_ice/data1/image/'
        self.img_path_label = 'd:/zhongkexing/sea_ice/data1/gt/'
        self.drc_path_train_label = 'd:/zhongkexing/sea_ice/data1/ann/train/'
        self.drc_path_train_map = 'd:/zhongkexing/sea_ice/data1/img/train/'
        
        self.drc_path_val_label = 'd:/zhongkexing/sea_ice/data1/ann/val/'
        self.drc_path_val_map = 'd:/zhongkexing/sea_ice/data1/img/val/'
        
        # self.drc_path_test_label = 'd:/zhongkexing/sea_ice/data1/ann/test/'
        # self.drc_path_test_map = 'd:/zhongkexing/sea_ice/data1/img/test/'

        self.map_img_type = '.tif'
        self.ann_img_type = '.png'


    def CopyData(self):
        imgtype_length = len(self.map_img_type)
        listfile = os.listdir(self.img_path_map)
        listfile.sort()
        for path in listfile:
            r = random.random()
            if path[-len(self.map_img_type):]!=self.map_img_type:
                continue
            img_path = path
            label_path = path.replace(self.map_img_type,self.ann_img_type)
            if r<0.25:
                copyfile(self.img_path_label+label_path,
                         self.drc_path_val_label+label_path)
                copyfile(self.img_path_map+img_path,
                         self.drc_path_val_map+img_path)
            else:
                copyfile(self.img_path_label+label_path,
                         self.drc_path_train_label+label_path)
                copyfile(self.img_path_map+img_path,
                         self.drc_path_train_map+img_path)

    def run(self):
        self.CopyData()

if __name__ == '__main__':
	Standard = Standard()
	Standard.run()
