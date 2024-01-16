import numpy as np
import os
import cv2
from shutil import copyfile
from tqdm import tqdm
import time
import random
# from spectral import *

class Standard():
    def __init__(self):
        self.img_path_map='C:/Users/Administrator/Desktop/gyy/data/map/'
        self.img_path_label = 'C:/Users/Administrator/Desktop/gyy/data/label/'
        self.drc_path_train_label = 'C:/Users/Administrator/Desktop/gyy/data/train/map/'
        self.drc_path_train_map = 'C:/Users/Administrator/Desktop/gyy/data/train/label/'
        
        self.drc_path_val_label = 'C:/Users/Administrator/Desktop/gyy/data/test/map/'
        self.drc_path_val_map = 'C:/Users/Administrator/Desktop/gyy/data/test/label/'
        
        # self.drc_path_test_label = 'd:/zhongkexing/sea_ice/data1/ann/test/'
        # self.drc_path_test_map = 'd:/zhongkexing/sea_ice/data1/img/test/'

        self.map_img_type = '.tif'
        self.ann_img_type = '.tif'


    def CopyData(self):
        imgtype_length = len(self.map_img_type)
        listfile = os.listdir(self.img_path_map)
        listfile.sort()

        for path in listfile:
            r = random.random()
            if path[-len(self.map_img_type):]!=self.map_img_type:
                continue
            img_path = path
            label_path = path
            
            if r<0.3:
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
