# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from osgeo import gdal
import cv2
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import resize_img
import random

# from augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
# from util import resize_img

class My_dataset(Dataset):

    def __init__(self, map_dir, map_seffix, label_dir='', label_seffix='',  have_label=True, class_num=2, input_h=500, input_w=500, transform=[]):
        super(My_dataset, self).__init__()

        map_set = []
        dem_set = []
        label_set = []
        maptype_length = len(map_seffix)
        listfile = os.listdir(map_dir)
        for path in listfile:
            if path[(-maptype_length):].upper() != map_seffix.upper():
                continue
            map_set.append(map_dir + path)
            path = path.replace(map_seffix,label_seffix)
            label_set.append(label_dir + path)

        self.map_set = map_set
        self.label_set = label_set
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.class_num = class_num
        self.is_train = have_label
        self.map_seffix = map_seffix
        self.label_seffix = label_seffix
        self.n_data = len(self.map_set)
        self.mean = np.array([108, 123.44, 95.69]).reshape((3,1,1))
        self.std = np.array([55.11, 49.32, 33.81]).reshape((3,1,1)) 
        # self.mean = np.array([85.479114, 103.568216, 78.568216]).reshape((3,1,1))
        # self.std = np.array([38.567626, 32.79, 28.8434]).reshape((3,1,1)) 
        # self.mean = np.array([79.69, 99.88, 75.35]).reshape((3,1,1))
        # self.std = np.array([32.41, 28.63, 20.82]).reshape((3,1,1)) 
    def read_image(self, name, folder):
        if folder == 'images':
            image = gdal.Open(name)
            image_wid = image.RasterXSize
            image_hei = image.RasterYSize
            image = image.ReadAsArray(0, 0, image_wid, image_hei)
            image = image.astype(np.float32)
            image = (image - self.mean)/self.std
            # print(image.shape)
        else:
            image = gdal.Open(name)
            image_wid = image.RasterXSize
            image_hei = image.RasterYSize
            image = image.ReadAsArray(0, 0, image_wid, image_hei)
       
            # image = image.astype(np.uint8)
            # image.flags.writeable = True
        return image

    def get_train_item(self, index):
        map_name = self.map_set[index]
        name = map_name.split('/')[-1]
        label_name = self.label_set[index]
        label_name = label_name.replace(self.map_seffix,self.label_seffix)
        image = self.read_image(map_name, 'images')
        label = self.read_image(label_name, 'labels')
        
        prob = random.random()
        if prob<0.5:
            for func in self.transform:
                image, label = func(image, label)

        image = resize_img(image, self.input_h, self.input_w)
        label = resize_img(label, self.input_h, self.input_w)

        image = np.array(image, dtype="float32")
        label = np.array(label, dtype="int64")

        return torch.tensor(image), torch.tensor(label), name

    def get_test_item(self, index):
        name = self.map_set[index]
        image = self.read_image(name, 'images')
        # image = resize_img(image, self.input_h, self.input_w)

        return torch.tensor(image), name

    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)

    def __len__(self):
        return self.n_data


if __name__ == '__main__':
    train_map = '/home/gyy/datasets/comp_hongtu/data/v1/map/train/'
    train_label = '/home/gyy/datasets/comp_hongtu/data/v1/label/train/'
    x = My_dataset(map_dir=train_map, map_seffix='.tif', label_dir=train_label, label_seffix='.png', have_label=True)
    image,label,name = x.get_train_item(1)
    print(image.shape)
    print(label.shape)
    print(label)
