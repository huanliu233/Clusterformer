# coding:utf-8
import os
import argparse
import time
import numpy as np
import cv2
from matplotlib.pyplot import close
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import model.TransUNet_main.networks.vit_seg_configs as get_config
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.my_dataset import My_dataset
from DataProcess.Util import ReflectImage,WriteImage


from test_parameters import src_dir,drc_dir
from test_parameters import map_seffix,label_band,class_num
from test_parameters import window_shape
from test_parameters import model_dir,model_name,gpu
from test_parameters import batch_size
import model
from osgeo import gdal
from DataProcess.SplitImage_mod import caiqie

def merge(probs,name,sizes=[6336,9504]):
    os.makedirs(drc_dir, exist_ok=True)
    stride = 500
    standard_row = 512
    standard_col = 512
    image_hei = sizes[0]
    image_wid = sizes[1]
    label_data = np.zeros(sizes)
    label3d = np.zeros((sizes[0],sizes[1],3))
    sign = np.zeros(sizes)
    img_num = 0
    current_hei = 0
    while current_hei < image_hei:
        if image_hei-current_hei<standard_row:
            current_hei = image_hei- standard_row

        # print(image_split.shape)
        current_wid = 0
        while current_wid < image_wid:
            if image_wid-current_wid < standard_col:
                current_wid = image_wid- standard_col

            label_data[current_hei:current_hei+standard_row,current_wid:current_wid+standard_col] \
                  = label_data[current_hei:current_hei+standard_row,current_wid:current_wid+standard_col] + probs[img_num,:,:] 
            sign[current_hei:current_hei+standard_row,current_wid:current_wid+standard_col] += np.ones((standard_row,standard_col))
            img_num = img_num + 1

            if current_wid == image_wid- standard_col:
                break
            else:
                current_wid = current_wid+stride
        if current_hei == image_hei- standard_row:
            break
        else:
            current_hei = current_hei + stride
    label =  (label_data/sign>=0.5)*255 
    label3d[:,:,0] = label
    file_name = drc_dir + name + '.tif'
    result_datatype = gdal.GDT_Byte
    result_tiff_bands = 3
    WriteImage(file_name,label3d,datatype=result_datatype,bands=result_tiff_bands)
    print('done')

def main():

    # cf = np.zeros((class_num, class_num))
    model_name_all = 'model.' + model_name
    model = eval(model_name_all)(in_chans=3,n_class=1)

    if gpu >= 0:
        model.cuda(gpu)
    print('| loading model file %s... ' % model_name, end='')

    model.load_state_dict(torch.load(model_dir, map_location='cpu'))
    print('done!')

    # image_img_type = '.JPG'
    image_img_type = '.tif'
    caiqie(src_dir,image_img_type = image_img_type)

    
    listfile = os.listdir(src_dir)
    listfile.sort()
    for img_name in listfile:
        if img_name[-len(image_img_type):] != image_img_type:
            continue
        single_src_dir = src_dir + img_name[:-len(image_img_type)] + '/'
        image_img_path = src_dir + img_name
        image_data = gdal.Open(image_img_path)
        image_wid = image_data.RasterXSize
        image_hei = image_data.RasterYSize
        test_dataset = My_dataset(map_dir=single_src_dir, map_seffix=map_seffix, class_num=class_num, have_label=False, input_h=window_shape, input_w=window_shape)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=False
        )
        test_loader.n_iter = len(test_loader)
        model.eval()
        pbar = tqdm(total=len(test_loader))
        total = torch.Tensor().cuda()
        with torch.no_grad():
            for it, (images, names) in enumerate(test_loader):
                pbar.update(1)
                images = Variable(images)
                if gpu >= 0:
                    images = images.cuda(gpu)
                    images = images.float()
                
                logits = model(images)

                # for i in range(logits.shape[0]):
                #     it_logit = logits[i]

                pred_result = F.sigmoid(logits)
                total = torch.cat((total,pred_result),dim=0)
            merge(total.squeeze(1).cpu().numpy(),img_name[:-len(image_img_type)],sizes=[image_hei,image_wid])

        pbar = close()


if __name__ == '__main__':
    model_dir_path = model_dir

    # checkpoint_model_file = os.path.join(model_dir_path, 'tmp.pth')

    os.makedirs(drc_dir + '/', exist_ok=True)

    main()


