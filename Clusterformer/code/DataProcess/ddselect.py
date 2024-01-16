import os 
from osgeo import gdal
import numpy as np
import cv2
gt_drc = r'C:\Users\Administrator\Desktop\examples\cankao/'

drc_src = r'C:\Users\Administrator\Desktop\examples\afformer/'

src = r'C:\Users\Administrator\Desktop\pillar_tree_segmentation\code\train_afformer\test_results/'



newfiles = os.listdir(gt_drc)
for file in newfiles:
    gt = gdal.Open(gt_drc+file)
    image_wid = gt.RasterXSize
    image_hei = gt.RasterYSize
    gt = gt.ReadAsArray(0, 0, image_wid, image_hei).astype(float)
    pred = gdal.Open(src+file)
    pred = pred.ReadAsArray(0, 0, image_wid, image_hei).astype(float)
    file_name = drc_src + file
    cv2.imwrite(file_name, pred)