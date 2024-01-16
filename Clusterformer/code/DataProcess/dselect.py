import os
from osgeo import gdal
from tqdm import tqdm
import numpy as np
from matplotlib.pyplot import close
import cv2
gt_path = r'C:\Users\Administrator\Desktop\pillar_tree_segmentation\groundTruth/'
src = r'C:\Users\Administrator\Desktop\pillar_tree_segmentation\code\train_Clusterformer\test_results_clusterformer/'
gt_drc = r'C:\Users\Administrator\Desktop\examples\gt/'
src_drc = r'C:\Users\Administrator\Desktop\examples\clusterformer/'
map_path = r'C:\Users\Administrator\Desktop\pillar_tree_segmentation\data\test\map/'
new_map_path = r'C:\Users\Administrator\Desktop\examples\map/'

files = os.listdir(gt_path)
# iou = []
iou = []
pos = []
pbar = tqdm(total=len(files))
newfiles = []
for file in files:
    pbar.update(1)
    gt = gdal.Open(gt_path+file)
    image_wid = gt.RasterXSize
    image_hei = gt.RasterYSize
    gt = gt.ReadAsArray(0, 0, image_wid, image_hei).astype(float)
    pred = gdal.Open(src+file)
    pred = pred.ReadAsArray(0, 0, image_wid, image_hei).astype(float)
    if np.sum(gt>0)>0:
        a = np.sum((pred>0)*(gt>0))
        b = np.sum(pred>0)+ np.sum(gt>0) - np.sum((pred>0)*(gt>0))
        iou.append(a/b)
        newfiles.append(file)
pbar = close()
index = np.argsort(-np.array(iou))
newfiles =  np.array(newfiles)
# newfiles = newfiles[index[:500]]
newfiles = list(newfiles)
    # a += np.sum((pred*gt>0)) 
    # b += np.sum(pred>0)+ np.sum(gt>0) - np.sum((pred>0)*(gt>0)) 
for file in newfiles:
    gt = gdal.Open(gt_path+file)
    image_wid = gt.RasterXSize
    image_hei = gt.RasterYSize
    gt = gt.ReadAsArray(0, 0, image_wid, image_hei).astype(float)
    pred = gdal.Open(src+file)
    pred = pred.ReadAsArray(0, 0, image_wid, image_hei).astype(float)
    # map_ = gdal.Open(map_path+file.split('.')[0] + '.jpg')
    # map_ = map_.ReadAsArray(0, 0, image_wid, image_hei).astype(float)
    map_ =cv2.imread(map_path + file.split('.')[0] + '.jpg')
    file_name = gt_drc + file
    cv2.imwrite(file_name, gt)
    file_name = src_drc + file
    cv2.imwrite(file_name, pred)
    file_name = new_map_path + file.split('.')[0] + '.jpg'
    cv2.imwrite(file_name, map_)