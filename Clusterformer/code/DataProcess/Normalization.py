import cv2
import os
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import shutil

#直方图均衡化
class EqualizeHist:
    def __init__(self, image, bins = 65536, normalize_max = 255, normalize_type = 'uint8'):
        self.image = image
        self.bins = image.max()+1
        self.normalize_max = normalize_max
        self.normalize_type = normalize_type

    def get_histogram(self,image):
        # array with size of bins, set to zeros
        histogram = np.zeros(self.bins)
        # loop through pixels and sum up counts of pixels
        for pixel in image:
            histogram[pixel] += 1
        # return our final result
        return histogram

    def cumsum(self,a):
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

    def operation(self):
        flat = self.image.flatten()
        hist = self.get_histogram(flat)
        # execute the fn
        cs = self.cumsum(hist)
        # numerator & denomenator
        nj = (cs - cs.min()) * self.normalize_max
        N = cs.max() - cs.min()
        # re-normalize the cdf
        cs = nj / N
        cs = cs.astype(self.normalize_type)
        image_new = cs[flat]
        image_new = np.reshape(image_new, self.image.shape)
        return image_new

#百分比截断
class TruncatedLinearStretch:
    def __init__(self, image, truncated_value=2, max_out = 255, min_out = 0, normalize_type = 'uint8'):
        self.image = image
        self.truncated_value = truncated_value
        self.max_out = max_out
        self.min_out = min_out
        self.normalize_type = normalize_type

    def operation(self):
        truncated_down = np.percentile(image, self.truncated_value)
        truncated_up = np.percentile(image, 100 - self.truncated_value)
        image_new = (self.image - truncated_down) / (truncated_up - truncated_down) * (self.max_out - self.min_out) + self.min_out
        image_new[image_new < self.min_out] = self.min_out
        image_new[image_new > self.max_out] = self.max_out
        image_new = image_new.astype(self.normalize_type)
        return image_new

##标准差拉伸
class StandardDeviation:
    def __init__(self, image, parameter=2, max_out = 255, min_out = 0, normalize_type = 'uint8'):
        self.image = image
        self.parameter = parameter
        self.max_out = max_out
        self.min_out = min_out
        self.normalize_type = normalize_type

    def operation(self):
        Mean = np.mean(self.image)
        StdDev = np.std(self.image, ddof=1)
        ucMax = Mean + self.parameter * StdDev
        ucMin = Mean - self.parameter * StdDev
        k = (self.max_out - self.min_out) / (ucMax - ucMin)
        b = (ucMax * self.min_out - ucMin * self.max_out) / (ucMax - ucMin)
        if (ucMin <= 0):
            ucMin = 0

        image_new = np.select([self.image==self.min_out, self.image<=ucMin, self.image>=ucMax,  k*self.image+b < self.min_out, k*self.image+b > self.max_out,
                             (k*self.image+b > self.min_out) & (k*self.image+b < self.max_out)],
                            [self.min_out, self.min_out, self.max_out, self.min_out, self.max_out, k * self.image + b], self.image)
        image_new = image_new.astype(self.normalize_type)
        return image_new

class ReflectImage:
    def __init__(self, image, standard_width=1000, standard_height=1000):
        self.image = image
        self.standard_width = standard_width
        self.standard_height = standard_height

    def operation(self):
        image_new = np.pad(self.image, ((0,self.standard_width-self.image.shape[0]), (0,self.standard_height-self.image.shape[1])) , 'reflect')
        return image_new

class WriteImage:
    def __init__(self,  drc_path,image):
        self.image = image
        self.drc_path = drc_path

    def operation(self):
        result_tiff_datatype= gdal.GDT_Byte#设置输出影像的参数-数据类型
        result_tiff_width = self.image.shape[1]  # 设置输出影像的参数-图片长
        result_tiff_height=self.image.shape[0]
        result_tiff_bands=1
        driver=gdal.GetDriverByName("GTiff")   #开辟输出的tiff的空间
        output_dataset = driver.Create(
        self.drc_path, result_tiff_width, result_tiff_height, result_tiff_bands, result_tiff_datatype)
        output_dataset.GetRasterBand(1).WriteArray(self.image)
        del output_dataset
        return image_new

ori_img_path = 'd:/switzerland/data/v5/val/map_ori/'  # 源文件夹
drc_img_path = 'd:/switzerland/data/v5/val/map/'  # 目标文件夹
ori_label_path = 'd:/switzerland/data/2018_CHM_LK25_3/'
drc_label_path = 'd:/switzerland/data/2018_CHM_LK25_3_new/'

os.makedirs(drc_img_path,exist_ok=True)
os.makedirs(drc_label_path,exist_ok=True)

img_seffix = '.tif'

img_list = os.listdir(ori_img_path)
for image_ori_name in img_list:
    if image_ori_name[-len(img_seffix):].lower() != img_seffix.lower():
        continue
    
    # if os.path.isfile(ori_label_path+image_ori_name)==False:
    #     continue
    
    # if image_ori_name[:4]=='1980':
    #     continue
    
    # label_ori_name = image_ori_name
    img_path = ori_img_path+image_ori_name
    # label_path  = ori_label_path + image_ori_name
    print(img_path)
    
    img = gdal.Open(img_path)
    tiffwid = img.RasterXSize
    tiffhei = img.RasterYSize
    # if tiffwid == 1000:
    #     continue
    image = img.ReadAsArray(0, 0, tiffwid, tiffhei)
    image = image.astype('uint16')

    ###eh
    eh = EqualizeHist(image)
    image_new = eh.operation()
    image_name = image_ori_name#.replace(img_seffix.lower(),'_eh.tif')
    # label_name = label_ori_name.replace(img_seffix.lower(),'_eh.tif')
    
    # reflect = ReflectImage(image_new)
    # image_new = reflect.operation()
    # label = gdal.Open(label_path)
    # label = label.ReadAsArray(0, 0, tiffwid, tiffhei)
    # reflect = ReflectImage(label)
    # label = reflect.operation()
    
    eh_map_write = WriteImage(drc_img_path+image_name,image_new)
    eh_map_write.operation()
    # eh_label_write = WriteImage(drc_label_path+label_name,label)
    # eh_label_write.operation()
    
    
    ####tls
    # tls = TruncatedLinearStretch(image,truncated_value = 2)
    # image_new = tls.operation()
    # image_name = image_ori_name.replace(img_seffix.lower(),'_tls.tif')
    # label_name = label_ori_name.replace(img_seffix.lower(),'_tls.tif')
    # reflect = ReflectImage(image_new)
    # image_new = reflect.operation()
    # label = gdal.Open(label_path)
    # label = label.ReadAsArray(0, 0, tiffwid, tiffhei)
    
    # tls_map_write = WriteImage(drc_img_path+image_name,image_new)
    # tls_map_write.operation()
    # tls_label_write = WriteImage(drc_label_path+label_name,label)
    # tls_label_write.operation()
    
    ###sd
    # sd = StandardDeviation(image)
    # image_new = sd.operation()
    # # reflect = ReflectImage(image)
    # image_name = image_ori_name.replace(img_seffix.lower(),'_sd.tif')
    # label_name = label_ori_name.replace(img_seffix.lower(),'_sd.tif')
    # reflect = ReflectImage(image_new)
    # image_new = reflect.operation()
    # label = gdal.Open(label_path)
    # label = label.ReadAsArray(0, 0, tiffwid, tiffhei)
    
    # sd_map_write = WriteImage(drc_img_path+image_name,image_new)
    # sd_map_write.operation()
    # sd_label_write = WriteImage(drc_label_path+label_name,label)
    # sd_label_write.operation()
    # break

# img = gdal.Open('/home/gyy/data/switzerland/data/v4/train_label_normalize_1000/2019_1174_222_eh.tif')
# tiffwid = img.RasterXSize
# tiffhei = img.RasterYSize
# image = img.ReadAsArray(0, 0, tiffwid, tiffhei)
# image = image*255
# cv2.imwrite('/home/gyy/data/switzerland/data/v4/train_label_normalize_1000/2019_1174_222_eh_1.tif',image)
