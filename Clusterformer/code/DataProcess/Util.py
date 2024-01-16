from osgeo import gdal
import numpy as np
import cv2


def WriteImage(drc_path, image, datatype=gdal.GDT_Byte, bands=1, proj='', gt='', compress=False):
    width = image.shape[1]  # 设置输出影像的参数-图片长
    height = image.shape[0]
    driver = gdal.GetDriverByName("GTiff")  # 开辟输出的tiff的空间
    if compress:
        dataset = driver.Create(drc_path, width, height, bands, datatype, options=[
            "TILED=YES", "COMPRESS=LZW"])
    else:
        dataset = driver.Create(drc_path, width, height, bands, datatype)
    if bands == 1 and len(image.shape) == 2:
        dataset.GetRasterBand(1).WriteArray(image)
    else:
        for b in range(bands):
            new_band = image[:, :, b]
            dataset.GetRasterBand(b+1).WriteArray(new_band)
    if gt != '':
        dataset.SetGeoTransform(gt)
    if proj != '':
        dataset.SetProjection(proj)
    del dataset
    # return image_new


class ReflectImage:
    def __init__(self, image, standard_width=1000, standard_height=1000):
        self.image = image
        self.standard_width = standard_width
        self.standard_height = standard_height

    def operation(self):
        if len(self.image.shape) == 3:
            image_new = np.pad(self.image, ((
                0, self.standard_width-self.image.shape[0]), (0, self.standard_height-self.image.shape[1]), (0, 0)), 'reflect')
        if len(self.image.shape) == 2:
            image_new = np.pad(self.image, ((
                0, self.standard_width-self.image.shape[0]), (0, self.standard_height-self.image.shape[1])), 'reflect')
        return image_new


def ResizeImage(image, shape_0, shape_1):
    print(image.shape)
    if len(image.shape)==2:
        new_img_data = cv2.resize(
            image, (shape_0, shape_1), interpolation=cv2.INTER_NEAREST)
    else:
        if 'int8' in image.dtype.name:
            result_datatype = 'uint8'
        elif 'int16' in image.dtype.name:
            result_datatype = 'uint16'
        else:
            result_datatype = 'uint32'
        new_img_data = np.zeros((shape_0,shape_1,image.shape[2]), dtype=result_datatype)
        for i in range(image.shape[2]):
            image = image[:,:,i]
            image = cv2.resize(image, (shape_0, shape_1),
                               interpolation=cv2.INTER_NEAREST)
            new_img_data[i] = image
    return new_img_data
