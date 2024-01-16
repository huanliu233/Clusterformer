# coding:utf-8
import numpy as np
from PIL import Image
import cv2

class RandomFlip():
    def __init__(self, prob=0.5):
        super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            image = image[:,:, ::-1]
            label = label[:, ::-1]
        if np.random.rand() < self.prob:
            image = image[:,::-1, :]
            label = label[::-1, :]
        return image, label


class RandomCrop():
    def __init__(self, crop_rate=0.2, prob=0.5):
        super(RandomCrop, self).__init__()
        self.crop_rate = crop_rate
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            # w, h, c = image.shape
            c,w,h = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = np.random.randint(h-h*self.crop_rate, h+1)
            w2 = np.random.randint(w-w*self.crop_rate, w+1)

            image = image[:,w1:w2, h1:h2]
            label = label[w1:w2, h1:h2]

        return image, label


class RandomCropOut():
    def __init__(self, crop_rate=0.1, prob=0.5):
        super(RandomCropOut, self).__init__()
        self.crop_rate = crop_rate
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            # w, h, c = image.shape
            c, w, h = image.shape

            crop_windows_h = np.random.randint(0, h*self.crop_rate)
            crop_windows_w = np.random.randint(0, w*self.crop_rate)

            h1 = np.random.randint(0, h-crop_windows_h-1)
            w1 = np.random.randint(0, w-crop_windows_w-1)

            image[:,w1:w1+crop_windows_w, h1:h1+crop_windows_h] = 0
            label[w1:w1+crop_windows_w, h1:h1+crop_windows_h] = 0

        return image, label


class RandomBrightness():
    def __init__(self, bright_range=0.15, prob=0.5):
        super(RandomBrightness, self).__init__()
        self.bright_range = bright_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            if image.max() < 256:
                bright_factor = np.random.uniform(
                    1-self.bright_range, 1+self.bright_range)
                new_image = (image*bright_factor)
                new_image[new_image > 255] = 255
                new_image[new_image < 0] = 0
            else:
                bright_factor = np.random.uniform(
                    1-self.bright_range, 1+self.bright_range)
                new_image = (image*bright_factor)
            image = new_image.astype(image.dtype)

        return image, label


class RandomNoise():
    def __init__(self, noise_range=10, prob=0.5):
        super(RandomNoise, self).__init__()
        self.noise_range = noise_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape
            clip_min = image.min()
            clip_max = image.max()
            noise = np.random.randint(
                -self.noise_range,
                self.noise_range,
                (w, h, c)
            )

            image = (image + noise).clip(clip_min, clip_max).astype(image.dtype)

        return image, label

class RandomRotate90():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if np.random.random() < self.prob:
            factor = np.random.randint(0, 4)
            img = np.rot90(img, factor,axes=(1,2))
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()


class Rotate():
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if np.random.random() < self.prob:
            angle = np.random.uniform(-self.limit, self.limit)
            height, width = img.shape[1:3]
            mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
            img = cv2.warpAffine(img.transpose((1,2,0)), mat, (height, width),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            img =img.transpose(2,0,1)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (height, width),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)

        return img, mask

class Shift():
    def __init__(self, limit=50, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if np.random.random() < self.prob:
            limit = self.limit
            dx = np.round(np.random.uniform(-limit, limit))
            dy = np.round(np.random.uniform(-limit, limit))

            channel,height, width = img.shape
            y1 = int(limit + 1 + dy)
            y2 = y1 + height
            x1 = int(limit + 1 + dx)
            x2 = x1 + width
            img1 = cv2.copyMakeBorder(img.transpose((1,2,0)), limit+1, limit + 1, limit + 1, limit +1,
                                      borderType=cv2.BORDER_REFLECT_101)
            img = img1[y1:y2, x1:x2, :]
            img = img.transpose((2,0,1))
            if mask is not None:
                mask1 = cv2.copyMakeBorder(mask, limit+1, limit + 1, limit + 1, limit +1,
                                      borderType=cv2.BORDER_REFLECT_101)
                mask = mask1[y1:y2, x1:x2]

        return img, mask
