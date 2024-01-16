# coding:utf-8
import numpy as np
import chainer
from PIL import Image
import torch
# from ipdb import set_trace as st


def resize_img(img_data, resize_img_hei, resize_img_wid):
    if len(img_data.shape) > 2:
        if img_data.shape[2] > img_data.shape[0]:
            band_num = img_data.shape[0]
        else:
            band_num = img_data.shape[2]
        if img_data.shape[2] > img_data.shape[0]:
            import cv2
            import numpy as np
            new_img_data = np.zeros([band_num, resize_img_hei, resize_img_wid])
            for b in range(band_num):
                new_band = img_data[b]
                new_band = cv2.resize(
                    new_band, (resize_img_hei, resize_img_wid))
                new_img_data[b] = new_band
        else:
            import cv2
            import numpy as np
            new_img_data = np.zeros([resize_img_hei, resize_img_wid, band_num])
            for b in range(band_num):
                new_band = img_data[:, :, b]
                new_band = cv2.resize(
                    new_band, (resize_img_hei, resize_img_wid))
                new_img_data[:, :, b] = new_band
    else:
        band_num = 1
        import cv2
        new_img_data = cv2.resize(img_data, (resize_img_hei, resize_img_wid))
    return new_img_data

def f_score(precision, recall, beta=1):
    """calcuate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(num_classes, logits, labels):
    logits = logits.argmax(0)
    intersect = logits[logits == labels]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        logits.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        labels.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label 

def intersect_and_union1(num_classes, logits, labels):
    logits = logits>0
    intersect = logits[logits == labels]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        logits.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        labels.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label

def total_intersect_and_union(num_classes,logits,labels):
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)

def calculate_accuracy(logits, labels):
    # inputs should be torch.tensor
    predictions = logits.argmax(1)
    no_count = (labels==-1).sum()
    count = ((predictions==labels)*(labels!=-1)).sum()
    acc = count.float() / (labels.numel()-no_count).float()
    return acc
def calculate_accuracy1(logits, labels):
    # inputs should be torch.tensor
    predictions = logits>0
    no_count = (labels==-1).sum()
    count = ((predictions==labels)*(labels!=-1)).sum()
    acc = count.float() / (labels.numel()-no_count).float()
    return acc

def calculate_result(cf):
    n_class = cf.shape[0]
    conf = np.zeros((n_class,n_class))
    IoU = np.zeros(n_class)
    conf[:,0] = cf[:,0]/cf[:,0].sum()
    for cid in range(0,n_class):
        if cf[:,cid].sum() > 0:
            conf[:,cid] = cf[:,cid]/cf[:,cid].sum()
            IoU[cid]  = cf[cid,cid]/(cf[cid,0:].sum()+cf[0:,cid].sum()-cf[cid,cid])
    overall_acc = np.diag(cf[0:,0:]).sum()/cf[0:,:].sum()
    acc = np.diag(conf)

    return overall_acc, acc, IoU



# for visualization
def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette


def visualize(names, predictions):
    palette = get_palette()

    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(1, int(predictions.max())):
            img[pred == cid] = palette[cid]

        img = Image.fromarray(np.uint8(img))
        img.save(names[i].replace('.png', '_pred.png'))
