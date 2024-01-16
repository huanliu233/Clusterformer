from pyexpat import model
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise,RandomRotate90,Rotate,Shift


'''dataset parameters'''
train_map = '/comp_hongtu/data/'
train_label = 'd:/seg_nets/data_256/ann/train/'
val_map = 'd:/seg_nets/data_256/img/val/'
val_label = 'd:/seg_nets/data_256/ann/val/'
map_seffix = '.jpg'
label_seffix = '.tif'
in_wh = 256

'''the savepath of models'''
model_dir = 'weights/'
tensorboard_log_dir = 'weights/tensorboard/'
project_name = 'ResUNet_c4'

'''the method of training'''
model_name = 'ResUNet'
model_name = 'UNet'
model_name = 'RTFNet'

loss_name = 'FocalLoss'


continue_checkpoint_model_file_name = ''
continue_checkpoint_optim_file_name = ''

# opt_name = 'torch.optim.SGD(model.parameters(), lr=lr_start, momentum=0.98, weight_decay=0.0003)'
opt_name = 'torch.optim.AdamW(model.parameters(),lr=5e-4,betas=(0.9, 0.999),weight_decay=0.01)'
lr_start = 5e-4
lr_decay = 0.9

batch_size = 3
num_workers = 4
epoch_from = 1
epoch_max = 100
gpu = 0

augmentation_methods = [
    RandomFlip(prob=0.5), 
    # RandomCrop(crop_rate=0.1, prob=0.5), 
    # RandomCropOut(crop_rate=0.06, prob=0.5), 
    RandomBrightness(bright_range=0.1, prob=0.5), 
    # RandomNoise(noise_range=10, prob=0.2), 
    ] # we provide some data augmentation methods, you can choose the appropriate method or do nothing
# augmentation_methods = []


n_class = 2

#v1
train_map = '/home/yinjw/gyy/seaice/img/train/'
train_label = '/home/yinjw/gyy/seaice/ann/train/'
val_map = '/home/yinjw/gyy/seaice/img/test/'
val_label = '/home/yinjw/gyy/seaice/ann/test/'

in_wh = 512
n_class = 2



# augmentation_methods = [
#     RandomFlip(prob=0.5), 
#     # RandomCrop(crop_rate=0.2, prob=0.5), 
#     RandomRotate90(prob=0.5),
#     # RandomCropOut(), 
#     # RandomBrightness(), 
#     # RandomNoise(), 
#     # RandomRotate90(),
#     # Rotate(),
#     # Shift(),
#     ] #
# model_name = 'RTFNet'
# project_name = 'RTFNet_c9'
# model_name = 'UNet'
# project_name = 'Unet_pillar_tree_segmentation'
model_name = 'UNet'
project_name = 'UNet'
train_map = '/home/sbf/liuhuan/tree/data/train/map/'
train_label = '/home/sbf/liuhuan/tree/data/train/label/'
val_map = '/home/sbf/liuhuan/tree/data/test/map/'
val_label = '/home/sbf/liuhuan/tree/data/test/label/'
# model_name = 'RTFNet'
# project_name = 'RTFNet_c9'
epoch_max = 100
opt_name = 'torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(0.9, 0.999),weight_decay=0.001)'
lr_start = 2e-4
lr_decay = 0.9
continue_checkpoint_model_file_name = ''
epoch_from = 1
in_chans = 3
out_chans = 1
loss_name = 'BinaryCrossEntropy'
gpu = 0
augmentation_methods = [
    # 
    # RandomCrop(crop_rate=0.75, prob=0.5), 
    RandomFlip(prob=0.5), 
    # RandomRotate90(prob=0.5),
    # RandomCropOut(), 
    # RandomBrightness(), 
    # RandomNoise(), 
    RandomRotate90(),
    # Rotate(),
    # Shift(),
    ] #
batch_size = 24
num_workers = 1
map_seffix = '.jpg'
label_seffix = '.png'