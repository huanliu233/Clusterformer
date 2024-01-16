gpu = 0  # the testing gpu number of your computer

window_shape = 512 # the width and height of training image

# model_name = 'ResUNet' # the training models of this project, there are 3 popular segmentation provided in this project
# model_name = 'UNet'
# model_name = 'RTFNet'

# model_dir = 'd:/seg_nets/SegModels//weights/ResUNet_c13.pth' # the model parameters path of testing model

# src_dir = 'd:/seg_nets/testdata/' # the path of testing images
# drc_dir = 'd:/seg_nets/test_result/' # the direction path of testing result

'''example of testing setting'''
class_num = 1
src_dir = r'data\ori_test\map/'
drc_dir = r'code\train_Unet\results/'
model_dir = r'code\train_Unet\weights\UNet\UNet_epo48_tiou0.7813_viou0.7666.pth'
model_name = 'UNet'
map_seffix = '.jpg'
batch_size = 16
label_band = 1
# src_dir = '/home/gyy/datasets/comp_hongtu/data/行车场景分割/'
# drc_dir = '/home/gyy/datasets/comp_hongtu/test_result/行车场景分割/'
