import os
import shutil

base_path = 'd:/switzerland/data/v4/map_train_9_1/'
# os.makedirs('d:/switzerland/data/v2/map/train_500bmp/',exist_ok=True)
file_source = base_path + '/'
label_seffix = '.tif'
file_list = os.listdir(file_source)

for file_name in  file_list:
    if file_name[(-len(label_seffix)):].upper() != label_seffix.upper():
        continue
    print(file_name)
    file_src_path = file_source+file_name
    # file_new_name = file_name.split('_')[1]+label_seffix
    file_new_name = file_name.replace('.tif', '_1.tif')

    print(file_new_name)
    
    file_dst_path = 'd:/switzerland/data/v4/map_train_9_1new/'+file_new_name
    # file_dst_path = file_dst_path.replace('.tif','.jpg')
    shutil.copyfile(file_src_path, file_dst_path)
