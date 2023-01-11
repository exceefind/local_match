import os
import random
import numpy as np
import csv
import shutil

csv_path = "data/miniimagenet/test.csv"
IMAGE_PATH = 'data/miniimagenet/images'
lines = [x.strip() for x in open(csv_path,'r').readlines()]
total_cls_num = 20
cls_img_num = 600
select_classes_num = 10
select_image_num = 50
select_classes = np.sort(np.random.choice(total_cls_num,select_classes_num,False))
select_idx = []
# print(select_classes)
for i in select_classes:
    select_image = np.sort(np.random.choice(cls_img_num,select_image_num,False))
    for img_idx in select_image:
        select_idx.append(img_idx)

# print(len(select_idx))
# print(select_idx)
csv_save = 'miniimagenet_fg/test_fg.csv'
img_path_new = 'miniimagenet_fg/images'
csv_save_old = 'miniimagenet_fg/split/test.csv'
if not os.path.isdir(img_path_new):
    os.mkdir(img_path_new)

select_dict  = []
select_name_old = []
with open(csv_save_old,'r') as f:
    for i, line in enumerate(f.readlines()):
        name, lab_id = line.strip().split(',')

        select_name_old.append(name)
        if lab_id not in select_dict:
            # print(line)
            select_dict.append(lab_id)
p = 0
q = 0
print(select_dict)
with open(csv_save,'w', newline='') as f:
    writer = csv.writer(f)
    for i,line in enumerate(lines):
        name, lab_id = line.split(',')
        if i == 0:
            row = line.split(',')
            writer.writerow(row)
        else:
            flag = False
            # print(lab_id in select_dict)
            # print(lab_id)
            if  lab_id in select_dict and i%cls_img_num == select_idx[p] :
                row = line.split(',')
                writer.writerow(row)
                file_name = line.split(',')[0]
                img_path = os.path.join(IMAGE_PATH, file_name)
                shutil.copy(img_path,os.path.join(img_path_new,file_name))
                p += 1
                flag = True
                if p%select_image_num == 0:
                    q += 1
                    p = q * select_image_num
            elif name in select_name_old and flag is False:
                row = line.split(',')
                writer.writerow(row)
                file_name = line.split(',')[0]
                img_path = os.path.join(IMAGE_PATH, file_name)
                shutil.copy(img_path, os.path.join(img_path_new, file_name))
                p += 1
                if p % select_image_num == 0:
                    q += 1
                    p = q * select_image_num
            if p >= len(select_idx):
                break



