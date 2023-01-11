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
select_image_num = 20
select_classes = np.sort(np.random.choice(total_cls_num,select_classes_num,False))
select_idx = []
# print(select_classes)
for i in select_classes:
    select_image = np.sort(np.random.choice(cls_img_num,select_image_num,False))
    for img_idx in select_image:
        select_idx.append(img_idx + i * cls_img_num)

# print(len(select_idx))
# print(select_idx)
csv_save = 'miniimagenet_fg/test_fg.csv'
img_path_new = 'miniimagenet_fg/images'
if not os.path.isdir(img_path_new):
    os.mkdir(img_path_new)

p = 0
with open(csv_save,'w', newline='') as f:
    writer = csv.writer(f)
    for i,line in enumerate(lines):

        if i == 0:
            row = line.split(',')
            writer.writerow(row)
        else:
            if p<len(select_idx) and (i-1) == select_idx[p]:
                row = line.split(',')
                writer.writerow(row)
                file_name = line.split(',')[0]
                img_path = os.path.join(IMAGE_PATH, file_name)
                shutil.copy(img_path,os.path.join(img_path_new,file_name))
                p += 1



