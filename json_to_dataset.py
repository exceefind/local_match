import os,csv,time
from PIL import Image
import numpy as np

if __name__ == '__main__':
    # root_dir = 'miniimagenet_fg/mask_label'
    # files = os.listdir(root_dir)
    # for file in files:
    #     # print(file.split('.')[1]=='json')
    #     if file.split('.')[1] == 'json':
    #         os.system('labelme_json_to_dataset '+ os.path.join(root_dir,file))

    path = 'miniimagenet_fg/mask_label/n0198127600000228_json'
    IMG = Image.open(os.path.join(path,'label.png')).convert('RGB')
    Img_arr = np.array(IMG)[:,:,0]