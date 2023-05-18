import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageNet_all(Dataset):
    def __init__(self, args, partition='train', data_aug = True,transform=None):
        super(Dataset, self).__init__()

        partition_list = ['train','val','test']
        self.data_aug = data_aug

        if transform is None:
            if partition == 'train' and self.data_aug:
                image_size = 84
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                         np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                         np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
                ])
        else:
            self.transform = transform

        data, label = [], []
        lab = 0
        for partition in partition_list:
            ROOT_PATH = os.path.join(args.data_root, 'miniImageNet_BDC', partition)
            self.label_set = os.listdir(ROOT_PATH)
            for cls_id in self.label_set:
                cls_imgs = os.listdir(os.path.join(ROOT_PATH, cls_id))
                for file in cls_imgs:
                    data.append(os.path.join(ROOT_PATH, cls_id, file))
                # data.extend(cls_imgs)
                label.extend([lab] * len(cls_imgs))
                lab += 1
        self.data = data
        self.label = label
        self.num_classes = len(set(label))


    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 5
    args.n_queries = 12
    args.data_root = '../../data'
    args.data_aug = True
    args.n_episodes = 5
    args.n_aug_support_samples = 5
    imagenet = ImageNet_all(args, 'train')
    print(len(imagenet))
    print(imagenet.__getitem__(500))

