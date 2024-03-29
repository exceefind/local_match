import json
import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CUB(Dataset):
    def __init__(self, args, partition='train', data_aug = True,transform=None):
        super(Dataset, self).__init__()
        # IMAGE_PATH = os.path.join(args.data_root, 'CUB/CUB_200_2011/images')
        SPLIT_PATH = os.path.join(args.data_root, 'CUB')
        json_path = os.path.join(SPLIT_PATH, partition + '.json')
        self.partition = partition
        self.data_aug = data_aug
        with open(json_path, 'r') as f:
            self.meta = json.load(f)

        self.data = self.meta['image_names']

        if transform is None:
            if self.partition == 'train' and self.data_aug:
                image_size = 224
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

        label = []
        lab = -1
        self.label_set = []
        for label_id in self.meta['image_labels']:

            if label_id not in self.label_set:
                self.label_set.append(label_id)
                lab += 1
            label.append(lab)

        self.label = label
        self.num_classes = len(set(label))
        # print(self.num_classes)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    def __len__(self):
        return len(self.data)


class MetaCUB(Dataset):

    def __init__(self, args, partition='test', train_transform=None, test_transform=None, fix_seed=True,image_size = 224):

        SPLIT_PATH = os.path.join(args.data_root, 'CUB')
        json_path = os.path.join(SPLIT_PATH, partition + '.json')
        with open(json_path, 'r') as f:
            self.meta = json.load(f)
        self.fix_seed = fix_seed
        self.n_ways = args.n_way
        self.n_shots = args.n_shot
        self.n_queries = args.n_queries
        self.n_episodes = args.n_episodes
        self.n_aug_support_samples = args.n_aug_support_samples
        self.args = args
        self.image_size = image_size
        self.n_sym_aug = args.n_symmetry_aug
        if image_size == 84:
            self.resize_size = 92
        elif image_size == 224:
            self.resize_size = 256
        # self.train_transform = train_transform
        # self.test_transform =test_transform
        if train_transform is None:
            pass
            # self.train_transform = transforms.Compose([
            #     transforms.RandomResizedCrop(image_size),
            #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
            #                          np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            pass
            # self.test_transform = transforms.Compose([
                # transforms.Resize(self.resize_size),
                # transforms.CenterCrop(self.image_size),
                # transforms.ToTensor(),
                # transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                #                      np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            self.test_transform = test_transform

        # self.data = self.meta['image_names']
        self.cl_list = np.unique(self.meta['image_labels']).tolist()
        self.sub_meta = {}
        for c in self.cl_list:
            self.sub_meta[c] = []
        batch_size = args.n_shot + args.n_queries
        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = subDataset(self.sub_meta[cl])
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

        self.num_classes = len(set(self.cl_list))
        self.seed_start = 0

        # print(test_transform.__dict__)
        # print(self.num_classes)

    def __getitem__(self, item):
            # if self.fix_seed:
            #     np.random.seed(item)
            cls_sampled = torch.randperm(self.num_classes)[:self.n_ways]
            support_xs, support_ys, query_xs, query_ys = [], [], [], []
            for idx, cls in enumerate(cls_sampled):
                sample_image = next(iter(self.sub_dataloader[cls]))
                idx_image = 0
                # print(sample_image[0])
                for _ in range(self.args.n_shot):
                    image_pil = Image.open(sample_image[idx_image]).convert('RGB')
                    idx_image += 1

                    image = self.test_transform(image_pil)
                    # image = self.train_transform(image_pil)
                    support_xs.append(image.unsqueeze(0))
                    support_ys.append(idx)

                    if self.n_aug_support_samples > 1:
                        for i in range(self.n_aug_support_samples - 1):
                            # print('---------------')
                            image = self.train_transform(image_pil)
                            support_xs.append(image.unsqueeze(0))
                            support_ys.append(idx)

                for _ in range(self.args.n_queries):
                    image_pil = Image.open(sample_image[idx_image]).convert('RGB')
                    idx_image += 1
                    if self.n_sym_aug > 1:
                        image = self.test_transform(image_pil)
                        # image = self.train_transform(image_pil)
                        query_xs.append(image.unsqueeze(0))
                        query_ys.append(idx)
                        if self.n_sym_aug > 1:
                            for i in range(self.n_sym_aug - 1):
                                image = self.train_transform(image_pil)
                                query_xs.append(image.unsqueeze(0))
                                query_ys.append(idx)
                    else:
                        image = self.test_transform(image_pil)
                        query_xs.append(image.unsqueeze(0))
                        query_ys.append(idx)
            # print(idx_image)
            # print(sample_record,end='\r')
            support_xs = torch.cat(support_xs, dim=0)
            support_ys = torch.tensor(support_ys)
            # print(support_ys)
            query_xs = torch.cat(query_xs, dim=0)
            query_ys = torch.tensor(query_ys)
            # print(query_ys)
            # print(support_xs.shape)
            return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_episodes

class subDataset:
    def __init__(self,sub_meta):
        self.sub_meta = sub_meta

    def __getitem__(self, i):
        image_path = os.path.join(self.sub_meta[i])
        return image_path

    def __len__(self):
        return len(self.sub_meta)

if __name__ == '__main__':
    args = lambda x: None
    args.n_way = 5
    args.n_shot = 5
    args.n_queries = 12
    args.data_root = '../../data'
    args.data_aug = True
    args.n_episodes = 5
    args.n_aug_support_samples = 5
    args.n_symmetry_aug = 5
    args.prompt = False
    imagenet = CUB(args, 'train')
    print(len(imagenet))
    # print(imagenet.__getitem__(500))

    metaimagenet = MetaCUB(args)
    print(len(metaimagenet))
    print(metaimagenet.__getitem__(500)[0].size())
    print(metaimagenet.__getitem__(500)[1].shape)
    print(metaimagenet.__getitem__(500)[2].size())
    print(metaimagenet.__getitem__(500)[3].shape)
