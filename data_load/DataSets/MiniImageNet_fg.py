import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageNet(Dataset):
    def __init__(self, args, partition='train', data_aug = True,transform=None):
        super(Dataset, self).__init__()
        IMAGE_PATH = os.path.join(args.data_root, 'miniimagenet/images')
        SPLIT_PATH = os.path.join(args.data_root, 'miniimagenet/split')
        csv_path = os.path.join(SPLIT_PATH, partition + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        self.partition = partition
        self.data_aug = data_aug

        if transform is None:
            if self.partition == 'train' and self.data_aug:
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

        data , label = [], []
        lab = -1
        self.label_set = []
        for line in lines:
            name , lab_id = line.split(',')
            path  = os.path.join(IMAGE_PATH,name)
            if lab_id not in self.label_set:
                self.label_set.append(lab_id)
                lab += 1
            data.append(path)
            label.append(lab)
        self.data = data
        self.label = label
        self.num_classes = len(set(label))


    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    def __len__(self):
        return len(self.data)


class MetaImageNet(Dataset):

    def __init__(self, args, partition='test', train_transform=None, test_transform=None, fix_seed=True,image_size = 84):
        super(MetaImageNet, self).__init__()
        IMAGE_PATH = os.path.join(args.data_root, 'miniimagenet_fg/images')
        MASK_PATH = os.path.join(args.data_root,'miniimagenet_fg/mask_label')
        SPLIT_PATH = os.path.join(args.data_root, 'miniimagenet_fg/split')
        csv_path = os.path.join(SPLIT_PATH, partition + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        self.fix_seed = fix_seed
        self.n_ways = args.n_way
        self.n_shots = args.n_shot
        self.n_queries = args.n_queries
        self.n_episodes = args.n_episodes
        self.n_aug_support_samples = args.n_aug_support_samples
        self.image_size = image_size
        self.fg_extract = args.fg_extract
        if image_size == 84:
            self.resize_size = 92
        elif image_size == 224:
            self.resize_size = 256
        if train_transform is None:
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            self.test_transform = test_transform

        data, label = [], []
        label_mask = []
        lab = -1
        self.label_set = []
        for line in lines:
            name, lab_id = line.split(',')
            path_mask = os.path.join(MASK_PATH,name.split('.')[0]+"_json","label.png")
            path = os.path.join(IMAGE_PATH, name)
            if lab_id not in self.label_set:
                self.label_set.append(lab_id)
                lab += 1
            label_mask.append(path_mask)
            data.append(path)
            label.append(lab)
        self.data = data
        self.label = label
        self.label_mask = label_mask
        self.num_classes = len(set(label))

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(range(self.num_classes), self.n_ways, False)
        # print(cls_sampled)
        support_xs, support_ys, query_xs, query_ys = [], [], [], []
        for idx, cls in enumerate(cls_sampled):
            # all idx of cls
            samples_cls = np.where(np.array(self.label)==cls)[0]
            support_xs_ids_sampled = np.random.choice(samples_cls, self.n_shots, False)
            for sample_id in support_xs_ids_sampled:
                image_pil = Image.open(self.data[sample_id]).convert('RGB')
                if self.fg_extract :
                    img_convert_ndarray = np.asarray(image_pil,)
                    image_mask = Image.open(self.label_mask[sample_id]).convert('RGB')
                    img_mask_arr = np.uint8(np.expand_dims(np.array(image_mask)[:,:,0]/128,2))
                    img_convert_ndarray *= img_mask_arr
                    image_pil = Image.fromarray(img_convert_ndarray)
                for i in range(self.n_aug_support_samples):
                    image = self.train_transform(image_pil)
                    support_xs.append(image.unsqueeze(0))
                    support_ys.append(idx)
            query_xs_ids = np.setxor1d(samples_cls, support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            for sample_id in query_xs_ids:
                image = Image.open(self.data[sample_id]).convert('RGB')
                if self.fg_extract :
                    img_ndarray = np.array(image)
                    image_mask = Image.open(self.label_mask[sample_id]).convert('RGB')
                    img_mask_arr = np.uint8(np.expand_dims(np.array(image_mask)[:, :, 0] / 128, 2))
                    img_ndarray *= img_mask_arr
                    image = Image.fromarray(img_ndarray)
                image = self.test_transform(image)
                query_xs.append(image.unsqueeze(0))
                query_ys.append(idx)
        support_xs = torch.cat(support_xs,dim=0)
        support_ys = torch.tensor(support_ys)
        query_xs = torch.cat(query_xs,dim=0)
        query_ys = torch.tensor(query_ys)

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_episodes


if __name__ == '__main__':
    args = lambda x: None
    args.n_way = 5
    args.n_shot = 5
    args.n_queries = 12
    args.data_root = '../../'
    args.data_aug = True
    args.n_episodes = 5
    args.n_aug_support_samples = 5

    metaimagenet = MetaImageNet(args)
    print(len(metaimagenet))
    # print(metaimagenet.__getitem__(500)[0].size())
    print(metaimagenet.__getitem__(500)[1].shape)
    print(metaimagenet.__getitem__(500)[2].size())
    print(metaimagenet.__getitem__(500)[3].shape)
