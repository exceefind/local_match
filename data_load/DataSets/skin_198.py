import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Skin_198(Dataset):
    def __init__(self, args, partition='train', data_aug = True,transform=None):
        super(Dataset, self).__init__()
        # IMAGE_PATH = os.path.join(args.data_root, 'skin198')
        # SPLIT_PATH = os.path.join(args.data_root, 'skin198/split')
        # csv_path = os.path.join(SPLIT_PATH, partition + '.csv')
        # lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        ROOT_PATH = os.path.join(args.data_root, 'skin198','images')
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
        # self.label_set = []
        # for line in lines:
        #     name , lab_id = line.split(',')
        #     path  = os.path.join(IMAGE_PATH,name)
        #     if lab_id not in self.label_set:
        #         self.label_set.append(lab_id)
        #         lab += 1
        #     data.append(path)
        #     label.append(lab)
        self.label_set = os.listdir(ROOT_PATH)
        for cls_id in self.label_set:
            cls_imgs = os.listdir(os.path.join(ROOT_PATH, cls_id))
            for file in cls_imgs:
                data.append(os.path.join(ROOT_PATH, cls_id, file))
            # data.extend(cls_imgs)
            label.extend([lab] * len(cls_imgs))
            lab += 1
        self.num_classes = len(set(label))
        self.num_pc = np.zeros(self.num_classes)
        for i in label:
            self.num_pc[i] += 1
        self.data = []
        self.label = []
        self.class_few = np.argsort(self.num_pc)[-100:]
        lab = -1
        lab_set_sl = []
        for i,lab_sl in enumerate(label):
            if lab_sl in self.class_few:
                self.data.append(data[i])
                if lab_sl not in lab_set_sl:
                    lab_set_sl.append(lab_sl)
                    lab += 1
                self.label.append(lab)


    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    def __len__(self):
        return len(self.data)


class MetaSkin(Skin_198):
    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True,image_size = 84):
        super(MetaSkin, self).__init__(args, partition, False)
        # IMAGE_PATH = os.path.join(args.data_root, 'skin198')
        # SPLIT_PATH = os.path.join(args.data_root, 'skin198/split')
        # csv_path = os.path.join(SPLIT_PATH, partition + '.csv')
        # lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        ROOT_PATH = os.path.join(args.data_root, 'skin198', 'images')
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
        lab = -1

        # self.label_set = []
        # for line in lines:
        #     name, lab_id = line.split(',')
        #     path = os.path.join(IMAGE_PATH, name)
        #     if lab_id not in self.label_set:
        #         self.label_set.append(lab_id)
        #         lab += 1
        #     data.append(path)
        #     label.append(lab)
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
        self.num_pc = np.zeros(self.num_classes)
        self.seed_start = 0

        for i in self.label:
            self.num_pc[i] += 1
        self.class_few = np.argsort(self.num_pc,)[:self.n_ways]
        # print(np.sort(self.num_pc))

    def __getitem__(self, item):
        # if self.fix_seed:
        #     np.random.seed(item)
        cls_sampled = np.random.choice(self.class_few, self.n_ways, False)
        # print(cls_sampled)
        support_xs, support_ys, query_xs, query_ys = [], [], [], []
        # sample_record = []
        for idx, cls in enumerate(cls_sampled):
            # all idx of cls
            samples_cls = np.where(np.array(self.label) == cls)[0]
            support_xs_ids_sampled = np.random.choice(samples_cls, int(self.num_pc[cls]*self.args.skin_split), False)
            for sample_id in support_xs_ids_sampled:
                # sample_record.append((self.data[sample_id]).split('\\\\')[-1])
                image_pil = Image.open(self.data[sample_id]).convert('RGB')

                image = self.test_transform(image_pil)
                support_xs.append(image.unsqueeze(0))
                support_ys.append(idx)
                if self.n_aug_support_samples > 1:
                    for i in range(self.n_aug_support_samples-1):
                        image = self.train_transform(image_pil)
                        support_xs.append(image.unsqueeze(0))
                        support_ys.append(idx)

            query_xs_ids = np.setxor1d(samples_cls, support_xs_ids_sampled)

            for sample_id in query_xs_ids:
                image_pil = Image.open(self.data[sample_id]).convert('RGB')
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
        # print(sample_record,end='\r')
        support_xs = torch.cat(support_xs, dim=0)
        support_ys = torch.tensor(support_ys)
        query_xs = torch.cat(query_xs, dim=0)
        query_ys = torch.tensor(query_ys)

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_episodes


if __name__ == '__main__':
    args = lambda x: None
    args.n_way = 5
    args.n_shot = 5
    args.n_queries = 12
    args.data_root = '../../data'
    args.data_aug = True
    args.n_episodes = 5
    args.n_aug_support_samples = 5
    args.n_symmetry_aug = 1
    imagenet = Skin_198(args, 'train')
    print(len(imagenet))
    print(imagenet.__getitem__(500))

    metaimagenet = MetaSkin(args)
    print(len(metaimagenet))
    print(metaimagenet.__getitem__(500)[0].size())
    print(metaimagenet.__getitem__(500)[1].shape)
    print(metaimagenet.__getitem__(500)[2].size())
    print(metaimagenet.__getitem__(500)[3].shape)
