
import json
import numpy as np

import torch.optim
from torch.autograd import Variable

import time
import os
import argparse
from torch.utils.data import DataLoader
# from data_load.DataSets.MiniImageNet import *
from data_load.DataSets.MiniImageNet_BDC import *

from data_load.DataSets.CUB import *
from data_load.DataSets.TieredImageNet import *
# from method.Few_rec import Net_rec

from method.local_match import *
from method.stl_deepbdc import *
from method.Few_rec import *
import pprint
from data_load.transform_cfg import *
from utils.utils import *
from method.good_metric import *
from method.good_metric import Net
import warnings

torch.set_num_threads(4)
warnings.filterwarnings("ignore")
from utils import *

DATA_DIR = 'data'


def train_distill(args, train_loader, meta_valloader, model, model_t):

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    if args.method in ['stl_deepbdc']:
        bas_params = filter(lambda p: id(p) != id(model.dcov.temperature), model.parameters())
        optimizer = torch.optim.SGD([
            {'params': bas_params},
            {'params': model.dcov.temperature, 'lr': args.t_lr}], lr=args.lr, weight_decay=5e-4, nesterov=True, momentum=0.9)
    elif args.method in [ 'good_metric']:
        bas_params = filter(lambda p: id(p) != id(model.temperature) and id(p) != id(model.fpn_alpha), model.parameters())
        optimizer = torch.optim.SGD([
            {'params': bas_params},
            {'params': model.temperature, 'lr': args.t_lr},
            {'params': model.fpn_alpha, 'lr': args.t_lr*100}], lr=args.lr, weight_decay=5e-4, nesterov=True,
            momentum=0.9)
        if args.idea == 'bdc':
            bas_params = filter(lambda p: id(p) != id(model.dcov.temperature) and id(p) != id(model.dcov_fpn.temperature), model.parameters())
            optimizer = torch.optim.SGD([
                {'params': bas_params},
                {'params': model.dcov.temperature, 'lr': args.t_lr},
                {'params': model.dcov_fpn.temperature, 'lr': args.t_lr}
            ], lr=args.lr, weight_decay=5e-4, nesterov=True,
                momentum=0.9)
    elif args.method in ['confusion']:
        bas_params = filter(lambda p: id(p) != id(model.temperature) and id(p) != id(model.dcov.temperature),
                            model.parameters())
        optimizer = torch.optim.SGD([
            {'params': bas_params},
            {'params': model.temperature, 'lr': args.t_lr},
            {'params': model.dcov.temperature, 'lr': args.t_lr}], lr=args.lr, weight_decay=5e-4, nesterov=True,
            momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, nesterov=True, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)

    best_acc = 0.0
    best_confidence = 0.0
    # model_t.eval()
    for epoch in range(args.max_epoch):
        model.train()
        tic = time.time()
        avg_loss, acc = model.distillation(epoch,train_loader,optimizer,model_t)
        print("Epoch {} of {} | Avg_loss: {:.2f} | Acc_train: {:.2f} |  training eplase: {:.2f} min".format(epoch, args.max_epoch, avg_loss, acc, (time.time()-tic)/60),  )
        lr_scheduler.step()
        if epoch%args.val_freq ==0 and epoch > args.max_epoch//2:
            model.eval()
            with torch.no_grad():
                # gen_val = tqdm.tqdm(meta_valloader)
                tic = time.time()
                mean , confidence = model.meta_val_loop(epoch,val_loader=meta_valloader)
                if  best_acc <= mean :
                    best_acc = mean
                    best_confidence = confidence
                    model_save(model,args,epoch)
                print()
                print('-'*50)
                print("meta_val acc: {:.2f} +- {:.2f} |  best meta_val acc :{:.2f} +- {:.2f}   elapse: {:.2f} min".format(mean*100, confidence*100, best_acc*100, best_confidence *100,(time.time()-tic)/60))
                print('-' * 50)
    model_save(model, args, args.max_epoch-1)
    print("-"*20+" model prtrain finish! "+"-"*20)
    print("meta_val  best acc :{:.2f} +- {:.2f}".format(best_acc*100, best_confidence*100))

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def model_save(model, args,epoch):
    state = {'params': args,
             'model': model.state_dict()}
    # method = 'deep_emd' if args.deep_emd else 'local_match'
    method = args.method
    if epoch == args.max_epoch - 1:
        save_path = os.path.join(args.save_dir, args.dataset +"_" + args.method+"_resnet12_distill"
                                                        +("_"+str(args.model_id) if args.model_id else "")+"_"+str(args.k_gen)
                                                        +"_gen_last.pth")
    else:
        save_path = os.path.join(args.save_dir, args.dataset + "_" + args.method + "_resnet12_distill"
                                 + ("_" + str(args.model_id) if args.model_id else "") + "_" + str(args.k_gen)
                                 + "_gen_best.pth")
    if not args.no_save_model:
        torch.save(state,save_path)

def model_load(model,args):
    if args.k_gen == 1:
        pretrain_path = os.path.join(args.save_dir, args.dataset + "_" + args.method + "_resnet12_"+args.model_type
                                 + ("_" + str(args.model_id) if args.model_id else "") + ".pth")
    else:
        if args.model_type == 'best':
            pretrain_path = os.path.join(args.save_dir, args.dataset + "_" + args.method + "_resnet12_"+args.model_type
                                            + ("_"+str(args.model_id) if args.model_id else "") + ".pth")
        else:
            pretrain_path = os.path.join(args.save_dir, args.dataset + "_" + args.method + "_resnet12_distill"
                                         + ("_" + str(args.model_id) if args.model_id else "") + "_" + str( args.k_gen - 1)
                                         + "_gen_last.pth")
    print('teacher model path: ' + pretrain_path)
    state_dict = torch.load(pretrain_path)['model']
    model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--image_size', default=84, type=int, choices=[84, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
    parser.add_argument('--batch_size', default=64, type=int, help='pre-training batch size')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--lr', type=float, default=0.05, help='initial learning rate of the backbone')
    parser.add_argument('--t_lr', type=float, default=0.05, help='initial learning rate uesd for the temperature of bdc module')

    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--milestones', nargs='+', type=int, default=[100, 150], help='milestones for MultiStepLR')
    parser.add_argument('--max_epoch', default=180, type=int, help='stopping epoch')
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--model_type', default='best', choices=['best', 'last'])


    parser.add_argument('--dataset', default='miniimagenet', choices=['mini_imagenet','tieredimagenet','cub'])
    parser.add_argument('--data_root', type=str, default=DATA_DIR)

    parser.add_argument('--model', default='resnet12', choices=['resnet12', 'resnet18'])
    parser.add_argument('--img_size', default=84, type=int, choices=[84, 224])

    parser.add_argument('--val_freq',default=5,type=int)
    parser.add_argument('--val', default='meta', choices=['meta', 'last'], help='validation method')
    parser.add_argument('--n_episodes', default=300, type=int, help='number of episode in meta validation')
    parser.add_argument('--n_way', default=5, type=int, help='class num to classify in meta validation')
    parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support during meta validation')
    parser.add_argument('--n_queries', default=15, type=int, help='number of unlabeled data in each class during meta validation')
    parser.add_argument('--n_aug_support_samples', type=int, default=1)

    parser.add_argument('--sfc_lr', default=100, type=float)
    parser.add_argument('--sfc_bs', default=5, type=int)
    parser.add_argument('--sfc_update_step', default=100)
    parser.add_argument('--include_bg', default=False, action='store_true')

    parser.add_argument('--reduce_dim', default=640, type=int, help='the output dimensions of BDC dimensionality reduction layer')
    parser.add_argument('--drop_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    # setting
    parser.add_argument('--test_batch_size', default=1)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_dir', default='checkpoint')
    parser.add_argument('--test_LR', default=False, action='store_true')
    parser.add_argument('--model_id', default=None, type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--no_save_model', default=False, action='store_true')
    parser.add_argument('--feature_pyramid', default=False, action='store_true')
    parser.add_argument('--method',default='local_proto',choices=['local_proto','good_metric','stl_deepbdc','confusion'])
    parser.add_argument('--k_gen', default= 1, type=int, help='k generation self distillation ')
    parser.add_argument('--penalty_c', default=1., type=float)

    # good metric
    parser.add_argument('--stop_grad', default=False, action='store_true')
    parser.add_argument('--learnable_alpha', default=False, action='store_true')
    parser.add_argument('--idea', default='a+-b', choices=['ab', 'a+-b'])
    parser.add_argument('--flatten_fpn', default=False, action='store_true')
    parser.add_argument('--idea_variant', default=False, action='store_true')
    parser.add_argument('--normalize_feat', default=False, action='store_true')
    parser.add_argument('--normalize_bdc', default=False, action='store_true')
    parser.add_argument('--FPN_list', default=None, nargs='+', type=int)

    # confusion representation:
    parser.add_argument('--no_diag', default=False, action='store_true')
    parser.add_argument('--confusion', default=False, action='store_true')
    parser.add_argument('--k_c', default=3, type=int, help='k of cofusion')
    parser.add_argument('--confusion_beta', type=float, default=0.2)
    parser.add_argument('--confusion_drop_rate', default=0.3, type=float)
    parser.add_argument('--n_symmetry_aug', default=1, type=int)
    parser.add_argument('--ths_confusion', default=None, type=float,help='threshold of confusion')
    parser.add_argument('--lego', default=False, action='store_true')
    parser.add_argument('--temp_element',default=False,action='store_true')
    parser.add_argument('--metric', default='LR', choices=['LR','DN4'])
    parser.add_argument('--prompt', default=False, action='store_true')
    parser.add_argument('--constrastive', default=False, action='store_true')
    parser.add_argument('--embeding_way', default='BDC', choices=['BDC','GE'])
    parser.add_argument('--wd_test', type=float, default=5e-4)
    parser.add_argument('--LR', default=False,action='store_true')


    args = parser.parse_args()
    if args.img_size == 224 and args.transform=='B':
        args.transform = 'B224'
    if args.gpu:
        gpu_device = str(args.gpu)
    else:
        gpu_device = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
    pprint(args)
    set_seed(args.seed)

    train_partition = 'train'
    if args.dataset == 'miniimagenet':
        train_trans, test_trans = transforms_options[args.transform]
        # val_sup_trans = test_trans if args.n_aug_support_samples == 1 else train_trans
        val_sup_trans =  train_trans
        train_loader = DataLoader(ImageNet(args=args, partition=train_partition, transform=train_trans),
                                  batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                  num_workers=args.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=args, partition='val',
                                                 train_transform=val_sup_trans,
                                                 test_transform=test_trans),
                                    batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=args.num_workers)
        num_cls = 64
    elif args.dataset == 'cub':
        train_trans, test_trans = transforms_options[args.transform]
        # val_sup_trans = test_trans if args.n_aug_support_samples == 1 else train_trans
        val_sup_trans =  train_trans
        train_loader = DataLoader(CUB(args=args, partition=train_partition, transform=train_trans),
                                  batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                  num_workers=args.num_workers)
        meta_valloader = DataLoader(MetaCUB(args=args, partition='val',
                                                 train_transform=val_sup_trans,
                                                 test_transform=test_trans),
                                    batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=args.num_workers)

        num_cls = 100
    elif args.dataset == 'tieredimagenet':
        train_trans, test_trans = transforms_options[args.transform]
        # val_sup_trans = test_trans if args.n_aug_support_samples == 1 else train_trans
        val_sup_trans =  train_trans
        train_loader = DataLoader(TieredImageNet(args=args, partition=train_partition, transform=train_trans),
                                  batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                  num_workers=args.num_workers)
        meta_valloader = DataLoader(MetaTierdImageNet(args=args, partition='val',
                                            train_transform=val_sup_trans,
                                            test_transform=test_trans),
                                    batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=args.num_workers)

        num_cls = 351
    else:
        ValueError('dataset error')

    if args.method == 'stl_deepbdc':
        model = stl_deepbdc(args, num_classes=num_cls).cuda()
        model_t = stl_deepbdc(args, num_classes=num_cls).cuda()
        model_t = model_load(model_t, args)
        param = model.parameters()
    elif args.method == 'good_metric':
        model = Net(args, num_classes=num_cls).cuda()
        model_t = Net(args, num_classes=num_cls).cuda()
        model_t = model_load(model_t, args)
        param = model.parameters()
    elif args.prompt:
        model = Net_rec(args, num_classes=num_cls, ).cuda()
        model_t = Net_rec(args, num_classes=num_cls).cuda()
        model_t = model_load(model_t, args)
        param = model.parameters()

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)


    model = train_distill(args, train_loader, meta_valloader, model, model_t)
