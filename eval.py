import argparse
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
# from data_load.DataSets.MiniImageNet import *
from data_load.DataSets.MiniImageNet_BDC import *
from utils.utils import load_model, set_seed

# from data_load.DataSets.CUB import *
from data_load.DataSets.CUB_json import *
from data_load.DataSets.TieredImageNet import *
from data_load.DataSets.skin_198 import *
from method.Confusion import ConfuNet
from method.Few_rec import Net_rec
from method.good_metric import Net
from method.local_match import *
from method.stl_deepbdc import *
from data_load.transform_cfg import *
import pprint

DATA_DIR = 'data'

torch.set_num_threads(4)
_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def parse_option():
    parser = argparse.ArgumentParser('arguments for model pre-train')
    # about dataset and network
    parser.add_argument('--dataset', type=str, default='miniimagenet',
                        choices=['miniimagenet', 'cub', 'tieredimagenet', 'fc100', 'tieredimagenet_yao', 'cifar_fs', 'skin198'])
    parser.add_argument('--data_root', type=str, default=DATA_DIR)
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--model', default='resnet12', choices=['resnet12', 'resnet18','conv64','resnet34'])

    parser.add_argument('--img_size', default=84, type=int, choices=[84,224])


    # about model :
    parser.add_argument('--drop_gama', default=0.5, type= float)
    parser.add_argument('--MLP_2', default=False, action='store_true')
    parser.add_argument("--beta", default=0.01, type=float)
    parser.add_argument('--drop_rate', default=0.5, type=float)
    parser.add_argument('--reduce_dim', default=128, type=int)
    parser.add_argument('--idea', default='a+-b', choices=['ab', 'a+-b', 'bdc'])
    parser.add_argument('--FPN_list', default=None, nargs='+', type=int)
    parser.add_argument('--flatten_fpn', default=False, action='store_true')


    # about meta test
    parser.add_argument('--val_freq',default=5,type=int)
    # parser.add_argument('--local_mode',default='local_mix', choices=['cell', 'local_mix' ,'cell_mix','mask_pool'])
    parser.add_argument('--set', type=str, default='test', choices=['val', 'test'], help='the set for validation')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shot', type=int, default=1)
    parser.add_argument('--n_aug_support_samples',type=int, default=1)
    parser.add_argument('--n_queries', type=int, default=15)
    # parser.add_argument('--temperature', type=float, default=12.5)
    # parser.add_argument('--metric', type=str, default='cosine')
    parser.add_argument('--n_episodes', type=int, default=1000)
    # parser.add_argument('--n_local_proto', default=3, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    #  test_batch_size is 1  maen  1 episode of fsl
    parser.add_argument('--test_batch_size',default=1)
    # parser.add_argument('--sfc_lr', default=100,type = float)
    # parser.add_argument('--sfc_bs', default=5, type=int)
    # parser.add_argument('--sfc_update_step', default=100)
    # parser.add_argument('--include_bg', default=False, action='store_true')
    # parser.add_argument('--norm',default='center')


    # about deepemd setting
    # parser.add_argument('--norm', type=str, default='center', choices=['center'])
    # parser.add_argument('--solver', type=str, default='opencv', choices=['opencv'])
    parser.add_argument('--deep_emd', default=False, action='store_true')
    # parser.add_argument('-form', type=str, default='L2', choices=['QP', 'L2'])
    # parser.add_argument('-l2_strength', type=float, default=0.000001)

    # setting
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_dir', default='checkpoint')
    parser.add_argument('--continue_pretrain',default=False,action='store_true')
    parser.add_argument('--test_LR', default=False, action='store_true')
    parser.add_argument('--model_id', default=None, type=str)
    parser.add_argument('--model_type',default='best',choices=['best','last'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--no_save_model', default=False, action='store_true')
    # parser.add_argument('--feature_pyramid', default=False, action='store_true')
    parser.add_argument('--method',default='local_proto',choices=['local_proto','good_metric','stl_deepbdc','confusion'])
    parser.add_argument('--distill_model', default=None,type=str,help='about distillation model path')
    parser.add_argument('--penalty_c', default=1.0, type=float)
    # parser.add_argument('--stop_grad', default=False, action='store_true')
    # parser.add_argument('--learnable_alpha', default=False, action='store_true')
    parser.add_argument('--idea_variant', default=False, action='store_true')
    # parser.add_argument('--normalize_feat', default=False, action='store_true')
    # parser.add_argument('--normalize_bdc', default=False, action='store_true')
    parser.add_argument('--test_times', default=1, type=int)


    # confusion representation:
    # parser.add_argument('--no_diag', default=False, action='store_true')
    parser.add_argument('--confusion', default=False, action='store_true')
    # parser.add_argument('--k_c', default=3, type=float, help='k of cofusion')
    # parser.add_argument('--confusion_beta', type=float, default=1.)
    # parser.add_argument('--confusion_drop_rate', default=0.3, type=float)
    parser.add_argument('--n_symmetry_aug', default=1, type=int)
    # parser.add_argument('--ths_confusion', default=None, type=float,help='threshold of confusion')
    # parser.add_argument('--lego', default=False, action='store_true')
    # parser.add_argument('--temp_element',default=False,action='store_true')
    parser.add_argument('--metric', default='LR', choices=['LR','DN4'])
    parser.add_argument('--prompt', default=False, action='store_true')
    parser.add_argument('--constrastive', default=False, action='store_true')
    parser.add_argument('--embeding_way', default='BDC', choices=['BDC','GE'])
    parser.add_argument('--wd_test', type=float, default=5e-4)
    parser.add_argument('--LR', default=False,action='store_true')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--optim', default='Adam',choices=['Adam', 'SGD'])
    parser.add_argument('--my_model', default=False,action='store_true')
    parser.add_argument('--LR_rec', default=False, action='store_true')
    parser.add_argument('--drop_few',default=0.5,type=float)
    parser.add_argument('--skin_split', default=0.5, type=float)
    parser.add_argument('--no_fix_seed',default=False,action='store_true')
    parser.add_argument('--Loss_ablation',default=3,type= int ,choices=[0,1,2,3])


    args = parser.parse_args()
    if args.deep_emd:
        args.method = 'deep_emd'

    return args


def model_load(args,model):
    # method = 'deep_emd' if args.deep_emd else 'local_match'
    method = args.method
    save_path = os.path.join(args.save_dir, args.dataset + "_" + method + "_resnet12_"+args.model_type
                                            + ("_"+str(args.model_id) if args.model_id else "") + ".pth")
    if args.distill_model is not None:
        save_path = os.path.join(args.save_dir, args.distill_model)
    print('teacher model path: ' + save_path)
    state_dict = torch.load(save_path)['model']
    model.load_state_dict(state_dict)
    return model


def main():
    args = parse_option()
    if args.img_size == 224 and args.transform == 'B':
        args.transform = 'B224'
    if args.img_size == 224 and args.transform == 'B_s':
        args.transform = 'Bs224'
    # if args.transform == 'B':
    #     args.transform = 'B_s'
    pprint(args)
    if args.gpu:
        gpu_device = str(args.gpu)
    else:
        gpu_device = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
    # set_seed(args.seed)


    if args.dataset == 'miniimagenet':
        train_trans, test_trans = transforms_options[args.transform]
        # val_sup_trans = test_trans if args.n_aug_support_samples == 1 else train_trans
        val_sup_trans =  train_trans
        meta_test_loader = DataLoader(MetaImageNet(args=args, partition=args.set,
                                                 train_transform=val_sup_trans,
                                                 test_transform=test_trans),
                                    batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=args.num_workers)
        num_cls = 64
    elif args.dataset == 'cub':
        train_trans, test_trans = transforms_options[args.transform]
        # val_sup_trans = test_trans if args.n_aug_support_samples == 1 else train_trans
        # val_sup_trans =  train_trans
        # print(args.set)
        meta_test_loader = DataLoader(MetaCUB(args=args, partition=args.set,
                                                   train_transform=train_trans,
                                                   test_transform=test_trans),
                                      batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                      num_workers=args.num_workers)
        num_cls = 100
        # cross_domain
        # num_cls = 64
    elif args.dataset == 'tieredimagenet':
        train_trans, test_trans = transforms_options[args.transform]
        # val_sup_trans = test_trans if args.n_aug_support_samples == 1 else train_trans
        val_sup_trans =  train_trans
        meta_test_loader = DataLoader(MetaTierdImageNet(args=args, partition=args.set,
                                              train_transform=val_sup_trans,
                                              test_transform=test_trans),
                                      batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                      num_workers=args.num_workers)
        num_cls = 351
    elif args.dataset == 'skin198':
        train_trans, test_trans = transforms_options[args.transform]
        # val_sup_trans = test_trans if args.n_aug_support_samples == 1 else train_trans
        val_sup_trans =  train_trans
        meta_test_loader = DataLoader(MetaSkin(args=args, partition=args.set,
                                                   train_transform=val_sup_trans,
                                                   test_transform=test_trans),
                                      batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                      num_workers=args.num_workers)
        num_cls = 100

    if args.method in [ 'local_proto','deep_emd']:
        model = Local_match(args,num_classes=num_cls,local_proto=args.n_local_proto,mask_ad=True).cuda()
    elif args.method in ['stl_deepbdc']:
        model = stl_deepbdc(args, num_classes=num_cls).cuda()
    elif args.method in ['good_metric']:
        model = Net(args, num_classes=num_cls, ).cuda()
    elif args.prompt:
        model = Net_rec(args, num_classes=num_cls, ).cuda()
    elif args.method == 'confusion':
        model = ConfuNet(args, num_classes=num_cls, ).cuda()
    else:
        model = None
        assert model != None
    model.eval()
    if args.continue_pretrain:
        if args.my_model :
            model = model_load(args,model)
        else:
            model = load_model(model,os.path.join(args.save_dir,args.distill_model))

    print("-"*20+"  start meta test...  "+"-"*20)
    # model.eval()
    # gen_test = tqdm.tqdm(meta_test_loader)
    acc_sum = 0
    confidence_sum = 0
    for t in range(args.test_times):
        meta_test_loader.seed_start = t*args.n_episodes
        with torch.no_grad():
            tic = time.time()
            mean, confidence = model.meta_test_loop(meta_test_loader)
            # mean, confidence = model.meta_val_loop(None,meta_test_loader,None)
            acc_sum += mean
            confidence_sum += confidence
            print()
            print("Time {} :meta_val acc: {:.2f} +- {:.2f}   elapse: {:.2f} min".format(t,mean * 100, confidence * 100,
                                                                               (time.time() - tic) / 60))

    print("{} times \t acc: {:.2f} +- {:.2f}".format(args.test_times, acc_sum/args.test_times * 100, confidence_sum/args.test_times * 100, ))

if __name__ == '__main__':
    main()