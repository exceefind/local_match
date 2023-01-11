import argparse
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from data_load.DataSets.MiniImageNet_fg import *
from method.good_metric import Net
from method.local_match import *
from method.stl_deepbdc import *
from data_load.transform_cfg import *
import pprint

DATA_DIR = ''

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def parse_option():
    parser = argparse.ArgumentParser('arguments for model pre-train')
    # about dataset and network
    parser.add_argument('--dataset', type=str, default='miniimagenet',
                        choices=['miniimagenet', 'cub', 'tieredimagenet', 'fc100', 'tieredimagenet_yao', 'cifar_fs'])
    parser.add_argument('--data_root', type=str, default=DATA_DIR)
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)

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
    parser.add_argument('--local_mode',default='local_mix', choices=['cell', 'local_mix' ,'cell_mix','mask_pool'])
    parser.add_argument('--set', type=str, default='val', choices=['val', 'test'], help='the set for validation')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shot', type=int, default=1)
    parser.add_argument('--n_aug_support_samples',type=int, default=1)
    parser.add_argument('--n_queries', type=int, default=15)
    parser.add_argument('--temperature', type=float, default=12.5)
    parser.add_argument('--metric', type=str, default='cosine')
    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--n_local_proto', default=3, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    #  test_batch_size is 1  maen  1 episode of fsl
    parser.add_argument('--test_batch_size',default=1)

    parser.add_argument('--sfc_lr', default=100,type = float)
    parser.add_argument('--sfc_bs', default=5, type=int)
    parser.add_argument('--sfc_update_step', default=100)
    parser.add_argument('--include_bg', default=False, action='store_true')
    # parser.add_argument('--norm',default='center')


    # about deepemd setting
    parser.add_argument('--norm', type=str, default='center', choices=['center'])
    parser.add_argument('--solver', type=str, default='opencv', choices=['opencv'])
    parser.add_argument('--deep_emd', default=False, action='store_true')
    parser.add_argument('-form', type=str, default='L2', choices=['QP', 'L2'])
    parser.add_argument('-l2_strength', type=float, default=0.000001)

    # setting
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_dir', default='checkpoint')
    parser.add_argument('--continue_pretrain',default=False,action='store_true')
    parser.add_argument('--test_LR', default=False, action='store_true')
    parser.add_argument('--model_id', default=None, type=str)
    parser.add_argument('--model_type',default='best',choices=['best','last'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--no_save_model', default=False, action='store_true')
    parser.add_argument('--feature_pyramid', default=False, action='store_true')
    parser.add_argument('--method', default='local_proto', choices=['local_proto', 'good_metric', 'stl_deepbdc'])
    parser.add_argument('--distill_model', default=None,type=str,help='about distillation model path')
    parser.add_argument('--penalty_c', default=1.0, type=float)
    parser.add_argument('--stop_grad', default=False, action='store_true')
    parser.add_argument('--learnable_alpha', default=False, action='store_true')
    parser.add_argument('--idea_variant', default=False, action='store_true')
    parser.add_argument('--normalize_feat', default=False, action='store_true')
    parser.add_argument('--fg_extract', default=False, action='store_true')

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
    pprint(args)
    if args.gpu:
        gpu_device = str(args.gpu)
    else:
        gpu_device = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device

    if args.dataset == 'miniimagenet':
        train_trans, test_trans = transforms_options[args.transform]
        val_sup_trans = test_trans if args.n_aug_support_samples == 1 else train_trans
        meta_test_loader = DataLoader(MetaImageNet(args=args, partition='test',
                                                 train_transform=val_sup_trans,
                                                 test_transform=test_trans),
                                    batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=args.num_workers)
        num_cls = 64
    if args.method in [ 'local_proto','deep_emd']:
        model = Local_match(args,num_classes=num_cls,local_proto=args.n_local_proto,mask_ad=True).cuda()
    elif args.method in ['stl_deepbdc']:
        model = stl_deepbdc(args, num_classes=num_cls).cuda()
    elif args.method in ['good_metric']:
        model = Net(args, num_classes=num_cls, ).cuda()
    else:
        model = None
        assert model != None

    if args.continue_pretrain:
        model = model_load(args,model)

    print("-"*20+"  start meta test...  "+"-"*20)
    model.eval()
    # gen_test = tqdm.tqdm(meta_test_loader)
    with torch.no_grad():
        tic = time.time()
        mean, confidence = model.meta_test_loop(meta_test_loader)
        print()
        print("meta_val acc: {:.2f} +- {:.2f}   elapse: {:.2f} min".format(mean * 100, confidence * 100,
                                                                           (time.time() - tic) / 60))


if __name__ == '__main__':
    main()