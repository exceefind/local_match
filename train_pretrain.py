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
from data_load.DataSets.MiniImageNet_BDC_all import *

from data_load.DataSets.CUB_json import *
from data_load.DataSets.TieredImageNet import *
from data_load.DataSets.skin_198 import *
from method.local_match import *
from method.stl_deepbdc import *
from method.Confusion import *
from method.FewCL import *
from method.Few_rec import *
import logging
import pprint

from data_load.transform_cfg import *

from utils.utils import *
from method.good_metric import Net
import warnings

torch.set_num_threads(4)
warnings.filterwarnings("ignore")

DATA_DIR = 'data'

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
    parser.add_argument('--model', default='resnet12',choices=['resnet12', 'resnet18','conv64','resnet34'])
    parser.add_argument('--img_size', default=84, type=int, choices=[84,224])

    # about model :
    parser.add_argument('--drop_gama', default=0.3, type= float)

    #  about pretrain
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--t_lr', type=float, default=0.05)
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--gama', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument("--beta",default=0.01,type=float)
    parser.add_argument('--MLP_2', default=False, action='store_true')
    parser.add_argument('--milestones',default=[80,120,150], nargs='+',type=int)
    parser.add_argument('--drop_rate', default=0.5 ,type=float)
    parser.add_argument('--reduce_dim', default=128, type = int )
    parser.add_argument('--FPN_list', default=None, nargs='+', type=int,help='FPN list: use for changing more layer feat_map . [0, 1, 2, 3]')
    parser.add_argument('--idea', default='a+-b', choices=['ab', 'a+-b', 'bdc'])
    parser.add_argument('--flatten_fpn', default=False, action='store_true')

    # about validation
    parser.add_argument('--val_freq',default=5,type=int)
    parser.add_argument('--local_mode',default='local_mix', choices=['cell', 'local_mix', 'cell_mix','mask_pool'])
    parser.add_argument('--set', type=str, default='val', choices=['val', 'test'], help='the set for validation')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shot', type=int, default=1)
    parser.add_argument('--n_aug_support_samples',type=int, default=1)
    parser.add_argument('--n_queries', type=int, default=15)
    parser.add_argument('--temperature', type=float, default=12.5)
    # parser.add_argument('--metric', type=str, default='cosine')
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--n_local_proto', default=3, type=int)
    parser.add_argument('--save_all', action='store_true', help='save models on each epoch')


    #  test_batch_size is 1  maen  1 episode of fsl
    parser.add_argument('--test_batch_size',default=1)
    parser.add_argument('--random_val_task', action='store_true',
                        help='random samples tasks for validation in each epoch')
    parser.add_argument('--sfc_lr', default=100 , type=float)
    parser.add_argument('--sfc_bs', default=5, type=int)
    parser.add_argument('--sfc_update_step', default=100)
    parser.add_argument('--include_bg', default=False, action='store_true')
    # parser.add_argument('--norm',default='center')

    # about deepemd setting
    parser.add_argument('--norm', type=str, default='center', choices=['center'])
    parser.add_argument('--solver', type=str, default='opencv', choices=['opencv'])
    parser.add_argument('--deep_emd', default=False, action = 'store_true')
    parser.add_argument('-form', type=str, default='L2', choices=['QP', 'L2'])
    parser.add_argument('-l2_strength', type=float, default=0.000001)

    # setting
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_dir', default='checkpoint')
    parser.add_argument('--continue_pretrain',default=False,action='store_true')
    parser.add_argument('--MultiStepLr', default=False, action='store_true')
    parser.add_argument('--test_LR', default=False, action='store_true')
    parser.add_argument('--model_id',default=None, type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--no_save_model', default= False, action = 'store_true')
    parser.add_argument('--feature_pyramid',default=False,action= 'store_true')
    parser.add_argument('--method',default='local_proto',choices=['local_proto','good_metric','stl_deepbdc','confusion'])
    parser.add_argument('--test', default=False, action = 'store_true')
    parser.add_argument('--penalty_c', default=1.0, type=float)
    parser.add_argument('--stop_grad', default=False, action='store_true')
    parser.add_argument('--learnable_alpha', default=False, action='store_true')
    parser.add_argument('--idea_variant', default=False, action='store_true')
    parser.add_argument('--normalize_feat', default=False, action='store_true')
    parser.add_argument('--normalize_bdc', default=False, action='store_true')

    # confusion representation:
    parser.add_argument('--no_diag', default=False, action='store_true')
    parser.add_argument('--confusion', default=False, action='store_true')
    parser.add_argument('--confusion_drop_rate', default=0, type=float)
    parser.add_argument('--k_c', default=5, type=int, help='k of cofusion')
    parser.add_argument('--confusion_beta', type=float, default=1.)
    parser.add_argument('--n_symmetry_aug', default=1, type=int)
    parser.add_argument('--ths_confusion', default=None, type=float,help='threshold of confusion')
    parser.add_argument('--lego', default=False, action='store_true')
    parser.add_argument('--temp_element',default=False,action='store_true')
    parser.add_argument('--metric', default='LR', choices=['LR','DN4'])
    parser.add_argument('--prompt', default=False, action='store_true')
    parser.add_argument('--constrastive', default=False, action='store_true')
    parser.add_argument('--embeding_way', default='BDC', choices=['BDC','GE','baseline++'])
    parser.add_argument('--wd_test', type=float, default=5e-4)
    parser.add_argument('--LR', default=False,action='store_true')
    parser.add_argument('--optim', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--my_model', default=False, action='store_true')
    parser.add_argument('--no_fix_seed', default=False, action='store_true')
    parser.add_argument('--skin_split', default=0.1, type=float)
    parser.add_argument('--cross_all', default=False,action='store_true')

    parser.add_argument('--drop_few', default=0.6, type=float)
    parser.add_argument('--ablation', default=0, type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--LR_rec', default=False,action='store_true')



    args = parser.parse_args()



    return args

def model_save(model, args,epoch):
    state = {'params': args,
             'model': model.state_dict()}
    # method = 'deep_emd' if args.deep_emd else 'local_match'
    method = args.method
    if epoch == args.max_epoch-1:
        save_path = os.path.join(args.save_dir, args.dataset+"_"+method+"_resnet12_last"
                                                +("_"+str(args.model_id) if args.model_id else "") +".pth")
    else:
        save_path = os.path.join(args.save_dir, args.dataset+"_"+method + "_resnet12_best"
                                                +("_"+str(args.model_id) if args.model_id else "") +".pth")
    if not args.no_save_model:
        torch.save(state,save_path)

def model_load(model,args):
    method = 'deep_emd' if args.deep_emd else 'local_match'
    pretrain_path = os.path.join(args.save_dir, args.dataset + "_" + method + "_resnet12_best.pth")
    state_dict = torch.load(pretrain_path)['model']
    model.load_state_dict(state_dict)
    return model

def main():
    args = parse_option()
    if args.img_size == 224 and args.transform=='B':
        args.transform = 'B224'
    pprint(args)
    if args.gpu:
        gpu_device = str(args.gpu)
    else:
        gpu_device = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
    set_seed(args.seed)
    # device = torch.device('cuda')
    train_partition = 'train'

    logging.basicConfig(filename='log/'+args.dataset+'_'+args.model+'_pretrain_'+(args.model_id if args.model_id else " " )+'.log', level=logging.INFO,
                            format="%(levelname)s: %(asctime)s : %(message)s")

    logging.info("-" * 100 + "Experiment " + (args.model_id if args.model_id else "") + "start!" + "-" * 100)
    # 记录实验超参数设置:
    logging.info(args.__dict__)
    meta_valloader =None
    if args.dataset == 'miniimagenet':
        train_trans, test_trans = transforms_options[args.transform]
        val_sup_trans =  train_trans
        if args.cross_all:

            train_loader = DataLoader(ImageNet_all(args=args, partition=train_partition, transform=train_trans),
                                      batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                      num_workers=args.num_workers)
            num_cls = 100
        else:
            # val_sup_trans = test_trans if args.n_aug_support_samples == 1 else train_trans
            train_loader = DataLoader(ImageNet(args=args, partition=train_partition, transform=train_trans),
                                      batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                      num_workers=args.num_workers)
            meta_valloader = DataLoader(MetaImageNet(args=args, partition='val',
                                                     train_transform=val_sup_trans,
                                                     test_transform=test_trans),
                                        batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                        num_workers=args.num_workers)
            if args.test:
                meta_test_loader = DataLoader(MetaImageNet(args=args, partition='test',
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
        if args.test:
            meta_test_loader = DataLoader(MetaCUB(args=args, partition='test',
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
        if args.test:
            meta_test_loader = DataLoader(MetaTierdImageNet(args=args, partition='test',
                                                  train_transform=val_sup_trans,
                                                  test_transform=test_trans),
                                          batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                          num_workers=args.num_workers)
        num_cls = 351
    elif args.dataset == 'skin198':
        train_trans, test_trans = transforms_options[args.transform]
        # val_sup_trans = test_trans if args.n_aug_support_samples == 1 else train_trans
        val_sup_trans =  train_trans
        train_loader = DataLoader(Skin_198(args=args, partition=train_partition, transform=train_trans),
                                  batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                  num_workers=args.num_workers)
        meta_valloader = DataLoader(MetaSkin(args=args, partition='val',
                                                 train_transform=val_sup_trans,
                                                 test_transform=test_trans),
                                    batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=args.num_workers)
        if args.test:
            meta_test_loader = DataLoader(MetaSkin(args=args, partition='test',
                                                       train_transform=val_sup_trans,
                                                       test_transform=test_trans),
                                          batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                          num_workers=args.num_workers)
        num_cls = 100

    if args.method == 'local_proto':
        model = Local_match(args,num_classes=num_cls,local_proto=args.n_local_proto,mask_ad=True).cuda()
        if args.continue_pretrain:
            model = model_load(model, args)
        param_net = filter((lambda x: id(x) not in list(map(id, model.classifier_disc.parameters()))),
                           model.parameters())
        param = [{'params': param_net}]
        optimizer_disc = torch.optim.Adam(model.classifier_disc.parameters(), lr=0.001)
    elif args.method == 'stl_deepbdc':
        model = stl_deepbdc(args,num_classes=num_cls).cuda()
        if args.continue_pretrain:
            model = model_load(model, args)
        param = model.parameters()
        optimizer_disc = None
    elif args.prompt:
        model = Net_rec(args, num_classes=num_cls, ).cuda()
        if args.continue_pretrain:
            model = model_load(model, args)
        param = model.parameters()
        optimizer_disc = None
    elif args.constrastive:
        model = Net_CL(args, num_classes=num_cls, ).cuda()
        if args.continue_pretrain:
            model = model_load(model, args)
        param = model.parameters()
        optimizer_disc = None
    elif args.method == 'confusion':
        model = ConfuNet(args, num_classes=num_cls, ).cuda()
        if args.continue_pretrain:
            model = model_load(model, args)
        param = model.parameters()
        optimizer_disc = None
    else:
        model = Net(args, num_classes=num_cls,).cuda()
        if args.continue_pretrain:
            model = model_load(model, args)
        param = model.parameters()
        optimizer_disc = None

    if args.method in [ 'good_metric']:
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
    elif args.method in ['stl_deepbdc',]:
        bas_params = filter(lambda p: id(p) != id(model.dcov.temperature), model.parameters())
        optimizer = torch.optim.SGD([
            {'params': bas_params},
            {'params': model.dcov.temperature, 'lr': args.t_lr}], lr=args.lr, weight_decay=5e-4, nesterov=True,
            momentum=0.9)
    elif args.method in [ 'confusion']:
        bas_params = filter(lambda p: id(p) != id(model.temperature) and id(p) != id(model.dcov.temperature), model.parameters())
        optimizer = torch.optim.SGD([
            {'params': bas_params},
            {'params': model.dcov.temperature, 'lr': args.t_lr},], lr=args.lr, weight_decay=5e-4, nesterov=True,momentum=0.9)
    else:
        optimizer = torch.optim.SGD(param,lr=args.lr,momentum=0.9,nesterov=True,weight_decay=5e-4)

    if args.MultiStepLr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gama)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(args.max_epoch / 20), T_mult=19)


    print("-"*20+"  start pretrain...  "+"-"*20)
    # gen_train = tqdm.tqdm(train_loader)
    best_acc = 0
    best_confidence = 0

    for epoch in range(args.max_epoch):
        model.train()
        tic = time.time()
        avg_loss, acc = model.train_loop(epoch,train_loader,[optimizer,optimizer_disc])
        print("Epoch {} of {} | Avg_loss: {:.2f} | Acc_train: {:.2f}  |  training eplase: {:.2f} min".format(epoch, args.max_epoch, avg_loss, acc, (time.time()-tic)/60) , )
        logging.info("Epoch {} of {} | Avg_loss: {:.2f} | Acc_train: {:.2f}  |  training eplase: {:.2f} min".format(epoch, args.max_epoch, avg_loss, acc, (time.time()-tic)/60) , )
        lr_scheduler.step()
        if epoch%args.val_freq ==0 and epoch > args.max_epoch//2 :
            model.eval()
            with torch.no_grad():
                # gen_val = tqdm.tqdm(meta_valloader)
                tic = time.time()
                mean , confidence = model.meta_test_loop(meta_valloader)
                if not args.continue_pretrain and best_acc <= mean:
                    best_acc = mean
                    best_confidence = confidence
                    model_save(model,args,epoch)
                print('-'*100)
                print("meta_val acc: {:.2f} +- {:.2f} |  best meta_val acc :{:.2f} +- {:.2f}  |  elapse: {:.2f} min".format(mean*100, confidence*100, best_acc*100, best_confidence *100,(time.time()-tic)/60))
                print('-'*100)
                logging.info('-'*200)
                logging.info("meta_val acc: {:.2f} +- {:.2f} |  best meta_val acc :{:.2f} +- {:.2f}  |  elapse: {:.2f} min".format(mean*100, confidence*100, best_acc*100, best_confidence *100,(time.time()-tic)/60))
                logging.info('-' * 200)

        if args.test and epoch > args.max_epoch//2 and (epoch % args.val_freq == 0 or  epoch == args.max_epoch-1):
            model.eval()
            with torch.no_grad():
                tic = time.time()
                print('-' * 100)
                mean, confidence = model.meta_test_loop(meta_test_loader)
                print()
                print("meta_test acc: {:.2f} +- {:.2f}   elapse: {:.2f} min".format(mean * 100, confidence * 100,
                                                                                   (time.time() - tic) / 60))
                print('-' * 100)
                logging.info("meta_test acc: {:.2f} +- {:.2f}   elapse: {:.2f} min".format(mean * 100, confidence * 100,(time.time() - tic) / 60))
                logging.info('-' * 200)
    model_save(model, args, args.max_epoch-1)
    print("-"*20+" model prtrain finish! "+"-"*20)
    print("meta_val  best acc :{:.2f} +- {:.2f}".format(best_acc*100, best_confidence*100))

if __name__ == '__main__':
    main()