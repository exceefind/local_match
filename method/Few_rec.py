import os
import sys
import time

import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from method.bdc_module import BDC
import torch.nn.functional as F

sys.path.append("..")
import numpy as np
import torch.nn as nn
import torch
import scipy
from scipy.stats import t
from model.resnet_new1 import ResNet12 as resnet12
from model.resnet_new1 import *
from model.convnet import *
# from .DN4_module import *
import random
from utils.loss import *
from sklearn.linear_model import LogisticRegression as LR
# torch.autograd.set_detect_anomaly(True)
from utils.distillation_utils import *
import math
from torch.nn.utils.weight_norm import WeightNorm
from utils.loss import uniformity_loss
from collections import Counter


# 最简单的方式进行匹配： 求各个local feat 与loca proto的匹配，取个local feat最匹配下的距离，再通过平均的方式确定匹配度量
def compute_match_scores(x,SFC):
    # x : local_feat * in_feature
    # SFC : num_cls * local_proto * in_feature
    x_norm = torch.norm(x, p=2, dim=1,).unsqueeze(1).expand_as(x)
    x_ = x.div(x_norm).unsqueeze(0).expand(SFC.shape[0],x.shape[0],x.shape[1]).permute(0,2,1)
    SFC_norm = torch.norm(SFC, p=2, dim=2).unsqueeze(2).expand_as(SFC)
    SFC_ = SFC.div(SFC_norm)
    # print(SFC_)
    out  = torch.bmm(SFC_ ,x_)
    # print(torch.max(out,dim=1)[0].shape)
    score = torch.mean(torch.max(out,dim=1)[0],dim=1)
    # print(score)
    return score

def mean_confidence_interval(data, confidence=0.95,multi = 1):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m * multi, h * multi


def Distance_Correlation(latent, control):
    latent = F.normalize(latent)
    control = F.normalize(control)

    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim=-1) + 1e-12)
    matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim=-1) + 1e-12)

    matrix_A = matrix_a - torch.mean(matrix_a, dim=0, keepdims=True) - torch.mean(matrix_a, dim=1,
                                                                                  keepdims=True) + torch.mean(matrix_a)
    matrix_B = matrix_b - torch.mean(matrix_b, dim=0, keepdims=True) - torch.mean(matrix_b, dim=1,
                                                                                  keepdims=True) + torch.mean(matrix_b)

    Gamma_XY = torch.sum(matrix_A * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_XX = torch.sum(matrix_A * matrix_A) / (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_YY = torch.sum(matrix_B * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])

    correlation_r = Gamma_XY / torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
    # correlation_r = torch.pow(Gamma_XY,2)/(Gamma_XX * Gamma_YY + 1e-9)
    return correlation_r

def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out

def random_sample(linspace, max_idx, num_sample=5):
    sample_idx = np.random.choice(range(linspace), num_sample)
    # sample_idx = np.array(random.sample(range(linspace), num_sample))

    # print(sample_idx)
    # print(len(list(range(0, max_idx, linspace))))
    # print(np.sort(random.sample(list(np.linspace(0, max_idx, max_idx//num_sample ,endpoint=False,dtype=int)),num_sample)))
    sample_idx += np.sort(random.sample(list(range(0, max_idx, linspace)),num_sample))
    return sample_idx

def Triuvec(x,no_diag = False):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.ones(dim, dim).triu()
    if no_diag:
        I -= torch.eye(dim,dim)
    I  = I.reshape(dim * dim)
    # I = torch.eye(dim,dim).reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    # y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    y = r[:, index].squeeze()
    return y

def Triumap(x,no_diag = False):

    batchSize, dim, dim, h, w = x.shape
    r = x.reshape(batchSize, dim * dim, h, w)
    I = torch.ones(dim, dim).triu()
    if no_diag:
        I -= torch.eye(dim,dim)
    I  = I.reshape(dim * dim)
    # I = torch.eye(dim,dim).reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    # y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    y = r[:, index, :, :].squeeze()
    return y

def Diagvec(x):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.eye(dim, dim).triu().reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    y = r[:, index].squeeze()
    return y

class reconstruct_layer(nn.Module):
    def __init__(self, in_channels = 128, out_channels=128, p_des=0.2, p_drop = 0.5,skip_connect=True):
        super(reconstruct_layer, self).__init__()
        # init : p_des = 0.2 p_drop=0.6
        # self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1, bias=False)
        # self.conv_soft = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=1, padding=1, bias=False)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.conv_soft = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,  bias=True)
        self.conv_rec = nn.Sequential(

            # nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

        )
        self.skip_connect = skip_connect
        # mini 和 tiered 1shot 有更好的表现
        # p_des = 0.5
        # p_drop = 0.5
        # ge:
        # p_des = 0.5
        # p_drop = 0.9
        # drop2d : 0.5对大多任务效果好
        self.drop_des2 = nn.Dropout2d(p_des)
        self.drop_des = nn.Dropout(p_des)
        self.p_drop = p_drop
        self.scale = torch.nn.Parameter(torch.ones(in_channels),requires_grad=True)
    #     初始化权重：
    #     print(self.conv_soft.weight.data.shape)
    #     self.conv_soft.weight.data[:,:,0,0] = torch.eye(in_channels)

    def forward(self, x):
        mask = torch.ones((x.shape[0], x.shape[1])).to(x.device)
        # if random.random() <= self.p_drop:
        #     # map_mask = (self.drop_des(mask)/(mask)).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        #     map_mask = (self.drop_des(mask)).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        #
        # else:
        #     map_mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        # print(self.conv.weight.shape)
        # map_mask = self.drop_des2(mask)
        w_soft = F.softmax(self.conv_soft.weight.data)
        # w_soft = F.sigmoid(self.conv_soft.weight.data)
        # w_soft = self.conv_soft.weight.data
        # w_soft = w_soft.clamp(0,1)

        # w_soft = self.conv_soft.weight.data
        # w_soft_nrom = torch.norm(w_soft, p=2, dim=1).unsqueeze(1).expand_as(w_soft)
        # w_soft = w_soft.div(w_soft_nrom + 1e-9)
        self.conv_soft.weight.data = w_soft

        # self.conv_soft.weight.data = F.sigmoid(self.conv.weight.data,)
        # self.conv_soft.weight.data.clamp_(-.5,.5)
        # x_mask = x*map_mask
        x_mask = self.drop_des2(x)
        out = self.conv_soft(x_mask)
        # out = self.conv_rec(out)
        if self.skip_connect:
            self.scale.data.clamp_(0.5,2)
            out = torch.cat([out.unsqueeze(-1), x.unsqueeze(-1)],dim=-1)
            # print(out.shape)
            # return out
            # return torch.max(out,dim=-1)[0]
            # return torch.sum(out, dim=-1)
            out = torch.mean(out,dim=-1)
            # out = self.conv_rec(out)
            return out
            # return torch.mean(out, dim=-1) * self.scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)


        else:
            return out

class fusion_module(nn.Module):
    def __init__(self,in_channels=2,out_channels=1,dim=8256):
        super(fusion_module, self).__init__()
        self.conv =  nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.conv_soft =  nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.fusion = nn.Sequential(
            # nn.BatchNorm1d(out_channels),
            # nn.ReLU(inplace=True),

        )
        self.weight =  torch.nn.Parameter(torch.randn(dim),requires_grad=True)

    def _init_weight(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv1d):
                nn.init.uniform_(m.weight, a=0, b=1)
                # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')


    def forward(self,x):
        # print(self.conv_soft.weight.data.shape)
        # w_soft = F.softmax(self.conv.weight.data,dim=1)
        # self.conv_soft.weight.data = w_soft
        # x = self.conv_soft(x)
        # print(x.shape)
        weight = self.weight.unsqueeze(0).expand(x.shape[0],x.shape[-1])
        x = F.sigmoid(weight) * x[:,0,:] + (1-F.sigmoid(weight)) * x[:,1,:]

        return self.fusion(x)

class Rec_Net(nn.Module):
    def __init__(self,reduce_dim=128):
        super(Rec_Net, self).__init__()
        self.rec_layer = reconstruct_layer(in_channels=reduce_dim, out_channels=reduce_dim, ).cuda()
        self.SFC = nn.Linear(reduce_dim, 5).cuda()
        self.drop = nn.Dropout(0.5)
        self.embeding_way='GE'
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self,sample_support,sample_support_cons,is_query=False):
        rec_map = self.rec_layer(sample_support)
        rec_map_cons = self.rec_layer(sample_support_cons)
        # ==============================

        if self.embeding_way in ['BDC']:
            BDC_rec = self.dcov(rec_map)
            BDC_rec_cons = self.dcov(rec_map_cons)

        else:
            BDC_rec = self.avg_pool(rec_map).view(sample_support.shape[0], -1)
            BDC_rec_cons = self.avg_pool(rec_map_cons).view(sample_support.shape[0], -1)

        spt_norm = torch.norm(BDC_rec, p=2, dim=1).unsqueeze(1).expand_as(BDC_rec)
        BDC_norm = BDC_rec.div(spt_norm + 1e-6)

        spt_norm = torch.norm(BDC_rec_cons, p=2, dim=1).unsqueeze(1).expand_as(BDC_rec_cons)
        BDC_norm_cons = BDC_rec_cons.div(spt_norm + 1e-6)
        if is_query:

            BDC_x = 0.5 * BDC_norm + 0.5 * (
                        1 / 4 * torch.sum(BDC_norm_cons.view(BDC_norm.shape[0],-1,BDC_norm.shape[-1]), dim=1))

        else:
            BDC_x = (BDC_norm + BDC_norm_cons) / 2
            BDC_x = self.drop(BDC_x)
        out = self.SFC(BDC_x)
        return out


class Net_rec(nn.Module):
    def __init__(self,params,num_classes = 5,):
        super(Net_rec, self).__init__()

        self.params = params
        self.out_map = False
        resnet_layer_dim = [64, 160, 320, 640]

        if params.model == 'resnet12':
            self.backbone = resnet12(avg_pool=True,num_classes=64)
        elif params.model == 'resnet18':
            self.backbone = ResNet18()
            resnet_layer_dim = [64, 128, 256, 512]
        elif params.model == 'resnet34':
            self.backbone = ResNet34(flatten=False)
            resnet_layer_dim = [64, 128, 256, 512]
        elif params.model == 'conv64':
            self.backbone = ConvNet()
            resnet_layer_dim = self.backbone.feat_dim

        self.resnet_layer_dim = resnet_layer_dim
        # if params.metric == 'DN4':
        #     self.avg_pool = nn.AdaptiveAvgPool2d(3)
        #     self.imgtoclass = ImgtoClass_Metric(neighbor_k=1)
        #     self.out_map = True

        self.reduce_dim = params.reduce_dim
        self.feat_dim = self.backbone.feat_dim
        self.dim = int(self.reduce_dim * (self.reduce_dim+1)/2)
        # print(resnet_layer_dim[-1])
        # print(self.reduce_dim)
        if resnet_layer_dim[-1] != self.reduce_dim:
            self.Conv = nn.Sequential(
                nn.Conv2d(resnet_layer_dim[-1], self.reduce_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.reduce_dim),
                nn.ReLU(inplace=True)
            )
            self._init_weight(self.Conv.modules())

        drop_rate = params.drop_rate
        if self.params.embeding_way in ['BDC']:
            self.SFC = nn.Linear(self.dim, num_classes)
            self.SFC.bias.data.fill_(0)
        elif self.params.embeding_way in ['baseline++']:
            self.SFC = nn.Linear(self.reduce_dim, num_classes,bias=False)
            WeightNorm.apply(self.SFC, 'weight', dim=0)
        else:
            self.SFC = nn.Linear(self.reduce_dim, num_classes)

        self.drop = nn.Dropout(drop_rate)

        self.temperature = nn.Parameter(torch.log((1. /(2 * self.feat_dim[1] * self.feat_dim[2])* torch.ones(1, 1))),
                                            requires_grad=True)

        self.dcov = BDC(is_vec=True, input_dim=[self.reduce_dim,self.backbone.feat_dim[1],self.backbone.feat_dim[2]], dimension_reduction=self.reduce_dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if not self.params.my_model :
            self.feature = self.backbone
            if resnet_layer_dim[-1] != self.reduce_dim:
                self.dcov.conv_dr_block = self.Conv
        # self.fpn_alpha = nn.Parameter(2.0 * torch.ones(self.dim), requires_grad=True)

        # self.Conv_2 = nn.Sequential(
        #     nn.Conv2d(self.reduce_dim, 2048, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(inplace=True)
        # )
        # self.classifier = nn.Linear(self.reduce_dim, num_classes)


    def _init_weight(self,modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self):
        pass

    def normalize(self,x):
        x = (x - torch.mean(x, dim=1).unsqueeze(1))
        return x

    def comp_relation(self,feat_map,out_map =False):
    #     batch * feat_dim * feat_map
        batchSize, dim, h, w = feat_map.shape
        feat_map = feat_map.view(batchSize, dim, -1)
        # feat_map= self.drop_confusion(feat_map)
        feat_map_1 = feat_map.unsqueeze(-2).repeat(1,1,feat_map.shape[1],1)
        feat_map_2 = feat_map.unsqueeze(-2).permute(0,2,1,3).repeat(1,feat_map.shape[1],1,1)

        if self.params.lego :
            item_1 = torch.abs(feat_map_1 + feat_map_2)
            item_2 = torch.abs(feat_map_1 - feat_map_2)
            out = (item_1 - item_2) / 2
            out = out.reshape(batchSize, dim * dim)
            I = torch.ones(dim, dim).triu().reshape(dim * dim)
            index = I.nonzero(as_tuple=False)
            feat_map = out[:, index,:]
            feat_map_1 = feat_map.unsqueeze(-2).repeat(1, 1, feat_map.shape[1], 1)
            feat_map_2 = feat_map.unsqueeze(-2).permute(0, 2, 1, 3).repeat(1, feat_map.shape[1], 1, 1)
            item_1 = torch.sum(torch.abs(feat_map_1 + feat_map_2), dim=-1)
            item_2 = torch.sum(torch.abs(feat_map_1 - feat_map_2), dim=-1)
            out = (item_1 - item_2) * torch.exp(self.temperature)

        else:

            item_1 = torch.sum(torch.abs(feat_map_1 + feat_map_2), dim=-1)
            item_2 = torch.sum(torch.abs(feat_map_1 - feat_map_2), dim=-1)

            out = (item_1 - item_2)/2 * torch.exp(self.temperature)


        # out = torch.clamp(out, min=1e-8)

        # ==================================
        if self.params.normalize_bdc:
            I_M = torch.ones(batchSize, dim, dim, device=feat_map.device).type(feat_map.dtype)
            # out = out - 1. / dim * out.bmm(I_M) - 1. / dim * I_M.bmm(out) + 1. / (dim * dim) * I_M.bmm(out).bmm(I_M)
            out = out - 1. / dim * out.bmm(I_M) - 1. / dim * I_M.bmm(out)

        if out_map:
            out = Triumap(out.reshape(batchSize,dim,dim,self.feat_dim[-2],self.feat_dim[-1]), no_diag=self.params.no_diag)
            out = self.avg_pool(out)
        else:
            out = Triuvec(out,no_diag=self.params.no_diag)

        if self.params.normalize_feat:
            out = self.normalize(out)

        return out

    def forward_feature(self, x, confusion=False, out_map=False):
        feat_map = self.backbone(x, is_FPN=(self.params.FPN_list is not None))
        # print(feat_map.shape)
        if self.resnet_layer_dim[-1] != self.reduce_dim:
            feat_map = self.Conv(feat_map)
        if self.params.LR :
            if self.params.embeding_way in ['BDC'] :
                out = self.dcov(feat_map)
            else:
                x = self.avg_pool(feat_map)
                out = x.view(x.shape[0], -1)
        else:
            out = feat_map
        return out

    def normalize_feature(self, x):
        if self.params.norm == 'center':
            x = x - x.mean(2).unsqueeze(2)
            return x
        else:
            return x

    def forward_feat(self,x,x_c, confusion=False,):
        feat_map = self.backbone(x, is_FPN=(self.params.FPN_list is not None))
        feat_map_cl = self.backbone(x_c, is_FPN=(self.params.FPN_list is not None))

        feat_map_1 = self.Conv(feat_map)
        feat_map_2 = self.Conv(feat_map_cl)

        map_mask = torch.ones((feat_map.shape[0], feat_map.shape[1])).cuda()
        # swap_idx = self.drop_swap(map_mask)/(map_mask)
        swap_idx = map_mask / map_mask
        swap_idx = swap_idx.unsqueeze(-1).unsqueeze(-1).expand_as(feat_map)
        # swap_idx = feat_map/(feat_map + 1e-6)
        feat_map_1 = feat_map_1 * swap_idx + feat_map_2 * (1 - swap_idx)
        feat_map_2 = feat_map_1 * (1 - swap_idx) + feat_map_2 * swap_idx

        out_confusion = self.avg_pool(feat_map).view(feat_map.shape[0], -1)
        BDC_1 = self.dcov(feat_map_1)
        BDC_2 = self.dcov(feat_map_2)


        out_1 = self.SFC(self.drop(BDC_1))
        out_2 = self.SFC(self.drop(BDC_2))


        return BDC_1, BDC_2, out_1, out_2

    def forward_pretrain(self, x, confusion = False):
        x = self.forward_feature(x,confusion=confusion,out_map=False)

        # x = self.comp_relation(x)
        if self.params.embeding_way in ['baseline++']:
            x = self.avg_pool(x)
            x = x.view(x.shape[0], -1)
            x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
            x = x.div(x_norm + 0.00001)
        else:
            x = self.drop(x)
        return self.SFC(x)

    def train_loop(self,epoch,train_loader,optimizers):
        print_step = 100
        avg_loss = 0
        [optimizer , optimizer_ad] = optimizers
        total_correct = 0
        iter_num = len(train_loader)
        total = 0
        loss_ce_fn = nn.CrossEntropyLoss()
        for i ,data in enumerate(train_loader):
            image , label = data
            image = image.cuda()
            label = label.cuda()
            out = self.forward_pretrain(image)
            loss =  loss_ce_fn(out, label)
            avg_loss = avg_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = torch.max(out, 1)
            correct = (pred == label).sum().item()
            total_correct += correct
            total += label.size(0)
            if i % print_step == 0:
                print('\rEpoch {:d} | Batch: {:d}/{:d} | Loss: {:.4f} | Acc_train: {:.2f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1),correct/label.shape[0]*100), end=' ')
        print()

        # print('k between {:d} and {:d}'.format(min(k_rocord),max(k_rocord),))
        # print('k between {:d} and {:d}'.format(min(k_rocord),max(k_rocord),))
        # print(Counter(k_rocord).most_common(4))
        return avg_loss / iter_num, float(total_correct) / total * 100

    def meta_val_loop(self,epoch,val_loader,classifier='emd'):
        acc = []
        classifier = 'emd' if not self.params.test_LR else 'LR'
        for i, data in enumerate(val_loader):
            tic = time.time()
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            split_size = 128
            if support_xs.squeeze(0).shape[0] >= split_size:
                feat_sup_ = []
                # print(support_xs.shape)
                for j in range(math.ceil(support_xs.squeeze(0).shape[0] / split_size)):
                    # print(support_xs.squeeze(0)[i*128:min((i+1)*128,support_xs.shape[1]),:,:,:].shape)
                    fest_sup_item = self.forward_feature(
                        support_xs.squeeze(0)[j * split_size:min((j + 1) * split_size, support_xs.shape[1]), :, :, :],
                        out_map=self.out_map)
                    feat_sup_.append(fest_sup_item if len(fest_sup_item.shape) >= 1 else fest_sup_item.unsqueeze(0))
                feat_sup = torch.cat(feat_sup_, dim=0)
            else:
                feat_sup = self.forward_feature(support_xs.squeeze(0), out_map=self.out_map)
            if query_xs.squeeze(0).shape[0] > split_size:
                feat_qry_ = []
                # print(support_xs.shape)
                for j in range(math.ceil(query_xs.squeeze(0).shape[0] / split_size)):
                    # print(support_xs.squeeze(0)[i*128:min((i+1)*128,support_xs.shape[1]),:,:,:].shape)
                    feat_qry_item = self.forward_feature(
                        query_xs.squeeze(0)[j * split_size:min((j + 1) * split_size, query_xs.shape[1]), :, :, :],
                        out_map=self.out_map)
                    feat_qry_.append(feat_qry_item if len(feat_qry_item.shape) > 1 else feat_qry_item.unsqueeze(0))

                feat_qry = torch.cat(feat_qry_, dim=0)
            else:
                feat_qry = self.forward_feature(query_xs.squeeze(0), out_map=self.out_map)
            if self.params.LR:
                pred = self.LR(feat_sup, support_ys, feat_qry, query_ys)
            else:
                with torch.enable_grad():
                    pred = self.softmax(feat_sup, support_ys, feat_qry, )
                    _, pred = torch.max(pred, dim=-1)
            if self.params.n_symmetry_aug > 1:
                # pred = pred.view(-1, self.params.n_symmetry_aug)
                query_ys = query_ys.view(-1, self.params.n_symmetry_aug)
                # pred = torch.mode(pred,dim=-1)[0]
                query_ys = torch.mode(query_ys, dim=-1)[0]

            # print(np.mean(pred.cpu().numpy() == query_ys.numpy()))
            acc_epo = np.mean(pred.cpu().numpy() == query_ys.numpy())
            # if acc_epo<0.78:
            #     acc_task = []
            #     recall = []
            #     for i in range(self.params.n_way):
            #         acc_task.append(np.mean((pred.cpu().numpy()==i) & (query_ys.numpy()==i))*self.params.n_way)
            #         recall.append(np.mean((pred.cpu().numpy() == i) & (query_ys.numpy() != pred.cpu().numpy())) * self.params.n_way)
            #     print(acc_task,recall)
            #     time.sleep(100)
            acc.append(acc_epo)
            # print("\repisode {} acc: {:.2f} | avg_acc: {:.2f} +- {:.2f}, elapse : {:.2f}".format(i, acc_epo * 100,
            #                                                                                      *mean_confidence_interval(
            #                                                                                          acc, multi=100), (
            #                                                                                              time.time() - tic) / 60),
            #       end='')
        return mean_confidence_interval(acc)

    def meta_test_loop(self,test_loader):
        acc = []
        classifier = 'emd' if not self.params.test_LR else 'LR'
        for i, data in enumerate(test_loader):
            tic = time.time()
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()

            # print(query_xs.shape)
            split_size = 256
            if support_xs.squeeze(0).shape[0] >= split_size:
                feat_sup_ = []
                # print(support_xs.shape)
                for j in range(math.ceil(support_xs.squeeze(0).shape[0]/split_size)):
                    # print(support_xs.squeeze(0)[i*128:min((i+1)*128,support_xs.shape[1]),:,:,:].shape)
                    fest_sup_item =self.forward_feature(support_xs.squeeze(0)[j*split_size:min((j+1)*split_size,support_xs.shape[1]),:,:,:],out_map=self.out_map)
                    feat_sup_.append(fest_sup_item if len(fest_sup_item.shape)>=1 else fest_sup_item.unsqueeze(0))
                feat_sup = torch.cat(feat_sup_,dim=0)
            else:
                feat_sup = self.forward_feature(support_xs.squeeze(0),out_map=self.out_map)
            if query_xs.squeeze(0).shape[0] >= split_size:
                feat_qry_ = []
                # print(support_xs.shape)
                for j in range(math.ceil(query_xs.squeeze(0).shape[0]/split_size)):
                    # print(support_xs.squeeze(0)[i*128:min((i+1)*128,support_xs.shape[1]),:,:,:].shape)
                    feat_qry_item = self.forward_feature(
                        query_xs.squeeze(0)[j * split_size:min((j + 1) * split_size, query_xs.shape[1]), :, :, :],out_map=self.out_map)
                    feat_qry_.append(feat_qry_item if len(feat_qry_item.shape) > 1 else feat_qry_item.unsqueeze(0))

                feat_qry = torch.cat(feat_qry_,dim=0)
            else:
                feat_qry = self.forward_feature(query_xs.squeeze(0),out_map=self.out_map)

            if self.params.LR:
                pred = self.LR(feat_sup, support_ys, feat_qry, query_ys)
            else:
                with torch.enable_grad():
                    pred = self.softmax(feat_sup, support_ys, feat_qry,)
                    if not self.params.LR_rec:
                        _,pred = torch.max(pred,dim=-1)
            if self.params.n_symmetry_aug > 1:
                # pred = pred.view(-1, self.params.n_symmetry_aug)
                query_ys = query_ys.view(-1, self.params.n_symmetry_aug)
                # pred = torch.mode(pred,dim=-1)[0]
                query_ys = torch.mode(query_ys, dim=-1)[0]

            # print(np.mean(pred.cpu().numpy() == query_ys.numpy()))
            acc_epo = np.mean(pred.cpu().numpy() == query_ys.numpy())
            # if acc_epo<0.78:
            #     acc_task = []
            #     recall = []
            #     for i in range(self.params.n_way):
            #         acc_task.append(np.mean((pred.cpu().numpy()==i) & (query_ys.numpy()==i))*self.params.n_way)
            #         recall.append(np.mean((pred.cpu().numpy() == i) & (query_ys.numpy() != pred.cpu().numpy())) * self.params.n_way)
            #     print(acc_task,recall)
            #     time.sleep(100)
            acc.append(acc_epo)
            print("\repisode {} acc: {:.2f} | avg_acc: {:.2f} +- {:.2f}, elapse : {:.2f}".format(i, acc_epo * 100,
                                                                                                 *mean_confidence_interval(
                                                                                                     acc, multi=100), (
                                                                                                             time.time() - tic) / 60),
                  end='')
        return mean_confidence_interval(acc)

    def distillation(self,epoch,train_loader,optimizer,model_t):
        print_step = 100
        avg_loss = 0
        total_correct = 0
        iter_num = len(train_loader)
        total = 0
        loss_div_fn = DistillKL(4)
        loss_ce_fn = nn.CrossEntropyLoss()
        # model_t.eval()
        for i, data in enumerate(train_loader):
            image, label = data
            image = image.cuda()
            label = label.cuda()
            with torch.no_grad():
                out_t = model_t.forward_pretrain(image)

            out= self.forward_pretrain(image)
            loss_ce = loss_ce_fn(out, label)
            loss_div = loss_div_fn(out, out_t)

            loss  = loss_ce * 0.5 + loss_div * 0.5
            avg_loss = avg_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(out, 1)
            correct = (pred == label).sum().item()
            total_correct += correct
            total += label.size(0)
            if i % print_step == 0:
                print('\rEpoch {:d} | Batch: {:d}/{:d} | Loss: {:.4f} | Acc_train: {:.2f}'.format(epoch, i,
                                                                                                  len(train_loader),
                                                                                                  avg_loss / float(
                                                                                                      i + 1),
                                                                                                  correct / label.shape[
                                                                                                      0] * 100),
                      end=' ')
        print()
        return avg_loss / iter_num, float(total_correct) / total * 100


    def LR_rec(self,support_z,support_y):
        clf = LR(penalty='l2',
                 random_state=0,
                 C=self.params.penalty_c,
                 solver='lbfgs',
                 max_iter=1000,
                 multi_class='multinomial')
        # spt_norm = torch.norm(support_z, p=2, dim=1).unsqueeze(1).expand_as(support_z)
        # spt_normalized = support_z.div(spt_norm + 1e-6)

        z_support = support_z.detach().cpu().numpy()
        y_support = support_y.reshape(-1).cpu().numpy()

        clf.fit(z_support, y_support)
        return clf

    def softmax(self,support_z,support_ys,query_z,):
        # proto : K * D
        prototype = torch.zeros((self.params.n_way, self.dim)).cuda()
        # prototype = torch.zeros((support_z.shape[0]//self.params.n_aug_support_samples, self.dim)).cuda()
        loss_ce_fn = nn.CrossEntropyLoss()
        lr_scheduler = None
        support_ys = support_ys.cuda()
        drop2 = nn.Dropout(0.3).cuda()

        if self.params.embeding_way in ['BDC']:
            rec_layer = reconstruct_layer().cuda()
            SFC = nn.Linear(self.dim, self.params.n_way).cuda()
            fusion =fusion_module(dim=self.dim).cuda()
            if self.params.optim in ['Adam']:
                optimizer = torch.optim.Adam([{'params':fusion.parameters()},{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=self.params.lr, weight_decay=self.params.wd_test)
                iter_num = 100
            else:
                # best
                # optimizer = torch.optim.SGD([{'params':fusion.parameters(),'lr':self.params.lr},{'params': SFC.parameters()}],lr=self.params.lr, momentum=0.9,  weight_decay=self.params.wd_test)
                optimizer = torch.optim.SGD([{'params':fusion.parameters(),'lr':self.params.lr},{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=self.params.lr, momentum=0.9, nesterov=True, weight_decay=self.params.wd_test)
                # 1shot : 69.20+-    5shot 85.73
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,150], gamma=0.1)
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1)
                iter_num = 180
        else:

            rec_layer = reconstruct_layer(in_channels=self.reduce_dim, out_channels=self.reduce_dim,).cuda()
            if self.params.embeding_way in ['baseline++']:
                SFC = nn.Linear(self.reduce_dim, self.params.n_way, bias=False).cuda()
                WeightNorm.apply(SFC, 'weight', dim=0)
            else:
                SFC = nn.Linear(self.reduce_dim, self.params.n_way).cuda()
            fusion =fusion_module(dim=self.reduce_dim).cuda()
            if self.params.optim in ['Adam']:
                # lr = 5e-3
                optimizer = torch.optim.Adam([{'params':fusion.parameters()},{'params': rec_layer.parameters()}, {'params': SFC.parameters()}],
                                             lr=self.params.lr, weight_decay=self.params.wd_test)
                # optimizer = torch.optim.Adam([{'params': SFC.parameters()}],lr=1e-3, weight_decay=5e-4)
                #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,], gamma=0.5)
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, ], gamma=0.2)
                # iter_num = 150

                optimizer = torch.optim.Adam([{'params': rec_layer.parameters()}, {'params': SFC.parameters()}],
                                             lr=self.params.lr, weight_decay=self.params.wd_test)
                iter_num = 100
            elif self.params.embeding_way in ['baseline++']:
                optimizer = torch.optim.Adam([{'params': SFC.parameters()}],
                                            lr=0.001, weight_decay=self.params.wd_test)
            else:
                optimizer = torch.optim.SGD([{'params':fusion.parameters()},{'params': rec_layer.parameters(),}, {'params': SFC.parameters()}],
                                            lr=self.params.lr, momentum=0.9, nesterov=True,
                                            weight_decay=self.params.wd_test)
                # optimizer = torch.optim.SGD([{'params': rec_layer.parameters(), }, {'params': SFC.parameters()}],
                #                             lr=self.params.lr, momentum=0.9,
                #                             weight_decay=self.params.wd_test)
                # optimizer = torch.optim.LBFGS([{'params': rec_layer.parameters(), }, {'params': SFC.parameters()}],lr=self.params.lr,
                #                               )
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,160,240], gamma=0.1)

                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160, 240, 360],
                #                                                     gamma=0.1)
                # iter_num = 450
                iter_num = 300
                # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=iter_num,eta_min=1e-4)

                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 150], gamma=0.1)
                iter_num = 180

                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1)
                # iter_num = 150

                # 1shot : 69.20+-    5shot 85.73
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 150], gamma=0.1)
                # iter_num = 180
                # 62.05 77.54
                # 62.05 77.54
                # optimizer = torch.optim.SGD([{'params': rec_layer.parameters(), 'lr': 2}, {'params': SFC.parameters()}],
                #                             lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
                # # optimizer = torch.optim.SGD([{'params': rec_layer.parameters(),}, {'params': SFC.parameters()}],lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
                # # optimizer = torch.optim.SGD([{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
                # # 1-shot pure : 62.57
                # optimizer = torch.optim.SGD([{'params':rec_layer.parameters(),'lr':5e-1},{'params': SFC.parameters()}],lr=self.params.lr, momentum=0.9, nesterov=True, weight_decay=5e-2)
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,400,600,800], gamma=0.1)
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 240], gamma=0.2)
                #
                # iter_num = 360
            # rec_layer = reconstruct_layer().cuda()
            # SFC = nn.Linear(self.dim, self.params.n_way).cuda()

            # self.drop = nn.Dropout(0.6)
            # Good Embedding
            # 62.4
            # optimizer = torch.optim.Adam([{'params':rec_layer.parameters(),'lr':5e-2},{'params': SFC.parameters()}],lr=4e-3, weight_decay=1e-4,eps=1e-5)
            # optimizer = torch.optim.Adam([{'params':rec_layer.parameters(),'lr': 1e-3},{'params': SFC.parameters()}],lr=1e-3, weight_decay=1e-4)
            # optimizer = torch.optim.Adam([{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=5e-3, weight_decay=5e-4)

        # loss_rec = 0
        rec_layer.train()
        SFC.train()
        batch_fsl =min(16,support_z.shape[0])
        no_stym_aug = False
        sam_local = True
        local_num = 5

        for i in range(iter_num):
            # sample_idxs = np.random.choice(range(support_z.shape[0]),min(64, support_z.shape[0]))
            # sample_idxs = random_sample(self.params.n_aug_support_samples,support_z.shape[0],self.params.n_way*self.params.n_aug_support_samples)
            # sample_idxs = random_sample(self.params.n_aug_support_samples,support_z.shape[0],min(25, self.params.n_way*self.params.n_shot))

            sample_idxs = range(0,support_z.shape[0],self.params.n_aug_support_samples)
            if sam_local:
                sample_idxs_cons = []
                for s in range(local_num):
                    sample_cons_part = torch.tensor(random_sample(self.params.n_aug_support_samples, support_z.shape[0],
                                                     support_z.shape[0] // self.params.n_aug_support_samples))
                    sample_idxs_cons.append(sample_cons_part.unsqueeze(0))
                sample_idxs_cons = torch.cat(sample_idxs_cons).view(-1)
            else:
                sample_idxs_cons = random_sample(self.params.n_aug_support_samples,support_z.shape[0],support_z.shape[0]//self.params.n_aug_support_samples)
            #
            # sample_idxs = np.random.choice(range(support_z.shape[0] // self.params.n_aug_support_samples),
            #                                batch_fsl) * self.params.n_aug_support_samples
            # sample_idxs_cons = sample_idxs + np.random.choice(range(self.params.n_aug_support_samples),batch_fsl)

            sample_support = support_z[sample_idxs, :, :, :]
            sample_support_cons = support_z[sample_idxs_cons, :, :, :]
            sample_label = support_ys[:, sample_idxs]

            rec_map = rec_layer(sample_support)
            rec_map_cons = rec_layer(sample_support_cons)
            # ==============================

            if self.params.embeding_way in ['BDC']:
                BDC_ori = self.dcov(sample_support)
                BDC_ori_cons = self.dcov(sample_support_cons)
                BDC_rec = self.dcov(rec_map)
                BDC_rec_cons = self.dcov(rec_map_cons)

            else:
                BDC_ori = self.avg_pool(sample_support).view(sample_support.shape[0],-1)
                BDC_ori_cons = self.avg_pool(sample_support_cons).view(sample_support_cons.shape[0],-1)
                BDC_rec = self.avg_pool(rec_map).view(sample_support.shape[0],-1)
                BDC_rec_cons = self.avg_pool(rec_map_cons).view(sample_support.shape[0],-1)

            # BDC_ori = self.comp_relation(sample_support)
            spt_norm = torch.norm(BDC_ori, p=2, dim=1).unsqueeze(1).expand_as(BDC_ori)
            BDC_ori = BDC_ori.div(spt_norm + 1e-6 )
            #
            spt_norm_cons = torch.norm(BDC_ori_cons, p=2, dim=1).unsqueeze(1).expand_as(BDC_ori_cons)
            BDC_ori_cons = BDC_ori_cons.div(spt_norm_cons + 1e-6)
            #
            # # if not self.params.LR_rec:
            spt_norm = torch.norm(BDC_rec, p=2, dim=1).unsqueeze(1).expand_as(BDC_rec)
            BDC_rec = BDC_rec.div(spt_norm + 1e-6)
            #
            spt_norm = torch.norm(BDC_rec_cons, p=2, dim=1).unsqueeze(1).expand_as(BDC_rec_cons)
            BDC_rec_cons = BDC_rec_cons.div(spt_norm + 1e-6)

            if sam_local:
                # print(BDC_ori_cons.shape)
                BDC_ori_cons = torch.mean(BDC_ori_cons.view(local_num,-1,*BDC_ori_cons.size()[1:]),dim=0)

            if self.params.n_aug_support_samples==1:
                BDC_x = BDC_ori
            else:
                BDC_x = (BDC_ori_cons + BDC_ori) / 2
            if np.random.rand() <= self.params.drop_few:
                # pass
                BDC_x = BDC_ori_cons

            # BDC_x_norm = torch.norm(BDC_x, p=2, dim=1).unsqueeze(1).expand_as(BDC_x)
            # BDC_x = BDC_x.div(BDC_x_norm + 1e-6)

            # if self.params.ablation % 2 == 1:
            #     BDC_x = BDC_ori_cons

            BDC_x = self.drop(BDC_x)
            out = SFC(BDC_x)
            # out = SFC(BDC_rec)
            if no_stym_aug:
                out = out.reshape(BDC_rec.shape[0], self.params.n_way, -1)
                # out = torch.max(out,dim=1)[0]
                # out = out - torch.mean(out,dim=1).unsqueeze(1) - torch.mean(out,dim=2).unsqueeze(2) + torch.mean(torch.mean(out,dim=-1),dim=-1).unsqueeze(-1).unsqueeze(-1)
                out =  out[:, range(self.params.n_way), range(self.params.n_way)]

            loss_ce = loss_ce_fn(out,sample_label.squeeze(0))


            loss = loss_ce
            # loss = loss_rec
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(fusion.weight.data[0])
            if lr_scheduler is not None:
                lr_scheduler.step()
            # print('loss_ce: {:.2f} \t loss_mse: {:.2f}'.format(loss_ce,loss_rec))

        rec_layer.eval()
        SFC.eval()
        fusion.eval()
        if self.params.LR_rec:
            support_z_rec = rec_layer(support_z)

            if self.params.embeding_way in ['BDC']:
                support_z_rec = self.dcov(support_z_rec)
            else:
                support_z_rec = self.avg_pool(support_z_rec)
            spt_norm = torch.norm(support_z_rec, p=2, dim=1).unsqueeze(1).expand_as(support_z_rec)
            support_z_rec = support_z_rec.div(spt_norm + 1e-6)
            support_z_rec = support_z_rec.reshape(support_z.shape[0]//self.params.n_aug_support_samples,self.params.n_aug_support_samples,-1)
            # support_ys = support_ys.view((support_z.shape[0]//self.params.n_aug_support_samples,-1))[:,0]
            support_ys = support_ys.reshape((support_z.shape[0]//self.params.n_aug_support_samples,-1))[:,1:]

            # support_z_rec = 0.5 * support_z_rec[:,0,:] + 0.5* torch.mean(support_z_rec[:,1:,:],dim=1)
            support_z_rec = 0.5 * support_z_rec[:,0,:].unsqueeze(1) + 0.5* support_z_rec[:,1:,:]
            # support_z_rec = support_z_rec[:,0,:]
            clf = self.LR_rec(
                support_z_rec.reshape(support_z_rec.shape[0] * (self.params.n_aug_support_samples - 1), -1),
                support_ys)
            # clf = self.LR_rec(
            #     support_z_rec,
            #     support_ys)

        with torch.no_grad():
            # print(query_z.shape)
            query_rec = rec_layer(query_z)
            # query_rec = query_z
            if self.params.embeding_way in ['BDC']:
                query_ori = self.dcov(query_z)
                query_rec = self.dcov(query_rec)
            else:
                query_ori = self.avg_pool(query_z)
                query_rec = self.avg_pool(query_rec)
                # query_rec = self.comp_relation(query_rec)

            # query_rec = self.dcov(query_z)
            # query_rec = self.comp_relation(query_rec)
            # spt_norm = torch.norm(query_rec, p=2, dim=1).unsqueeze(1).expand_as(query_rec)
            # query_rec = query_rec.div(spt_norm + 1e-6)

            spt_norm = torch.norm(query_ori, p=2, dim=1).unsqueeze(1).expand_as(query_ori)
            query_ori = query_ori.div(spt_norm + 1e-6)
            # print(query_rec.shape)
            # print(query_rec.shape)
            # query_rec = query_rec.view(query_rec.shape[0]//self.params.n_symmetry_aug, self.params.n_symmetry_aug, -1)
            query_rec = query_rec.view(query_rec.shape[0]//self.params.n_symmetry_aug, self.params.n_symmetry_aug, -1)
            query_ori = query_ori.view(query_ori.shape[0]//self.params.n_symmetry_aug, self.params.n_symmetry_aug, -1)

            # query_rec = 1 / (self.params.n_symmetry_aug) * torch.sum(query_rec, dim=1)
            if self.params.n_symmetry_aug>1:
                # query_rec = 1 / (self.params.n_symmetry_aug) * torch.sum(query_rec, dim=1)
                # query_rec = 0.5 * query_rec[:,0,:] + 0.5 * (1 / (self.params.n_symmetry_aug-1) * torch.sum(query_rec[:,1:,:], dim=1))
                # query_rec = 0.5 * query_ori[:,0,:] + 0.5 * (1 / (self.params.n_symmetry_aug-1) * torch.sum(query_rec[:,1:,:], dim=1))
                # query_rec = 0.5 * query_ori[:,0,:] + 0.5 * (1 / (self.params.n_symmetry_aug-1) * torch.sum(query_ori[:,1:self.params.n_symmetry_aug,:], dim=1))
                # query_cons = (1 / (self.params.n_symmetry_aug - 1) * torch.sum(query_ori[:, 1:self.params.n_symmetry_aug, :], dim=1))
                # query_cons = torch.max(query_ori[:, 1:self.params.n_symmetry_aug, :], dim=1)[0]
                query_cons = torch.mean(query_ori[:, 1:self.params.n_symmetry_aug, :], dim=1)

                query_rec = fusion(torch.cat([query_ori[:,0,:].unsqueeze(1),query_cons.unsqueeze(1)],dim=1)).view(query_ori.shape[0],-1)
                # if int(self.params.ablation/2) == 1 and self.params.ablation>1:
                #     # query_rec =  query_cons
                #     query_rec = query_cons
                # query_rec = 0.3*query_ori[:,0,:]+0.7*query_cons

            else:
                # query_rec =  query_rec[:, 0, :]
                query_rec = query_ori[:, 0, :]
            # spt_norm = torch.norm(query_rec, p=2, dim=1).unsqueeze(1).expand_as(query_rec)
            # query_rec = query_rec.div(spt_norm + 1e-6)
            # if no_stym_aug:
            #     # print(query_rec.shape)
            #     query_rec = (query_rec.unsqueeze(1).expand(query_rec.shape[0],prototype.shape[0],self.dim)+prototype.unsqueeze(0))/2
            if no_stym_aug:
                query_rec = (query_rec.unsqueeze(1).expand(query_rec.shape[0], prototype.shape[0],
                                                           self.dim) + prototype.unsqueeze(0)) / 2
            if self.params.LR_rec:

                # print(query_rec.shape)
                z_query = query_rec.detach().cpu().numpy()
                out = torch.from_numpy(clf.predict(z_query))
            else:
                out = SFC(query_rec)
                if no_stym_aug:
                    out = out.reshape(query_rec.shape[0],self.params.n_way,-1)
                    # out = torch.max(out,dim=1)[0]
                    # out = F.softmax(out,dim=-1)
                    return out[:,range(self.params.n_way),range(self.params.n_way)]
                    # print(out.shape)
        return out


    # def softmax(self,support_z,support_ys,query_z,):
    #     # proto : K * D
    #     prototype = torch.zeros((self.params.n_way, self.dim)).cuda()
    #     # prototype = torch.zeros((support_z.shape[0]//self.params.n_aug_support_samples, self.dim)).cuda()
    #
    #     loss_ce_fn = nn.CrossEntropyLoss()
    #     lr_scheduler = None
    #     # support_z = support_z.cpu().detach()
    #     # query_z = query_z.cpu().detach()
    #     # support_ys = support_ys.cpu().detach()
    #     support_ys = support_ys.cuda()
    #
    #     if self.params.embeding_way in ['BDC']:
    #         rec_layer = reconstruct_layer(p_des=self.params.drop_few).cuda()
    #         SFC = nn.Linear(self.dim, self.params.n_way).cuda()
    #         # rec_layer = reconstruct_layer().cpu()
    #         # SFC = nn.Linear(self.dim, self.params.n_way).cpu()
    #         # self.dcov = self.dcov.cpu()
    #         # SFC.bias.data.fill_(0)
    #         # 1shot : 0.01 + wd:0.05
    #         if self.params.optim in ['Adam']:
    #             # lr = 5e-3
    #             optimizer = torch.optim.Adam([{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=self.params.lr, weight_decay=self.params.wd_test)
    #         # optimizer = torch.optim.Adam([{'params': SFC.parameters()}],lr=1e-3, weight_decay=5e-4)
    #         #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,], gamma=0.5)
    #         #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,], gamma=0.1)
    #         #     iter_num = 200
    #
    #             optimizer = torch.optim.Adam([{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=self.params.lr, weight_decay=self.params.wd_test)
    #             iter_num = 100
    #         else:
    #             # best
    #             optimizer = torch.optim.SGD([{'params':rec_layer.parameters(),},{'params': SFC.parameters()}],lr=self.params.lr, momentum=0.9, nesterov=True, weight_decay=self.params.wd_test)
    #             # try:
    #             # optimizer = torch.optim.SGD([{'params':rec_layer.parameters(),},{'params': SFC.parameters()}],lr=self.params.lr, momentum=0.9, weight_decay=self.params.wd_test)
    #
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.2)
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,160,240,360], gamma=0.1)
    #             # iter_num = 450
    #
    #             # 1shot : 69.20+-    5shot 85.73
    #             lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,150], gamma=0.1)
    #             iter_num = 180
    #
    #             # 5shot try
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 180, 240], gamma=0.1)
    #             # iter_num = 300
    #     else:
    #         rec_layer = reconstruct_layer(in_channels=self.reduce_dim, out_channels=self.reduce_dim, p_drop=0.8,
    #                                       p_des=self.params.drop_few).cuda()
    #         rec_layer = reconstruct_layer(in_channels=self.reduce_dim, out_channels=self.reduce_dim,).cuda()
    #         SFC = nn.Linear(self.reduce_dim, self.params.n_way).cuda()
    #         if self.params.optim in ['Adam']:
    #             # lr = 5e-3
    #             optimizer = torch.optim.Adam([{'params': rec_layer.parameters()}, {'params': SFC.parameters()}],
    #                                          lr=self.params.lr, weight_decay=self.params.wd_test)
    #             # optimizer = torch.optim.Adam([{'params': SFC.parameters()}],lr=1e-3, weight_decay=5e-4)
    #             #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,], gamma=0.5)
    #             lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, ], gamma=0.2)
    #             iter_num = 150
    #
    #             # optimizer = torch.optim.Adam([{'params': rec_layer.parameters()}, {'params': SFC.parameters()}],
    #             #                              lr=self.params.lr, weight_decay=self.params.wd_test)
    #             # iter_num = 100
    #         else:
    #             optimizer = torch.optim.SGD([{'params': rec_layer.parameters(),}, {'params': SFC.parameters()}],
    #                                         lr=self.params.lr, momentum=0.9, nesterov=True,
    #                                         weight_decay=self.params.wd_test)
    #             # optimizer = torch.optim.SGD([{'params': rec_layer.parameters(), }, {'params': SFC.parameters()}],
    #             #                             lr=self.params.lr, momentum=0.9,
    #             #                             weight_decay=self.params.wd_test)
    #             # optimizer = torch.optim.LBFGS([{'params': rec_layer.parameters(), }, {'params': SFC.parameters()}],lr=self.params.lr,
    #             #                               )
    #             lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,160,240], gamma=0.1)
    #
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160, 240, 360],
    #             #                                                     gamma=0.1)
    #             # iter_num = 450
    #             iter_num = 300
    #             # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=iter_num,eta_min=1e-4)
    #
    #
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1)
    #             # iter_num = 150
    #
    #             # 1shot : 69.20+-    5shot 85.73
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 150], gamma=0.1)
    #             # iter_num = 180
    #             # 62.05 77.54
    #             # 62.05 77.54
    #             # optimizer = torch.optim.SGD([{'params': rec_layer.parameters(), 'lr': 2}, {'params': SFC.parameters()}],
    #             #                             lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
    #             # # optimizer = torch.optim.SGD([{'params': rec_layer.parameters(),}, {'params': SFC.parameters()}],lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
    #             # # optimizer = torch.optim.SGD([{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
    #             # # 1-shot pure : 62.57
    #             # optimizer = torch.optim.SGD([{'params':rec_layer.parameters(),'lr':5e-1},{'params': SFC.parameters()}],lr=self.params.lr, momentum=0.9, nesterov=True, weight_decay=5e-2)
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,400,600,800], gamma=0.1)
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 240], gamma=0.2)
    #             #
    #             # iter_num = 360
    #         # rec_layer = reconstruct_layer().cuda()
    #         # SFC = nn.Linear(self.dim, self.params.n_way).cuda()
    #
    #         # self.drop = nn.Dropout(0.6)
    #         # Good Embedding
    #         # 62.4
    #         # optimizer = torch.optim.Adam([{'params':rec_layer.parameters(),'lr':5e-2},{'params': SFC.parameters()}],lr=4e-3, weight_decay=1e-4,eps=1e-5)
    #         # optimizer = torch.optim.Adam([{'params':rec_layer.parameters(),'lr': 1e-3},{'params': SFC.parameters()}],lr=1e-3, weight_decay=1e-4)
    #         # optimizer = torch.optim.Adam([{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=5e-3, weight_decay=5e-4)
    #
    #
    #
    #     # loss_rec = 0
    #     rec_layer.train()
    #     SFC.train()
    #     batch_fsl =min(8,support_z.shape[0])
    #     no_stym_aug = False
    #
    #     for i in range(iter_num):
    #         # sample_idxs = np.random.choice(range(support_z.shape[0]),min(64, support_z.shape[0]))
    #         # sample_idxs = random_sample(self.params.n_aug_support_samples,support_z.shape[0],self.params.n_way*self.params.n_aug_support_samples)
    #         # sample_idxs = random_sample(self.params.n_aug_support_samples,support_z.shape[0],min(25, self.params.n_way*self.params.n_shot))
    #         print(support_z.shape)
    #         sample_idxs = range(0,support_z.shape[0],self.params.n_aug_support_samples)
    #         sample_idxs_cons = random_sample(self.params.n_aug_support_samples,support_z.shape[0],support_z.shape[0]//self.params.n_aug_support_samples)
    #         sample_support = support_z[sample_idxs, :, :, :]
    #         sample_support_cons = support_z[sample_idxs_cons, :, :, :]
    #         sample_label = support_ys[:, sample_idxs]
    #         # print(support_z.shape[0])
    #         # print(sample_idxs_cons.shape)
    #         # sample_idxs = np.random.choice(range(support_z.shape[0]//self.params.n_aug_support_samples),batch_fsl)*self.params.n_aug_support_samples
    #         # # print(sample_idxs.shape)
    #         # sample_idxs_cons = sample_idxs + np.random.choice(range(self.params.n_aug_support_samples),batch_fsl)
    #         # sample_support = support_z[sample_idxs, :, :, :]
    #         # sample_support_cons = support_z[sample_idxs_cons, :, :, :]
    #         # sample_label = support_ys[:, sample_idxs]
    #
    #
    #         # ===========================
    #         rec_map = rec_layer(sample_support)
    #         rec_map_cons = rec_layer(sample_support_cons)
    #         # ==============================
    #
    #         if self.params.embeding_way in ['BDC']:
    #             BDC_ori = self.dcov(sample_support)
    #             BDC_ori_cons = self.dcov(sample_support_cons)
    #             # print(rec_map.shape)
    #             BDC_rec = self.dcov(rec_map)
    #             BDC_rec_cons = self.dcov(rec_map_cons)
    #
    #         else:
    #             BDC_ori = self.avg_pool(sample_support).view(sample_support.shape[0],-1)
    #             BDC_ori_cons = self.avg_pool(sample_support_cons).view(sample_support.shape[0],-1)
    #             BDC_rec = self.avg_pool(rec_map).view(sample_support.shape[0],-1)
    #             BDC_rec_cons = self.avg_pool(rec_map_cons).view(sample_support.shape[0],-1)
    #
    #             # BDC_ori = self.comp_relation(sample_support).view(sample_support.shape[0], -1)
    #             # BDC_ori_cons = self.comp_relation(sample_support_cons).view(sample_support.shape[0], -1)
    #             # BDC_rec = self.comp_relation(rec_map).view(sample_support.shape[0], -1)
    #             # BDC_rec_cons = self.comp_relation(rec_map_cons).view(sample_support.shape[0], -1)
    #
    #         # BDC_ori = self.comp_relation(sample_support)
    #         spt_norm = torch.norm(BDC_ori, p=2, dim=1).unsqueeze(1).expand_as(BDC_ori)
    #         BDC_ori = BDC_ori.div(spt_norm + 1e-6 )
    #
    #         # # BDC_ori_cons = self.comp_relation(sample_support_cons)
    #         spt_norm_cons = torch.norm(BDC_ori_cons, p=2, dim=1).unsqueeze(1).expand_as(BDC_ori_cons)
    #         BDC_ori_cons = BDC_ori_cons.div(spt_norm_cons + 1e-6)
    #
    #         # if not self.params.LR_rec:
    #         spt_norm = torch.norm(BDC_rec, p=2, dim=1).unsqueeze(1).expand_as(BDC_rec)
    #         BDC_rec = BDC_rec.div(spt_norm + 1e-6)
    #
    #         spt_norm = torch.norm(BDC_rec_cons, p=2, dim=1).unsqueeze(1).expand_as(BDC_rec_cons)
    #         BDC_rec_cons = BDC_rec_cons.div(spt_norm + 1e-6)
    #
    #         # for p in range(self.params.n_way):
    #         prototype = 0.8 * prototype.detach() + 0.2 * torch.mean(BDC_rec_cons.reshape(self.params.n_shot,self.params.n_way,-1),dim=0)
    #
    #         # ==================================
    #         # BDC_x =self.drop((BDC_norm+BDC_norm_cons)/2)
    #         # ==================================
    #         # BDC_x = self.drop((BDC_ori + BDC_ori_cons)/2)
    #         # BDC_x = BDC_ori
    #         # print(BDC_norm_cons.shape)
    #         # print(BDC_norm.shape)
    #         # # BDC_x =(BDC_norm+BDC_norm_cons)/2
    #         BDC_x =(BDC_rec+BDC_rec_cons)/2
    #         # BDC_x = BDC_rec
    #         if no_stym_aug:
    #             # print(query_rec.shape)
    #             BDC_x = (BDC_rec.unsqueeze(1).expand(BDC_rec.shape[0],prototype.shape[0],self.dim)+prototype.unsqueeze(0))/2
    #         # BDC_x = (BDC_rec+prototype)/2
    #         # BDC_x_norm = torch.norm(BDC_x, p=2, dim=1).unsqueeze(1).expand_as(BDC_rec_cons)
    #         # BDC_x = BDC_rec_cons.div(BDC_x_norm + 1e-6)
    #
    #         BDC_x = self.drop(BDC_x)
    #         out = SFC(BDC_x)
    #         if no_stym_aug:
    #             # out = out.reshape(BDC_rec.shape[0], self.params.n_way, -1)
    #             out = torch.max(out,dim=1)[0]
    #             # out =  out[:, range(self.params.n_way), range(self.params.n_way)]
    #         # out_cons = SFC(BDC_norm_cons)
    #
    #         # loss_ce = 0.5 * (F.cross_entropy(out,sample_label.cuda().squeeze(0))+F.cross_entropy(out_cons,sample_label.cuda().squeeze(0)))
    #         loss_ce = loss_ce_fn(out,sample_label.squeeze(0))
    #         # loss_rec = 0.5 *(F.mse_loss(BDC_ori, BDC_norm_cons)+F.mse_loss(BDC_ori_cons, BDC_norm))
    #         # loss_rec = 0.5*(uniformity_loss(BDC_ori, BDC_norm_cons,sample_label)+uniformity_loss(BDC_ori_cons, BDC_norm,sample_label))
    #         # =============================
    #         # BDC_rec_cons = self.comp_relation(rec_map_cons).view(sample_support.shape[0], -1)
    #         # spt_norm = torch.norm(BDC_rec_cons, p=2, dim=1).unsqueeze(1).expand_as(BDC_rec_cons)
    #         # BDC_norm_cons = BDC_rec_cons.div(spt_norm + 1e-6)
    #
    #         # BDC_ori = self.drop(BDC_ori)
    #         # BDC_norm_cons = self.drop(BDC_norm_cons)
    #
    #         # loss_rec = uniformity_loss(BDC_ori, BDC_norm_cons,sample_label,temp=0.5)
    #         # 1.5 try
    #         # loss_rec = uniformity_loss(BDC_ori_cons, BDC_rec_cons,temp=0.2)
    #         # loss_rec += uniformity_loss(BDC_ori, BDC_rec, temp=0.2)
    #         # 对比s
    #         # loss_rec = uniformity_loss(BDC_norm, BDC_norm_cons,sample_label)
    #         # loss_rec = -Distance_Correlation(rec_map.view(rec_map.shape[0],-1),rec_map_cons.view(rec_map.shape[0],-1))
    #         # loss_rec = Distance_Correlation(BDC_ori, prototype)
    #         # loss_rec = Distance_Correlation(BDC_ori,BDC_rec_cons)
    #         if int(self.params.Loss_ablation/2)>0:
    #             loss_rec = Distance_Correlation(BDC_ori, BDC_rec)
    #         else:
    #             loss_rec = 0
    #         if self.params.Loss_ablation%2>0:
    #             loss_rec += Distance_Correlation(BDC_rec,BDC_rec_cons)
    #
    #
    #         # loss_rec = 1 * F.l1_loss(BDC_norm, BDC_ori)
    #
    #
    #         # 2.12try
    #         # loss_rec = uniformity_loss(BDC_rec, BDC_rec_cons,temp=0.5)
    #         # print(loss_rec)
    #         # loss =0.5 * loss_ce + 0.5 * loss_rec
    #         loss = loss_ce
    #         # loss = loss_rec
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         if lr_scheduler is not None:
    #             lr_scheduler.step()
    #         # print('loss_ce: {:.2f} \t loss_mse: {:.2f}'.format(loss_ce,loss_rec))
    #
    #     rec_layer.eval()
    #     SFC.eval()
    #     if self.params.LR_rec:
    #         support_z_rec = rec_layer(support_z)
    #         if self.params.embeding_way in ['BDC']:
    #             support_z_rec = self.dcov(support_z_rec)
    #         else:
    #             support_z_rec = self.avg_pool(support_z_rec)
    #         spt_norm = torch.norm(support_z_rec, p=2, dim=1).unsqueeze(1).expand_as(support_z_rec)
    #         support_z_rec = support_z_rec.div(spt_norm + 1e-6)
    #         support_z_rec = support_z_rec.reshape(support_z.shape[0]//self.params.n_aug_support_samples,self.params.n_aug_support_samples,-1)
    #         # support_ys = support_ys.view((support_z.shape[0]//self.params.n_aug_support_samples,-1))[:,0]
    #         support_ys = support_ys.reshape((support_z.shape[0]//self.params.n_aug_support_samples,-1))[:,1:]
    #
    #         # support_z_rec = 0.5 * support_z_rec[:,0,:] + 0.5* torch.mean(support_z_rec[:,1:,:],dim=1)
    #         support_z_rec = 0.5 * support_z_rec[:,0,:].unsqueeze(1) + 0.5* support_z_rec[:,1:,:]
    #         # support_z_rec = support_z_rec[:,0,:]
    #         clf = self.LR_rec(
    #             support_z_rec.reshape(support_z_rec.shape[0] * (self.params.n_aug_support_samples - 1), -1),
    #             support_ys)
    #         # clf = self.LR_rec(
    #         #     support_z_rec,
    #         #     support_ys)
    #
    #     with torch.no_grad():
    #         query_rec = rec_layer(query_z)
    #         # query_rec = query_z
    #         if self.params.embeding_way in ['BDC']:
    #             query_rec = self.dcov(query_rec)
    #         else:
    #             query_rec = self.avg_pool(query_rec)
    #             # query_rec = self.comp_relation(query_rec)
    #
    #         # query_rec = self.dcov(query_z)
    #         # query_rec = self.comp_relation(query_rec)
    #         spt_norm = torch.norm(query_rec, p=2, dim=1).unsqueeze(1).expand_as(query_rec)
    #         query_rec = query_rec.div(spt_norm + 1e-6)
    #
    #         # print(query_rec.shape)
    #         query_rec = query_rec.view(query_rec.shape[0]//self.params.n_symmetry_aug, self.params.n_symmetry_aug, -1)
    #         # query_rec = 1 / (self.params.n_symmetry_aug) * torch.sum(query_rec, dim=1)
    #         if self.params.n_symmetry_aug>1:
    #             # query_rec = 1 / (self.params.n_symmetry_aug) * torch.sum(query_rec, dim=1)
    #             query_rec = 0.5 * query_rec[:,0,:] + 0.5 * (1 / (self.params.n_symmetry_aug-1) * torch.sum(query_rec[:,1:,:], dim=1))
    #         else:
    #             query_rec =  query_rec[:, 0, :]
    #         # spt_norm = torch.norm(query_rec, p=2, dim=1).unsqueeze(1).expand_as(query_rec)
    #         # query_rec = query_rec.div(spt_norm + 1e-6)
    #         if no_stym_aug:
    #             # print(query_rec.shape)
    #             query_rec = (query_rec.unsqueeze(1).expand(query_rec.shape[0],prototype.shape[0],self.dim)+prototype.unsqueeze(0))/2
    #         if self.params.LR_rec:
    #
    #             # print(query_rec.shape)
    #             z_query = query_rec.detach().cpu().numpy()
    #             out = torch.from_numpy(clf.predict(z_query))
    #         else:
    #             out = SFC(query_rec)
    #             if no_stym_aug:
    #                 out = out.reshape(query_rec.shape[0],self.params.n_way,-1)
    #                 # out = torch.max(out,dim=1)[0]
    #                 # out = F.softmax(out,dim=-1)
    #                 return out[:,range(self.params.n_way),range(self.params.n_way)]
    #                 # print(out.shape)
    #     return out

    def LR(self,support_z,support_ys,query_z,query_ys):
        # clf = make_pipeline(StandardScaler(), SVC(gamma='auto',
        #                                           C=self.params.penalty_c,
        #                                           kernel='linear',
        #                                           decision_function_shape='ovr'))
        clf = LR(penalty='l2',
                 random_state=0,
                 C=self.params.penalty_c,
                 solver='lbfgs',
                 max_iter=1000,
                 multi_class='multinomial')
        # print(support_z.shape)
        spt_norm = torch.norm(support_z, p=2, dim=1).unsqueeze(1).expand_as(support_z)
        # spt_norm = torch.sqrt(spt_norm )
        spt_normalized = support_z.div(spt_norm  + 1e-6)
        qry_norm = torch.norm(query_z, p=2, dim=1).unsqueeze(1).expand_as(query_z)
        # qry_norm = torch.sqrt(qry_norm )
        qry_normalized = query_z.div(qry_norm + 1e-6)
        #
        z_support = spt_normalized.detach().cpu().numpy()
        y_support = support_ys.view(-1).cpu().numpy()
        z_query = qry_normalized.detach().cpu().numpy()

        # z_support = support_z.detach().cpu().numpy()
        # y_support = support_ys.view(-1).cpu().numpy()
        # z_query = query_z.detach().cpu().numpy()
        clf.fit(z_support, y_support)

        return torch.from_numpy(clf.predict(z_query))


if __name__ == '__main__':
    # print(random_sample(5,125,25))
    print(random_sample(5, 25, 20))