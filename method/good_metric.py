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
from .DN4_module import ImgtoClass_Metric
# from model.resnet12 import resnet12
from model.ADL import ADL,ADL_variant,ADL_sig
from torch.nn.utils.weight_norm import WeightNorm
from utils.loss import entropy_loss
from sklearn.cluster import KMeans
from utils.loss import *
from utils.emd_utils import *
import random
from utils.loss import *
from sklearn.linear_model import LogisticRegression as LR
# torch.autograd.set_detect_anomaly(True)
from utils.distillation_utils import *
import math
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


def random_sample(linspace, max_idx, num_sample=5):
    sample_idx = np.random.choice(range(linspace), num_sample)
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


class Net(nn.Module):
    def __init__(self,params,num_classes = 5, local_proto =3 ,  mask_ad =False,drop_rate = .5):
        super(Net, self).__init__()

        self.params = params
        self.out_map = False
        if params.model == 'resnet12':
            self.backbone = resnet12(avg_pool=True,num_classes=64)
            resnet_layer_dim = [64, 160, 320, 640]
        elif params.model == 'resnet18':
            self.backbone = ResNet18()
            resnet_layer_dim = [64, 128, 256, 512]

        if params.metric == 'DN4':
            self.avg_pool = nn.AdaptiveAvgPool2d(3)
            self.imgtoclass = ImgtoClass_Metric(neighbor_k=1)
            self.out_map = True

        self.reduce_dim = params.reduce_dim
        self.feat_dim = self.backbone.feat_dim
        self.dim = int(self.reduce_dim * (self.reduce_dim+1)/2)
        if self.params.FPN_list is not None:
            # self.Conv_list = []
            # for i in self.params.FPN_list:
            #     conv_i = nn.Sequential(
            #         nn.Conv2d(resnet12_layer_dim[i], self.reduce_dim, kernel_size=1, stride=1, bias=False),
            #         nn.BatchNorm2d(self.reduce_dim),
            #         nn.ReLU(inplace=True)
            #     )
            #     self.Conv_list.append(conv_i)
            # 只考虑一个fpn
            self.Conv_fpn = nn.Sequential(
                nn.Conv2d(resnet_layer_dim[self.params.FPN_list[0]], self.reduce_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.reduce_dim),
                nn.ReLU(inplace=True)
            )
        self.Conv = nn.Sequential(
            nn.Conv2d(resnet_layer_dim[-1], self.reduce_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.reduce_dim),
            nn.ReLU(inplace=True)
        )
        drop_rate = params.drop_rate
        if self.params.flatten_fpn:
            self.SFC = nn.Linear(self.dim*2, num_classes)
        elif self.params.no_diag:
            self.SFC = nn.Linear(self.dim-self.reduce_dim, num_classes)
        else:
            self.SFC = nn.Linear(self.dim, num_classes)
        # self.SFC = nn.Linear(self.reduce_dim, num_classes)
        self.SFC.bias.data.fill_(0)
        if params.confusion:
            self.classifier = nn.Linear(self.reduce_dim, num_classes)
        # self.SFC = nn.Linear(reduce_dim*reduce_dim, num_classes, bias=False)
        self.drop = nn.Dropout(drop_rate)
        if params.confusion:
            self.drop_confusion = nn.Dropout(params.confusion_drop_rate)
        #     ---------------update----------
        # self.drop_confusion = nn.Dropout2d(params.confusion_drop_rate)


        if self.params.normalize_feat:
            pass
            # self.bn = nn.BatchNorm1d(self.dim)
        # self.temperature = nn.Parameter(torch.log((1. / (2 * self.feat_dim[1] * self.feat_dim[2])) * torch.ones(1, 1)),
        #                                 requires_grad=True)

        if self.params.idea_variant:
            self.temperature = nn.Parameter(torch.log((1. /(2 * self.feat_dim[1] * self.feat_dim[2])* torch.ones(1, 1))),
                                            requires_grad=True)
        else:
            self.temperature = nn.Parameter(torch.log((1. / 2  * torch.ones(1, 1))),
                                        requires_grad=True)
        if self.params.temp_element:
            self.temperature = nn.Parameter(
                torch.log((1. / (2 * self.feat_dim[1] * self.feat_dim[2]) * torch.ones(self.reduce_dim, self.reduce_dim))),
                requires_grad=True)

        if self.params.idea == 'bdc':
            self.dcov = BDC(is_vec=True, input_dim=self.feat_dim, dimension_reduction=self.reduce_dim)
            self.dcov_fpn = BDC(is_vec=True, input_dim=[320,10,10],dimension_reduction=self.reduce_dim)

        # if params.learnable_alpha:
        if params.no_diag:
            self.fpn_alpha = nn.Parameter(2.0 * torch.ones(self.dim-self.reduce_dim), requires_grad=True)
        else:
            self.fpn_alpha = nn.Parameter(2.0 * torch.ones(self.dim),requires_grad=True)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self):
        pass

    def normalize(self,x):
        # x = (x - torch.mean(x,dim=1).unsqueeze(1))/torch.std(x,dim=1).unsqueeze(1)
        # x = (x - torch.mean(x, dim=0).unsqueeze(0))
        x = (x - torch.mean(x, dim=1).unsqueeze(1))
        # x = (x - torch.mean(x, dim=1).unsqueeze(1))/(torch.std(x,dim=1).unsqueeze(1) + 1e-8)
        return x

    def comp_relation(self,feat_map,out_map =False):
    #     batch * feat_dim * feat_map
        batchSize, dim, map_dim = feat_map.shape
        # drop_map_id  =  torch.randperm(dim)[:int(dim*0.3)]
        # feat_map_copy = feat_map.clone()

        # feat_norm = torch.norm(feat_map, p=2, dim=1).unsqueeze(1).expand_as(feat_map)
        # feat_map = feat_map.div(feat_norm+1e-6)
        # feat_map = torch.softmax(feat_map/10,dim=-1)
        # feat_map = feat_map.div(torch.norm(feat_map,dim=-1,keepdim=True))
        feat_map= self.drop_confusion(feat_map)


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
            if out_map:
                item_1 = torch.abs(feat_map_1 + feat_map_2)
                item_2 = torch.abs(feat_map_1 - feat_map_2)
            else:
                item_1 = torch.sum(torch.abs(feat_map_1 + feat_map_2), dim=-1)
                item_2 = torch.sum(torch.abs(feat_map_1 - feat_map_2), dim=-1)
            if self.params.idea_variant :
                if self.params.temp_element:
                    temperature = self.temperature.unsqueeze(0).expand_as(item_1)
                    out = (item_1 - item_2) * torch.exp(temperature)
                else:
                    out = (item_1 - item_2)/2 * torch.exp(self.temperature)

            else:
                out = (item_1 - item_2)/(2 * feat_map.shape[2]) * torch.exp(self.temperature)

        # out = torch.clamp(out, min=1e-8)

        # ==================================
        if self.params.normalize_bdc:
            I_M = torch.ones(batchSize, dim, dim, device=feat_map.device).type(feat_map.dtype)
            # out = out - 1. / dim * out.bmm(I_M) - 1. / dim * I_M.bmm(out) + 1. / (dim * dim) * I_M.bmm(out).bmm(I_M)
            out = out - 1. / dim * out.bmm(I_M) - 1. / dim * I_M.bmm(out)
        # ==================================

        # idxs = np.where(np.triu(np.ones([self.reduce_dim, self.reduce_dim]), k=0).flatten() == 1)[0]
        # out = out[:, idxs]
        if out_map:

            out = Triumap(out.reshape(batchSize,dim,dim,self.feat_dim[-2],self.feat_dim[-1]), no_diag=self.params.no_diag)
            out = self.avg_pool(out)
        else:
            out = Triuvec(out,no_diag=self.params.no_diag)

        if self.params.normalize_feat:
            # out = self.bn(out)
            out = self.normalize(out)

        # out_norm = torch.norm(out,dim=-1,keepdim=True) + 1e-6
        # out = out.div(out_norm)

        return out

    def forward_feature(self,x, confusion=False, out_map = False):
        # [feat, feat_map], logit = self.backbone(x, is_feat=True)
        #  if is_FPN = Ture  : feat_map = [f0, f1, f2, f4]
        feat_map = self.backbone(x,is_FPN = (self.params.FPN_list is not None))
        # feat_map_center = F.adaptive_avg_pool1d(feat_map_reduce,1)
        # feat_map_reduce = feat_map_reduce - feat_map_center
        idea = self.params.idea
        if idea =='ab':
            feat_map_reduce = self.Conv(feat_map).view(feat_map.shape[0], self.reduce_dim, -1)
            # ==============================================================
            feat_map_reduce_norm = torch.norm(feat_map_reduce,dim=1,keepdim=True)
            feat_map_reduce = feat_map_reduce.div(feat_map_reduce_norm)
            # ==============================================================
            # 尝试detach的方式，使得梯度不会多次更新,进而导致梯度必须很小  √
            # 或者可以试一下啊norm之类的操作
            relation_map = torch.bmm(feat_map_reduce, feat_map_reduce.detach().permute(0,2,1)).view(x.size(0),-1)

            # relation_norm = torch.norm(relation_map,dim=-1,keepdim=True).expand_as(relation_map) + 1e-3
            # relation_map = relation_map.div(relation_norm)
            # ===============================================
            relation_map = torch.abs(relation_map)
            # ===============================================
            idxs = np.where(np.triu(np.ones([self.reduce_dim, self.reduce_dim]), k=0).flatten() == 1)[0]
            out = relation_map[:,idxs]
        elif idea == 'a+-b':
            # print('a +- b')
            if self.params.FPN_list is not None:
                feat_map_fpn = feat_map[self.params.FPN_list[0]]
                if self.params.stop_grad:
                    feat_map_fpn = feat_map_fpn.detach()
                feat_map_reduce_fpn = self.Conv_fpn(feat_map_fpn).view(feat_map_fpn.shape[0], self.reduce_dim, -1)
                out_fpn = self.comp_relation(feat_map_reduce_fpn)
                feat_map_reduce = self.Conv(feat_map[-1]).view(feat_map[-1].shape[0], self.reduce_dim, -1)
                out = self.comp_relation(feat_map_reduce)
                # for i in range(len(self.Conv_list)):
                #     feat_map_i =feat_map[self.params.FPN_list[i]]
                #     out_i = self.comp_relation(self.Conv_list[i](feat_map_i.cpu()).view(feat_map_i.shape[0],self.reduce_dim,-1))
                #     out.append(out_i.unsqueeze(1))
                if self.params.learnable_alpha:
                    # print(torch.sum(F.sigmoid(self.fpn_alpha)>0.7))
                    # time.sleep(100)
                    alpha = self.fpn_alpha.unsqueeze(0).repeat(out.shape[0],1)
                    out = F.sigmoid(alpha) * out + (1 - F.sigmoid(alpha)) * out_fpn
                elif self.params.flatten_fpn:
                    # print(out.shape)
                    # print(out_fpn.shape)
                    out = torch.cat([out,out_fpn],dim=-1)
                else:
                    out = 0.5 * out + 0.5 * out_fpn
                    # out = torch.sum(torch.cat([out_fpn,out],dim=1).cuda(),dim=1)/len(self.params.FPN_list)

            else:
                feat_map_reduce = self.Conv(feat_map).view(feat_map.shape[0], self.reduce_dim, -1)
                out = self.comp_relation(feat_map_reduce,out_map)
        elif idea == 'bdc':
            if self.params.FPN_list is not None:
                feat_map_fpn = feat_map[self.params.FPN_list[0]]
                if self.params.stop_grad:
                    feat_map_fpn = feat_map_fpn.detach()
                out_fpn = self.dcov_fpn(feat_map_fpn)
                out  = self.dcov(feat_map[-1])
                if self.params.learnable_alpha:
                    alpha = self.fpn_alpha.unsqueeze(0).repeat(out.shape[0], 1)
                    out = F.sigmoid(alpha) * out + (1 - F.sigmoid(alpha)) * out_fpn
                else:
                    out = 0.5 * out + 0.5 * out_fpn
            else:
                out = self.dcov(feat_map)
        #  取上三角元素

        # print(idxs)
        # print((time.time()-start_time)/60)
        if confusion:
            I = torch.eye(self.reduce_dim, self.reduce_dim).reshape(self.reduce_dim ** 2)
            ind_range = torch.arange(0,self.reduce_dim)
            index = I.nonzero(as_tuple=False).squeeze() - (ind_range+1)*ind_range/2
            out_diag = out[:,index.long()]
            return out,out_diag

        return out
        # return relation_map

    def normalize_feature(self, x):
        if self.params.norm == 'center':
            # print(x.shape)
            x = x - x.mean(2).unsqueeze(2)
            return x
        else:
            return x

    def forward_feat(self,x):
        [feat, feat_map], logit = self.backbone(x,is_feat=True)
        return feat

    def forward_pretrain(self, x, confusion = False):
        x = self.forward_feature(x,confusion=confusion,out_map=False)
        if confusion:
            [x_,x_diag] = x
            x_ = self.drop(x_)
            # ---------------12.22 修改了confusion drop 的定义 --------------
            # x_diag = self.drop_confusion(x_diag)
            # print(x_.shape)
            # print(x_diag.shape)
            return self.SFC(x_), self.classifier(x_diag)
        else:
            x = self.drop(x)
            return self.SFC(x)
    # def get_sfc(self,x,label):
    #     [feat, feat_map], logit = self.backbone(x)
    #     fore_mask = self.ADL(feat_map, is_importance_area=True)
    #     # print(torch.max(label))
    #     for i in range(torch.max(label).cpu().numpy()+1):
    #         idxs = torch.where(label==i)[0].cpu().numpy()
    #         local_feat = []
    #         clf = KMeans(n_clusters=self.local_proto,random_state=0)
    #         for j in idxs:
    #             feat_map_j = feat_map[j, :, :, :].unsqueeze(0)
    #             fore_mask_j = fore_mask[j, :, :].unsqueeze(0)
    #             local_mask = torch.from_numpy(bwlabel(fore_mask_j.cpu().numpy()))
    #             for k in range(0, torch.max(local_mask) + 1):
    #                 local_mask_k = torch.where(local_mask == k, 1, 0).unsqueeze(1).expand(feat_map_j.shape)
    #                 feat_mask_k = torch.sum(torch.sum(feat_map_j.cpu() * local_mask_k, dim=-1), dim=-1) / torch.sum(
    #                     torch.sum(local_mask_k, dim=-1),
    #                     dim=-1)
    #                 local_feat.append(feat_mask_k)
    #         local_feat = torch.cat(local_feat, dim=0).to(feat_map_j.device)
    #         self.SFC[i,:,:].data = torch.nn.Parameter(
    #             torch.from_numpy(clf.fit(local_feat.detach().cpu().numpy()).cluster_centers_),requires_grad=True).cuda()

    # def init_sfc(self, SFC, proto_local_list, support_y):
    #
    #     for i in range(self.params.n_way):
    #         clf = KMeans(n_clusters=self.local_proto-1, random_state=0,max_iter=1000)
    #         sample_idxs = torch.where(support_y == i)[0]
    #         local_feat =[]
    #         # SFC[i, 0, :] = 0
    #         for idx in sample_idxs:
    #             # print(proto_local_list[idx].shape)
    #             # SFC[i, 0, :] += proto_local_list[idx].squeeze(0)[0, :].cpu()
    #             # local_feat.append(proto_local_list[idx].squeeze(0)[1:,:])
    #             local_feat.append(proto_local_list[idx].squeeze(0))
    #
    #         # print(len(sample_idxs))
    #         # SFC[i, 0, :] = torch.nn.Parameter(SFC[i, 0, :].float() / len(sample_idxs),requires_grad=True).cuda()
    #         local_feat = torch.cat(local_feat, dim=0)
    #         # SFC[i, 1:, :] = torch.nn.Parameter(
    #         #     torch.from_numpy(clf.fit(local_feat.detach().cpu().numpy()).cluster_centers_),
    #         #     requires_grad=True).cuda()
    #         SFC[i, :, :].data = torch.nn.Parameter(
    #             torch.from_numpy(clf.fit(local_feat.detach().cpu().numpy()).cluster_centers_),
    #             requires_grad=True).cuda()
    #     return SFC
    #
    # # def forward_pretrain(self,x):
    # #     [feat, feat_map], logit = self.backbone(x,is_feat=True)
    # #     fore_feat, back_feat, map_sig = self.ADL_sig(feat_map, area_control=True)
    # #     if self.params.deep_emd or self.params.drop_gama ==  1:
    # #         fore_feat = feat
    # #     fore_out = self.classifier(fore_feat.view(fore_feat.size(0),-1))
    # #     back_out = self.classifier_disc(back_feat.clone().view(back_feat.size(0),-1))
    # #     back_out_detach = self.classifier_disc(back_feat.detach().view(back_feat.size(0),-1))
    # #     return fore_out,back_out,back_out_detach,map_sig
    #
    # def get_weight_vector(self, A, B):
    #     #  A : n_batch * n_local * in_dim
    #
    #     # print(A.shape)
    #     # print(B.shape)
    #
    #     M = A.shape[0]
    #     N = B.shape[0]
    #
    #     # B = torch.mean(B, dim=1).unsqueeze(1)
    #     B = B[:,0,:].unsqueeze(1)
    #     B = B.repeat( 1, A.shape[1], 1)
    #
    #     A = A.unsqueeze(1)
    #     B = B.unsqueeze(0)
    #
    #     A = A.repeat(1, N, 1, 1)
    #     B = B.repeat(M, 1, 1, 1)
    #
    #     # print(A.shape)
    #     combination = (A * B).sum(3)
    #     combination = combination.view(M, N, -1)
    #     # print(combination)
    #     # ---------------**update : increase base value 1e-3 --> 100 ----------------------------
    #     combination = F.relu(combination) + 1e-3
    #     # combination = torch.abs(combination)
    #     # print(combination)
    #     # time.sleep(10)
    #     return combination
    #
    # def get_similiarity_map(self, proto, query):
    #     way = proto.shape[0]
    #     num_query = query.shape[0]
    #     # query = query.view(query.shape[0], query.shape[1], -1)
    #     # proto = proto.view(proto.shape[0], proto.shape[1], -1)
    #
    #     proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
    #     query = query.unsqueeze(1).repeat([1, way, 1, 1])
    #     # proto = proto.permute(0, 1, 3, 2)
    #     # query = query.permute(0, 1, 3, 2)
    #     feature_size = proto.shape[-2]
    #
    #     # print(proto.shape)
    #     if self.params.metric == 'cosine':
    #         proto = proto.unsqueeze(-3)
    #         query = query.unsqueeze(-2)
    #         query = query.repeat(1, 1, 1, feature_size, 1)
    #         similarity_map = F.cosine_similarity(proto, query, dim=-1)
    #     if self.params.metric == 'l2':
    #         proto = proto.unsqueeze(-3)
    #         query = query.unsqueeze(-2)
    #         query = query.repeat(1, 1, 1, feature_size, 1)
    #         similarity_map = (proto - query).pow(2).sum(-1)
    #         similarity_map = 1 - similarity_map
    #
    #     return similarity_map
    #
    # def get_sfc(self,support_xs, support_ys, ):
    #     support_ys = support_ys.squeeze(0).cuda()
    #     if self.params.deep_emd:
    #         support_xs_t = torch.cat(support_xs, dim=0)
    #         SFC = support_xs_t.view(-1, self.params.n_shot, support_xs_t.shape[-2], support_xs_t.shape[-1]).mean(
    #             dim=1).clone().detach()
    #         # print(support_xs_t.view(self.params.n_shot, -1, support_xs_t.shape[-2], support_xs_t.shape[-1])[0:4,0,0,0])
    #         SFC = nn.Parameter(SFC.detach(), requires_grad=True)
    #         # return SFC
    #     else:
    #         SFC = torch.nn.Parameter(torch.rand(self.params.n_way,self.local_proto,self.in_feature),requires_grad=True)
    #         # SFC = torch.nn.Parameter(torch.zeros(self.params.n_way, self.local_proto, self.in_feature).float(),)
    #         SFC = self.init_sfc(SFC,support_xs,support_ys).cuda()
    #         SFC.requires_grad_(True)
    #     # print(torch.sum(torch.isnan(SFC)))
    #     # print(self.params.sfc_lr)
    #     optimizer = torch.optim.SGD([SFC], lr=self.params.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0, )
    #     Schuler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.params.sfc_update_step,eta_min=1)
    #     Schuler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.4)
    #     # optimizer = torch.optim.SGD([SFC], lr=self.params.sfc_lr, momentum=0.9, )
    #     with torch.enable_grad():
    #
    #         for step in range(0, self.params.sfc_update_step):
    #             rand_aug = random_sample(self.params.n_aug_support_samples, len(support_xs),len(support_xs)//self.params.n_aug_support_samples)
    #             random.shuffle(rand_aug)
    #             for i in range(0, self.params.n_way * self.params.n_shot, self.params.sfc_bs):
    #                 selected_id = rand_aug[i: min(i + self.params.sfc_bs, self.params.n_way * self.params.n_shot)]
    #                 # print(support_ys)
    #                 batch_label = support_ys[selected_id]
    #                 logits = []
    #                 for j in range(len(selected_id)):
    #                     batch_shot = support_xs[selected_id[j]].unsqueeze(0)
    #                     # print(torch.sum(torch.isnan(batch_shot)))
    #                     logit = self.forward_emd_1shot_loss( batch_shot.detach(), SFC,)
    #                     logits.append(logit)
    #                     # print(logit)
    #                 logits = torch.cat(logits,dim=0)
    #                 # print(logits)
    #
    #                 # print(batch_label)
    #                 # print(logits)
    #                 # time.sleep(5)
    #                 loss = F.cross_entropy(logits, batch_label)
    #                 # loss = ce_loss(logits, batch_label)
    #                 # print(torch.sum(torch.isnan(loss)))
    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 optimizer.step()
    #
    #             # Schuler.step()
    #     return SFC
    #
    # def forward_emd_1shot_loss(self, qury_list, proto,  ):
    #     logits = []
    #     # print(len(qury_list))
    #     for query in qury_list:
    #         # weight_1:  n_batch * n_way * n_local
    #         # print(proto.shape)
    #         weight_1 = self.get_weight_vector(query, proto)
    #         weight_2 = self.get_weight_vector(proto, query)
    #
    #         # print(torch.sum(torch.isnan(proto)))
    #         # print(torch.sum(torch.isnan(weight_2)))
    #
    #         proto = self.normalize_feature(proto)
    #         query = self.normalize_feature(query)
    #
    #         # sim_map: n_batch * n_way * num_local * num_porto
    #         similarity_map = self.get_similiarity_map(proto, query)
    #         logit = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv')
    #         logits.append(logit)
    #     return torch.cat(logits,dim=0)
    #
    # def forward_emd_1shot_inference(self,  qury_list, proto_list, ):
    #     logits = []
    #
    #     for q in range(len(qury_list)):
    #         logit = []
    #         # print(len(proto_list))
    #         for p in range(len(proto_list)):
    #
    #             query = qury_list[q]
    #             proto = proto_list[p]
    #             # query = query.unsqueeze(0)
    #             # proto = proto.unsqueeze(0)
    #             # print(query.shape)
    #             # print(proto.shape)
    #             weight_1 = self.get_weight_vector(query, proto)
    #             weight_2 = self.get_weight_vector(proto, query)
    #
    #             proto = self.normalize_feature(proto)
    #             query = self.normalize_feature(query)
    #
    #
    #             # sim_map: n_batch * n_way * num_local * num_porto
    #             similarity_map = self.get_similiarity_map(proto, query)
    #
    #             weight_1 = weight_1.squeeze(0).squeeze(0)
    #             weight_2 = weight_2.squeeze(0).squeeze(0)
    #             num_node = weight_1.shape[-1]
    #             # print(weight_1)
    #             # print(weight_2)
    #
    #
    #             similarity_map = similarity_map.squeeze(0).squeeze(0)
    #
    #             # print(weight_1.shape)
    #             # print(weight_2.shape)
    #             # print(similarity_map.shape)
    #             _, flow = emd_inference_opencv(1 - similarity_map, weight_1, weight_2)
    #             similarity_map = (similarity_map) * torch.from_numpy(flow).cuda()
    #             temperature = (self.params.temperature / num_node)
    #             logit.append( similarity_map.sum(-1).sum(-1).unsqueeze(0) * temperature)
    #         # print(logit)
    #         logit = torch.cat(logit,dim=0).unsqueeze(0)
    #         logits.append(logit)
    #     scores = torch.cat(logits,dim=0)
    #     return scores
    #
    # def forward_local(self,x,is_cell=False):
    #     [feat, feat_map], logit = self.backbone(x,is_feat=True)
    #     if self.params.local_mode == 'cell_mix' :
    #         is_cell = True
    #     out = []
    #     start_lab = 0 if self.params.include_bg else 1
    #     for j in range(feat_map.shape[0]):
    #         feat_map_j = feat_map[j, :, :, :].unsqueeze(0)
    #         mask = self.ADL(feat_map_j, is_importance_area=True)
    #         local_mask = torch.from_numpy(bwlabel(mask.cpu().numpy()))
    #         if self.params.local_mode == 'mask_pool':
    #             local_mask_ = torch.where(local_mask.view(-1) != 0)[0]
    #             if self.params.drop_gama == 1:
    #                 local_mask_ = range(0,25)
    #             feat_mask_j = feat_map_j.view(feat_map_j.shape[0], feat_map_j.shape[1], -1)[:, :, local_mask_]
    #             # if is_cell is False or (self.params.local_mode == 'cell_mix' and i != 0):
    #             feat_mask_j = F.adaptive_avg_pool1d(feat_mask_j, 1)
    #             out.append(feat_mask_j.squeeze(-1))
    #         else:
    #             local_feat = []
    #             if self.params.feature_pyramid:
    #                 local_feat.append(F.adaptive_avg_pool1d(feat_map_j.view(feat_map_j.shape[0], feat_map_j.shape[1], -1),1).permute(0, 2, 1))
    #                 local_feat.append(feat_map_j.view(feat_map_j.shape[0],feat_map_j.shape[1],-1).permute(0,2,1))
    #
    #             for i in range(start_lab, torch.max(local_mask) + 1):
    #                 # local_mask_i = torch.where(local_mask == i, 1, 0).unsqueeze(1).expand(feat_map_j.shape)
    #                 local_mask_idx = torch.where(local_mask.view(-1) == i)[0]
    #                 feat_mask_i = feat_map_j.view(feat_map_j.shape[0],feat_map_j.shape[1],-1)[:,:,local_mask_idx]
    #                 if is_cell is False or (self.params.local_mode == 'cell_mix' and i != 0):
    #
    #                     feat_mask_i = F.adaptive_avg_pool1d(feat_mask_i,1)
    #                 # feat_mask_i : 1 * 640 *  1 if not is_cell else n_cell_of_local
    #
    #                 local_feat.append(feat_mask_i.permute(0,2,1))
    #             local_feat = torch.cat(local_feat, dim=1).to(feat_map_j.device)
    #             # print(local_feat.shape)
    #             out.append(local_feat)
    #     return out
    #
    # def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):
    #     num_query = similarity_map.shape[0]
    #     num_proto = similarity_map.shape[1]
    #
    #     num_node = weight_1.shape[-1]
    #     assert num_node!=0
    #     # print(weight_1.shape)
    #     # print(weight_2.shape)
    #     # print(similarity_map.shape)
    #     # time.sleep(10)
    #     if solver == 'opencv':  # use openCV solver
    #         for i in range(num_query):
    #             for j in range(num_proto):
    #                 _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
    #
    #                 similarity_map[i, j, :, :] = (similarity_map[i, j, :, :]) * torch.from_numpy(flow).cuda()
    #
    #         temperature = (self.params.temperature / num_node)
    #         logitis = similarity_map.sum(-1).sum(-1) * temperature
    #         return logitis
    #     elif solver == 'qpth':
    #         # print(weight_2.shape)
    #         # print(similarity_map.shape)
    #         weight_2 = weight_2.permute(1, 0, 2)
    #         similarity_map = similarity_map.view(num_query * num_proto, similarity_map.shape[-2],
    #                                              similarity_map.shape[-1])
    #         weight_1 = weight_1.view(num_query * num_proto, weight_1.shape[-1])
    #         weight_2 = weight_2.reshape(num_query * num_proto, weight_2.shape[-1])
    #
    #         _, flows = emd_inference_qpth(1 - similarity_map, weight_1, weight_2,form=self.params.form, l2_strength=self.params.l2_strength)
    #
    #         logitis=(flows*similarity_map).view(num_query, num_proto,flows.shape[-2],flows.shape[-1])
    #         temperature = (self.params.temperature / num_node)
    #         logitis = logitis.sum(-1).sum(-1) *  temperature
    #     else:
    #         raise ValueError('Unknown Solver')
    #
    #     return logitis

    def metric_DN4(self,support,x_y,query,q_y):
        # support : (k_shot*n_way) * channel * h * w
        support_set = [support[torch.where(x_y.squeeze(0)==i)[0],:,:,:] for i in range(self.params.n_way)]
        # print(x_y.shape)

        S = []
        # 对各个类构造local feat bag
        for i in range(self.params.n_way):
            support_set_sam = support_set[i]
            B, C, h, w = support_set_sam.size()
            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().view(C, -1)
            S.append(support_set_sam)
        # print(self.imgtoclass(query,S))
        return torch.max(self.imgtoclass(query,S),dim=1)[1]


    def train_loop(self,epoch,train_loader,optimizers):
        ths_confusion = False
        if self.params.ths_confusion:
            ths_confusion = True
            ths = self.params.ths_confusion
        print_step = 100
        smooth_rate = 0.3
        avg_loss = 0
        [optimizer , optimizer_ad] = optimizers
        total_correct = 0
        iter_num = len(train_loader)
        total = 0
        k_rocord = []
        for i ,data in enumerate(train_loader):
            image , label = data
            image = image.cuda()
            label = label.cuda()
            if self.params.confusion:
                [out, out_diag] = self.forward_pretrain(image, confusion=self.params.confusion)
                loss = 0.5 * F.cross_entropy(out, label)
                value, ind = torch.sort(out.detach(),descending=True)
                if ths_confusion:
                    if np.abs(ths_confusion - 1) <1e-6:
                        # print('---------------------------------')
                        label_c = torch.softmax(out/self.params.temperature,dim=-1)
                    else:
                    # --------------12.24---------------
                        cum_value = torch.cumsum(torch.softmax(value/2,dim=-1),dim=-1)
                        label_c = torch.ones_like(out) * smooth_rate / (out.shape[1] - self.params.k_c)
                        k = torch.sum(cum_value < ths, dim=-1) + 1
                        for j in range(ind.shape[0]):
                            label_c[j, ind[j,:k[j]]] = (1-smooth_rate)/k[j]
                            k_rocord.append(k[j].item())

                else:
                    ind_k = ind[:,:self.params.k_c]
                    # label_c = torch.ones_like(out) * (1 / ((self.params.k_c+1) * out.shape[1]))
                    label_c = torch.ones_like(out) * smooth_rate/(out.shape[1]-self.params.k_c)
                    # label_c = torch.zeros_like(out) + smooth_rate/out.shape[0]
                    for j in range(ind_k.shape[0]):
                        label_c[j,ind_k[j,:]] = (1-smooth_rate)/(self.params.k_c)
                        # label_c[j, ind_k[j, :]] += 1 / (self.params.k_c + 1)
                    # print(loss_div_fn(label_c, out_diag).item())
                loss += 0.5 * F.kl_div(F.log_softmax(out_diag,dim=-1), label_c,reduction='batchmean')
            else:
                out= self.forward_pretrain(image)
                loss = F.cross_entropy(out, label)
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
        for i,data in enumerate(val_loader):
            # tic = time.time()
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            split_size = 75
            if support_xs.squeeze(0).shape[0] > split_size:
                feat_sup_ = []
                # print(support_xs.shape)
                for j in range(math.ceil(support_xs.squeeze(0).shape[0] / split_size)):
                    # print(support_xs.squeeze(0)[i*128:min((i+1)*128,support_xs.shape[1]),:,:,:].shape)
                    fest_sup_item = self.forward_feature(
                        support_xs.squeeze(0)[j * split_size:min((j + 1) * split_size, support_xs.shape[1]), :, :,
                        :],out_map=self.out_map).cpu()
                    feat_sup_.append(fest_sup_item if len(fest_sup_item.shape) > 1 else fest_sup_item.unsqueeze(0))
                feat_sup = torch.cat(feat_sup_, dim=0)
            else:
                feat_sup = self.forward_feature(support_xs.squeeze(0),out_map=self.out_map)
            if query_xs.squeeze(0).shape[0] >= split_size:
                feat_qry_ = []
                # print(support_xs.shape)
                for j in range(math.ceil(query_xs.squeeze(0).shape[0] / split_size)):
                    # print(support_xs.squeeze(0)[i*128:min((i+1)*128,support_xs.shape[1]),:,:,:].shape)
                    feat_qry_item = self.forward_feature(query_xs.squeeze(0)[j * split_size:min((j + 1) * split_size, query_xs.shape[1]), :, :,:],out_map=self.out_map).cpu()
                    feat_qry_.append(feat_qry_item if len(feat_qry_item.shape) >= 1 else feat_qry_item.unsqueeze(0))
                # print(feat_qry_[0].shape)

                feat_qry = torch.cat(feat_qry_, dim=0)
            else:
                feat_qry = self.forward_feature(query_xs.squeeze(0),out_map=self.out_map)
            if self.params.metric =='DN4':
                pred = self.metric_DN4(feat_sup,support_ys,feat_qry, query_ys)
            else:
                pred = self.LR(feat_sup, support_ys, feat_qry, query_ys)
            if self.params.n_symmetry_aug > 1:
                pred = pred.view(-1, self.params.n_symmetry_aug)
                query_ys = query_ys.view(-1, self.params.n_symmetry_aug)
                pred = torch.mode(pred, dim=-1)[0]
                query_ys = torch.mode(query_ys, dim=-1)[0]
            # print(np.mean(pred.cpu().numpy() == query_ys.numpy()))
            acc_epo = np.mean(pred.cpu().numpy() == query_ys.numpy())
            acc.append(acc_epo)
            # print("\repisode {} acc: {:.2f} | avg_acc: {:.2f} +- {:.2f}, elapse : {:.2f}".format(i,acc_epo*100,*mean_confidence_interval(acc,multi=100),(time.time()-tic)/60),end='')
        return mean_confidence_interval(acc)

    def meta_test_loop(self,test_loader):
        acc = []
        classifier = 'emd' if not self.params.test_LR else 'LR'
        for i, data in enumerate(test_loader):
            tic = time.time()
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            split_size = 75
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
            if query_xs.squeeze(0).shape[0] > split_size:
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

            if self.params.metric == 'DN4':
                pred = self.metric_DN4(feat_sup, support_ys, feat_qry, query_ys)
                # print(pred)

            else:
                pred = self.LR(feat_sup, support_ys, feat_qry, query_ys)
            if self.params.n_symmetry_aug > 1:
                pred = pred.view(-1, self.params.n_symmetry_aug)
                query_ys = query_ys.view(-1, self.params.n_symmetry_aug)
                pred = torch.mode(pred,dim=-1)[0]
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
        model_t.eval()
        for i, data in enumerate(train_loader):
            image, label = data
            image = image.cuda()
            label = label.cuda()
            with torch.no_grad():
                out_t = model_t.forward_pretrain(image)

            if self.params.confusion:
                [out , out_diag] = self.forward_pretrain(image,confusion=self.params.confusion)
                loss = 0.5 * F.cross_entropy(out, label)
                ind_k = torch.sort(out.detach(),descending=True)[1][:,:self.params.k_c]
                label_c = torch.zeros_like(out)
                label_c = torch.ones_like(out) * (1 / ((self.params.k_c+1) * out.shape[1]))
                for j in range(ind_k.shape[0]):
                    label_c[j,ind_k[j,:]] += 1/(self.params.k_c+1)
                # print(loss_div_fn(label_c, out_diag).item())
                loss += 0.5 * F.kl_div(F.log_softmax(out_diag,dim=-1), label_c,reduction='batchmean')
            else:
                out= self.forward_pretrain(image)
                loss = 0.5 * F.cross_entropy(out, label)

            # out = self.forward_pretrain(image)
            # loss_cls = F.cross_entropy(out, label)
            loss_div = loss_div_fn(out, out_t)
            loss  += loss_div * 0.5

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
        if self.params.prompt:
            # get prompt pool
            total_shot = self.params.n_way*self.params.n_shot
            prob = torch.from_numpy(clf.predict_proba(z_support))
            # prob = prob.view(total_shot, -1)
            y_support = np.repeat(range(self.params.n_way), self.params.n_shot * self.params.n_aug_support_samples)
            prob = prob.view(-1,  self.params.n_way)[range(total_shot*self.params.n_aug_support_samples),y_support].squeeze()
            # 应该先筛选一下！！！！
            # print(prob.shape)
            _, idx = torch.max(prob.view(-1, self.params.n_aug_support_samples,),dim=-1)
            z_sup_most_confidence = torch.from_numpy(z_support).view(total_shot, -1, self.dim)[range(total_shot),idx,:]
            z_sup_oringin = torch.from_numpy(z_support).view(total_shot, -1, self.dim)[range(total_shot),0,:]
            x = z_sup_oringin.unsqueeze(-1)
            y = z_sup_most_confidence.unsqueeze(-1)

            # prompt_matrix = torch.bmm((x + y), (x + y).permute(0, 2, 1)) / (1 + torch.bmm(x, y.permute(0, 2, 1)))
            # prompt_matrix -= torch.eye(self.dim,self.dim).unsqueeze(0)
            prompt_matrix = torch.bmm(y,x.transpose(1,2))/torch.norm(x,dim=1,keepdim=True)

            # 5way5shot:  25 * dim *dim -->  75 * 25 * dim * dim
            prompt_matrix = prompt_matrix.unsqueeze(0).repeat(self.params.n_way*self.params.n_queries,1,1,1).cuda()
            # query:  75*dim -->    75 * 25 * dim * 1
            z_qry = torch.from_numpy(z_query).unsqueeze(1).unsqueeze(-1).repeat(1,prompt_matrix.shape[1],1,1).cuda()
            # transform query sample
            prompt_matrix_batch = prompt_matrix.view(-1,self.dim,self.dim).cuda()
            z_qry_batch = z_qry.view(-1,self.dim,1).cuda()
            res = torch.bmm(prompt_matrix_batch,z_qry_batch).squeeze(-1)

            # qry_norm = torch.norm(res, p=2, dim=1).unsqueeze(1).expand_as(res)
            # # qry_norm = torch.sqrt(qry_norm )
            # res = res.div(qry_norm + 1e-6)

            prob = torch.from_numpy(clf.predict_proba(res.cpu().numpy()))
            res = prob.view(self.params.n_way * self.params.n_queries, -1, self.params.n_way)
            return torch.max(torch.max(res,dim=1)[0],dim=-1)[1]

        else:
            return torch.from_numpy(clf.predict(z_query))


if __name__ == '__main__':
    print(random_sample(5,125,25))