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


class Net_CL(nn.Module):
    def __init__(self,params,num_classes = 5, local_proto =3 ,  mask_ad =False,drop_rate = .5):
        super(Net_CL, self).__init__()

        self.params = params
        self.out_map = False
        resnet_layer_dim = [64, 160, 320, 640]
        if params.model == 'resnet12':
            self.backbone = resnet12(avg_pool=True,num_classes=64)
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
        self.Conv = nn.Sequential(
            nn.Conv2d(resnet_layer_dim[-1], self.reduce_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.reduce_dim),
            nn.ReLU(inplace=True)
        )
        drop_rate = params.drop_rate
        self.SFC = nn.Linear(self.dim, num_classes)
        self.SFC.bias.data.fill_(0)

        self.drop = nn.Dropout(drop_rate)
        self.drop_swap = nn.Dropout(0.2)

        self.temperature = nn.Parameter(torch.log((1. /(2 * self.feat_dim[1] * self.feat_dim[2])* torch.ones(1, 1))),
                                            requires_grad=True)

        self.dcov = BDC(is_vec=True, input_dim=[self.reduce_dim,10,10], dimension_reduction=self.reduce_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

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
        x = (x - torch.mean(x, dim=1).unsqueeze(1))
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

    def forward_feature(self, x, confusion=False, out_map=False):
        feat_map = self.backbone(x, is_FPN=(self.params.FPN_list is not None))
        feat_map = self.Conv(feat_map)

        return feat_map

    def normalize_feature(self, x):
        if self.params.norm == 'center':
            # print(x.shape)
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
        x = self.dcov(x)
        x = self.drop(x)
        return self.SFC(x)

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
            image , image_cl, label = data
            image = image.cuda()
            image_cl = image_cl.cuda()
            label = label.cuda()
            BDC_1, BDC_2, out_1, out_2= self.forward_feat(image, image_cl)
            loss = 0.5 * F.cross_entropy(out_1, label)
            loss += 0.5 * F.cross_entropy(out_2, label)
            loss += 0.5 * uniformity_loss(BDC_1, BDC_2)
            avg_loss = avg_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(out_1, 1)
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