import math
import os
import sys
import time

import tqdm

sys.path.append("..")
import numpy as np
import torch.nn as nn
import torch
import scipy
from scipy.stats import t
from model.resnet_new import ResNet12 as resnet12
from model.resnet_new import *
from model.ADL import ADL,ADL_variant,ADL_sig
from torch.nn.utils.weight_norm import WeightNorm
from utils.loss import entropy_loss
from sklearn.cluster import KMeans
from utils.loss import *
from utils.emd_utils import *
import random
from utils.loss import *
from sklearn.linear_model import LogisticRegression as LR
from .bdc_module import BDC
from utils.distillation_utils import *

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

class stl_deepbdc(nn.Module):
    def __init__(self,params,num_classes = 5,drop_rate = .5):
        super(stl_deepbdc, self).__init__()
        if params.model == 'resnet12':
            self.backbone = resnet12(avg_pool=True, num_classes=64)
        elif params.model == 'resnet18':
            self.backbone = ResNet18()
        reduce_dim = 128
        self.feat_dim = self.backbone.feat_dim
        self.reduce_dim = reduce_dim
        self.dim = int(reduce_dim * (reduce_dim+1)/2)
        self.dcov = BDC(is_vec=True, input_dim=self.feat_dim, dimension_reduction=reduce_dim)
        self.SFC = nn.Linear(self.dim, num_classes)
        self.SFC.bias.data.fill_(0)
        # self.SFC = nn.Linear(reduce_dim*reduce_dim, num_classes, bias=False)
        self.drop = nn.Dropout(drop_rate)
        self.params = params
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

    def forward_feature(self,x):
        x = self.backbone(x)
        out = self.dcov(x)
        return out

    def forward_pretrain(self, x):
        x = self.forward_feature(x)
        x = self.drop(x)
        return self.SFC(x)

    def train_loop(self,epoch,train_loader,optimizers):
        print_step = 100
        avg_loss = 0
        [optimizer , optimizer_ad] = optimizers
        total_correct = 0
        iter_num = len(train_loader)
        total = 0
        for i ,data in enumerate(train_loader):
            image , label = data
            image = image.cuda()
            label = label.cuda()
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
        return avg_loss / iter_num, float(total_correct) / total * 100

    def meta_val_loop(self,epoch,val_loader,classifier='emd'):
        acc = []
        classifier = 'emd' if not self.params.test_LR else 'LR'
        for i,data in enumerate(val_loader):

            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            # print(support_xs.shape)
            # print(query_xs.shape)
            feat_sup = self.forward_feature(support_xs.squeeze(0))
            feat_qry = self.forward_feature(query_xs.squeeze(0))

            # print(feat_sup.shape)
            # tic = time.time()
            pred = self.LR(feat_sup,support_ys,feat_qry,query_ys)

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
            if support_xs.squeeze(0).shape[0] > split_size:
                feat_sup_ = []
                # print(support_xs.shape)
                for j in range(math.ceil(support_xs.squeeze(0).shape[0] / split_size)):
                    # print(support_xs.squeeze(0)[i*128:min((i+1)*128,support_xs.shape[1]),:,:,:].shape)
                    feat_sup_.append(self.forward_feature(
                        support_xs.squeeze(0)[j * split_size:min((j + 1) * split_size, support_xs.shape[1]), :, :,
                        :]).cpu())
                feat_sup = torch.cat(feat_sup_, dim=0)
            else:
                feat_sup = self.forward_feature(support_xs.squeeze(0))
            if query_xs.squeeze(0).shape[0] > split_size:
                feat_qry_ = []
                # print(support_xs.shape)
                for j in range(math.ceil(query_xs.squeeze(0).shape[0] / split_size)):
                    # print(support_xs.squeeze(0)[i*128:min((i+1)*128,support_xs.shape[1]),:,:,:].shape)
                    feat_qry_.append(self.forward_feature(
                        query_xs.squeeze(0)[j * split_size:min((j + 1) * split_size, query_xs.shape[1]), :, :,
                        :]).cpu())
                # print(feat_qry_[0].shape)
                feat_qry = torch.cat(feat_qry_, dim=0)
            else:
                feat_qry = self.forward_feature(query_xs.squeeze(0))

            pred = self.LR(feat_sup, support_ys, feat_qry, query_ys)
            if self.params.n_symmetry_aug > 1:
                pred = pred.view(-1, self.params.n_symmetry_aug)
                query_ys = query_ys.view(-1, self.params.n_symmetry_aug)
                pred = torch.mode(pred, dim=-1)[0]
                query_ys = torch.mode(query_ys, dim=-1)[0]
            acc_epo = np.mean(pred.cpu().numpy() == query_ys.numpy())
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
            out = self.forward_pretrain(image)
            loss_cls = F.cross_entropy(out, label)
            loss_div = loss_div_fn(out, out_t)
            loss = loss_cls * 0.5 + loss_div * 0.5

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
        clf = LR(penalty='l2',
                 random_state=0,
                 C=self.params.penalty_c,
                 solver='lbfgs',
                 max_iter=1000,
                 multi_class='multinomial')
        spt_norm = torch.norm(support_z, p=2, dim=1).unsqueeze(1).expand_as(support_z)
        spt_normalized = support_z.div(spt_norm)
        z_support = spt_normalized.detach().cpu().numpy()
        y_support = support_ys.view(-1).cpu().numpy()
        qry_norm = torch.norm(query_z, p=2, dim=1).unsqueeze(1).expand_as(query_z)
        qry_normalized = query_z.div(qry_norm)
        z_query = qry_normalized.detach().cpu().numpy()
        clf.fit(z_support, y_support)

        return torch.from_numpy(clf.predict(z_query))

if __name__ == '__main__':
    print(random_sample(5,125,25))