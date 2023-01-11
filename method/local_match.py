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
from model.resnet12 import resnet12
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

class Local_match(nn.Module):
    def __init__(self,params,num_classes = 5, local_proto =3 ,pretrain=False,  mask_ad =False):
        super(Local_match, self).__init__()
        self.backbone = resnet12(avg_pool=True,num_classes=64)
        if params.deep_emd:
            print("model is pure fcn deep_emd!")
            params.local_mode = 'cell'
            params.include_bg = True
            params.beta = 0

        if params.drop_gama == 1:
            params.beta = 0
        if params.test_LR:
            params.local_mode = 'mask_pool'

        # if pretrain:
        #     state_dict = torch.load(os.path.join('checkpoint',params.model_dir))
        #     self.backbone.load_state_dict(state_dict)
        self.in_feature = self.backbone.classifier.in_features
        self.local_proto = local_proto
        self.num_classes = num_classes
        self.ADL_sig = ADL_sig(mask_ad, gama=params.drop_gama)
        self.ADL = ADL_variant(gama=params.drop_gama)
        if params.MLP_2:
            hidden_dim = 128
            self.classifier = nn.Sequential(
                nn.Linear(self.in_feature, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_classes))
            self.classifier_disc =  nn.Sequential(
                nn.Linear(self.in_feature, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_classes))
        else:
            self.classifier = nn.Linear(self.in_feature, num_classes)
            self.classifier_disc = nn.Linear(self.in_feature, num_classes)
        self.SFC  = torch.nn.Parameter(torch.rand(self.num_classes,3,self.in_feature),requires_grad=True)
        # deep_emd方式


        self.params = params

    def forward(self,x):
        [feat,feat_map],logit = self.backbone(x, is_feat=True)
        out = []
        start_lab = 0 if self.params.include_bg else 1
        for j in range(feat_map.shape[0]):
            feat_map_j = feat_map[j, :, :, :].unsqueeze(0)
            # print(feat_map_j.shape)
            mask = self.ADL(feat_map_j, is_importance_area=True)
            local_mask = torch.from_numpy(bwlabel(mask.cpu().numpy()))
            local_feat = []
            for i in range(start_lab, torch.max(local_mask) + 1):
                local_mask_i = torch.where(local_mask == i, 1, 0).unsqueeze(1).expand(feat_map_j.shape)
                feat_mask_i = torch.sum(torch.sum(feat_map_j.cpu() * local_mask_i, dim=-1), dim=-1) / torch.sum(
                    torch.sum(local_mask_i, dim=-1),
                    dim=-1)
                local_feat.append(feat_mask_i)
                # print(local_feat.shape)
            local_feat = torch.cat(local_feat, dim=0).to(feat_map_j.device)
            score = compute_match_scores(local_feat,self.SFC)
            out.append(score.unsqueeze(0))
        return torch.cat(out, dim=0).to(feat_map.device)

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

    def init_sfc(self, SFC, proto_local_list, support_y):

        for i in range(self.params.n_way):
            clf = KMeans(n_clusters=self.local_proto-1, random_state=0,max_iter=1000)
            sample_idxs = torch.where(support_y == i)[0]
            local_feat =[]
            # SFC[i, 0, :] = 0
            for idx in sample_idxs:
                # print(proto_local_list[idx].shape)
                # SFC[i, 0, :] += proto_local_list[idx].squeeze(0)[0, :].cpu()
                # local_feat.append(proto_local_list[idx].squeeze(0)[1:,:])
                local_feat.append(proto_local_list[idx].squeeze(0))

            # print(len(sample_idxs))
            # SFC[i, 0, :] = torch.nn.Parameter(SFC[i, 0, :].float() / len(sample_idxs),requires_grad=True).cuda()
            local_feat = torch.cat(local_feat, dim=0)
            # SFC[i, 1:, :] = torch.nn.Parameter(
            #     torch.from_numpy(clf.fit(local_feat.detach().cpu().numpy()).cluster_centers_),
            #     requires_grad=True).cuda()
            SFC[i, :, :].data = torch.nn.Parameter(
                torch.from_numpy(clf.fit(local_feat.detach().cpu().numpy()).cluster_centers_),
                requires_grad=True).cuda()
        return SFC

    def forward_pretrain(self,x):
        [feat, feat_map], logit = self.backbone(x,is_feat=True)
        fore_feat, back_feat, map_sig = self.ADL_sig(feat_map, area_control=True)
        if self.params.deep_emd or self.params.drop_gama ==  1:
            fore_feat = feat
        fore_out = self.classifier(fore_feat.view(fore_feat.size(0),-1))
        back_out = self.classifier_disc(back_feat.clone().view(back_feat.size(0),-1))
        back_out_detach = self.classifier_disc(back_feat.detach().view(back_feat.size(0),-1))
        return fore_out,back_out,back_out_detach,map_sig

    def get_weight_vector(self, A, B):
        #  A : n_batch * n_local * in_dim

        # print(A.shape)
        # print(B.shape)

        M = A.shape[0]
        N = B.shape[0]

        # B = torch.mean(B, dim=1).unsqueeze(1)
        B = B[:,0,:].unsqueeze(1)
        B = B.repeat( 1, A.shape[1], 1)

        A = A.unsqueeze(1)
        B = B.unsqueeze(0)

        A = A.repeat(1, N, 1, 1)
        B = B.repeat(M, 1, 1, 1)

        # print(A.shape)
        combination = (A * B).sum(3)
        combination = combination.view(M, N, -1)
        # print(combination)
        # ---------------**update : increase base value 1e-3 --> 100 ----------------------------
        combination = F.relu(combination) + 1e-3
        # combination = torch.abs(combination)
        # print(combination)
        # time.sleep(10)
        return combination

    def get_similiarity_map(self, proto, query):
        way = proto.shape[0]
        num_query = query.shape[0]
        # query = query.view(query.shape[0], query.shape[1], -1)
        # proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        # proto = proto.permute(0, 1, 3, 2)
        # query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        # print(proto.shape)
        if self.params.metric == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)
        if self.params.metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map

        return similarity_map

    def get_sfc(self,support_xs, support_ys, ):
        support_ys = support_ys.squeeze(0).cuda()
        if self.params.deep_emd:
            support_xs_t = torch.cat(support_xs, dim=0)
            SFC = support_xs_t.view(-1, self.params.n_shot, support_xs_t.shape[-2], support_xs_t.shape[-1]).mean(
                dim=1).clone().detach()
            # print(support_xs_t.view(self.params.n_shot, -1, support_xs_t.shape[-2], support_xs_t.shape[-1])[0:4,0,0,0])
            SFC = nn.Parameter(SFC.detach(), requires_grad=True)
            # return SFC
        else:
            SFC = torch.nn.Parameter(torch.rand(self.params.n_way,self.local_proto,self.in_feature),requires_grad=True)
            # SFC = torch.nn.Parameter(torch.zeros(self.params.n_way, self.local_proto, self.in_feature).float(),)
            SFC = self.init_sfc(SFC,support_xs,support_ys).cuda()
            SFC.requires_grad_(True)
        # print(torch.sum(torch.isnan(SFC)))
        # print(self.params.sfc_lr)
        optimizer = torch.optim.SGD([SFC], lr=self.params.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0, )
        Schuler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.params.sfc_update_step,eta_min=1)
        Schuler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.4)
        # optimizer = torch.optim.SGD([SFC], lr=self.params.sfc_lr, momentum=0.9, )
        with torch.enable_grad():

            for step in range(0, self.params.sfc_update_step):
                rand_aug = random_sample(self.params.n_aug_support_samples, len(support_xs),len(support_xs)//self.params.n_aug_support_samples)
                random.shuffle(rand_aug)
                for i in range(0, self.params.n_way * self.params.n_shot, self.params.sfc_bs):
                    selected_id = rand_aug[i: min(i + self.params.sfc_bs, self.params.n_way * self.params.n_shot)]
                    # print(support_ys)
                    batch_label = support_ys[selected_id]
                    logits = []
                    for j in range(len(selected_id)):
                        batch_shot = support_xs[selected_id[j]].unsqueeze(0)
                        # print(torch.sum(torch.isnan(batch_shot)))
                        logit = self.forward_emd_1shot_loss( batch_shot.detach(), SFC,)
                        logits.append(logit)
                        # print(logit)
                    logits = torch.cat(logits,dim=0)
                    # print(logits)

                    # print(batch_label)
                    # print(logits)
                    # time.sleep(5)
                    loss = F.cross_entropy(logits, batch_label)
                    # loss = ce_loss(logits, batch_label)
                    # print(torch.sum(torch.isnan(loss)))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Schuler.step()
        return SFC


    def forward_emd_1shot_loss(self, qury_list, proto,  ):
        logits = []
        # print(len(qury_list))
        for query in qury_list:
            # weight_1:  n_batch * n_way * n_local
            # print(proto.shape)
            weight_1 = self.get_weight_vector(query, proto)
            weight_2 = self.get_weight_vector(proto, query)

            # print(torch.sum(torch.isnan(proto)))
            # print(torch.sum(torch.isnan(weight_2)))

            proto = self.normalize_feature(proto)
            query = self.normalize_feature(query)

            # sim_map: n_batch * n_way * num_local * num_porto
            similarity_map = self.get_similiarity_map(proto, query)
            logit = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv')
            logits.append(logit)
        return torch.cat(logits,dim=0)

    def forward_emd_1shot_inference(self,  qury_list, proto_list, ):
        logits = []

        for q in range(len(qury_list)):
            logit = []
            # print(len(proto_list))
            for p in range(len(proto_list)):

                query = qury_list[q]
                proto = proto_list[p]
                # query = query.unsqueeze(0)
                # proto = proto.unsqueeze(0)
                # print(query.shape)
                # print(proto.shape)
                weight_1 = self.get_weight_vector(query, proto)
                weight_2 = self.get_weight_vector(proto, query)

                proto = self.normalize_feature(proto)
                query = self.normalize_feature(query)


                # sim_map: n_batch * n_way * num_local * num_porto
                similarity_map = self.get_similiarity_map(proto, query)

                weight_1 = weight_1.squeeze(0).squeeze(0)
                weight_2 = weight_2.squeeze(0).squeeze(0)
                num_node = weight_1.shape[-1]
                # print(weight_1)
                # print(weight_2)


                similarity_map = similarity_map.squeeze(0).squeeze(0)

                # print(weight_1.shape)
                # print(weight_2.shape)
                # print(similarity_map.shape)
                _, flow = emd_inference_opencv(1 - similarity_map, weight_1, weight_2)
                similarity_map = (similarity_map) * torch.from_numpy(flow).cuda()
                temperature = (self.params.temperature / num_node)
                logit.append( similarity_map.sum(-1).sum(-1).unsqueeze(0) * temperature)
            # print(logit)
            logit = torch.cat(logit,dim=0).unsqueeze(0)
            logits.append(logit)
        scores = torch.cat(logits,dim=0)
        return scores

    def forward_local(self,x,is_cell=False):
        [feat, feat_map], logit = self.backbone(x,is_feat=True)
        if self.params.local_mode == 'cell_mix' :
            is_cell = True
        out = []
        start_lab = 0 if self.params.include_bg else 1
        for j in range(feat_map.shape[0]):
            feat_map_j = feat_map[j, :, :, :].unsqueeze(0)
            mask = self.ADL(feat_map_j, is_importance_area=True)
            local_mask = torch.from_numpy(bwlabel(mask.cpu().numpy()))
            if self.params.local_mode == 'mask_pool':
                local_mask_ = torch.where(local_mask.view(-1) != 0)[0]
                if self.params.drop_gama == 1:
                    local_mask_ = range(0,25)
                feat_mask_j = feat_map_j.view(feat_map_j.shape[0], feat_map_j.shape[1], -1)[:, :, local_mask_]
                # if is_cell is False or (self.params.local_mode == 'cell_mix' and i != 0):
                feat_mask_j = F.adaptive_avg_pool1d(feat_mask_j, 1)
                out.append(feat_mask_j.squeeze(-1))
            else:
                local_feat = []
                if self.params.feature_pyramid:
                    local_feat.append(F.adaptive_avg_pool1d(feat_map_j.view(feat_map_j.shape[0], feat_map_j.shape[1], -1),1).permute(0, 2, 1))
                    local_feat.append(feat_map_j.view(feat_map_j.shape[0],feat_map_j.shape[1],-1).permute(0,2,1))

                for i in range(start_lab, torch.max(local_mask) + 1):
                    # local_mask_i = torch.where(local_mask == i, 1, 0).unsqueeze(1).expand(feat_map_j.shape)
                    local_mask_idx = torch.where(local_mask.view(-1) == i)[0]
                    feat_mask_i = feat_map_j.view(feat_map_j.shape[0],feat_map_j.shape[1],-1)[:,:,local_mask_idx]
                    if is_cell is False or (self.params.local_mode == 'cell_mix' and i != 0):

                        feat_mask_i = F.adaptive_avg_pool1d(feat_mask_i,1)
                    # feat_mask_i : 1 * 640 *  1 if not is_cell else n_cell_of_local

                    local_feat.append(feat_mask_i.permute(0,2,1))
                local_feat = torch.cat(local_feat, dim=1).to(feat_map_j.device)
                # print(local_feat.shape)
                out.append(local_feat)
        return out

    def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]

        num_node = weight_1.shape[-1]
        assert num_node!=0
        # print(weight_1.shape)
        # print(weight_2.shape)
        # print(similarity_map.shape)
        # time.sleep(10)
        if solver == 'opencv':  # use openCV solver
            for i in range(num_query):
                for j in range(num_proto):
                    _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])

                    similarity_map[i, j, :, :] = (similarity_map[i, j, :, :]) * torch.from_numpy(flow).cuda()

            temperature = (self.params.temperature / num_node)
            logitis = similarity_map.sum(-1).sum(-1) * temperature
            return logitis
        elif solver == 'qpth':
            # print(weight_2.shape)
            # print(similarity_map.shape)
            weight_2 = weight_2.permute(1, 0, 2)
            similarity_map = similarity_map.view(num_query * num_proto, similarity_map.shape[-2],
                                                 similarity_map.shape[-1])
            weight_1 = weight_1.view(num_query * num_proto, weight_1.shape[-1])
            weight_2 = weight_2.reshape(num_query * num_proto, weight_2.shape[-1])

            _, flows = emd_inference_qpth(1 - similarity_map, weight_1, weight_2,form=self.params.form, l2_strength=self.params.l2_strength)

            logitis=(flows*similarity_map).view(num_query, num_proto,flows.shape[-2],flows.shape[-1])
            temperature = (self.params.temperature / num_node)
            logitis = logitis.sum(-1).sum(-1) *  temperature
        else:
            raise ValueError('Unknown Solver')

        return logitis

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
            fore_out, back_out, back_out_detach, map_sig = self.forward_pretrain(image)
            loss_ce = F.cross_entropy(fore_out, label)

            loss_ent = entropy_loss(back_out)
            loss = loss_ce - (self.params.beta * loss_ent if self.params.deep_emd is False else 0)
            avg_loss = avg_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not self.params.deep_emd:
                loss_disc = F.cross_entropy(back_out_detach, label)
                optimizer_ad.zero_grad()
                loss_disc.backward()
                optimizer_ad.step()

            _, pred = torch.max(fore_out, 1)
            correct = (pred == label).sum().item()
            total_correct += correct
            total += label.size(0)
            # if i % print_step == 0:
            #     print(' Epoch {:d} | Batch: {:d}/{:d} | Loss: {:.4f} | Acc_train: {:.2f}'.format(epoch, i, len(train_loader),
            #                                                             avg_loss / float(i + 1),correct/label.shape[0]*100), end=' ')

        return avg_loss / iter_num, float(total_correct) / total * 100

    def meta_val_loop(self,epoch,val_loader,classifier='emd'):
        mode = self.params.local_mode  # model in ['cell', 'local_mix']
        acc = []
        classifier = 'emd' if not self.params.test_LR else 'LR'
        for i,data in enumerate(val_loader):
            tic = time.time()
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            # print(support_xs.shape)
            # print(query_xs.shape)
            proto_local = self.forward_local(support_xs.squeeze(0), is_cell=True if mode == 'cell' else False)
            qry_local = self.forward_local(query_xs.squeeze(0),is_cell=True if mode == 'cell' else False)

            if classifier == 'emd':
                if self.params.n_shot>1:
                    proto_local = self.get_sfc(proto_local,support_ys)
                    scores = self.forward_emd_1shot_loss(qry_local, proto_local)

                else:
                    scores = self.forward_emd_1shot_inference(qry_local, proto_local)
                _, pred = torch.max(scores, dim=1)
            elif classifier == 'LR':
                proto_local = torch.cat(proto_local,dim=0)
                qry_local = torch.cat(qry_local,dim=0)
                pred = self.LR(proto_local,support_ys,qry_local,query_ys)

            # print(np.mean(pred.cpu().numpy() == query_ys.numpy()))
            acc_epo = np.mean(pred.cpu().numpy() == query_ys.numpy())
            acc.append(acc_epo)
            # print("\repisode {} acc: {:.2f} | avg_acc: {:.2f} +- {:.2f}, elapse : {:.2f}".format(i,acc_epo*100,*mean_confidence_interval(acc,multi=100),(time.time()-tic)/60),end='')

        return mean_confidence_interval(acc)

    def meta_test_loop(self,test_loader):
        mode = self.params.local_mode  # model in ['cell', 'local_mix']
        acc = []
        classifier = 'emd' if not self.params.test_LR else 'LR'
        for i, data in enumerate(test_loader):
            tic = time.time()
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            # print(support_xs.shape)
            # print(query_xs.shape)
            proto_local = self.forward_local(support_xs.squeeze(0), is_cell=True if mode == 'cell' else False)
            qry_local = self.forward_local(query_xs.squeeze(0), is_cell=True if mode == 'cell' else False)

            if classifier == 'emd':
                if self.params.n_shot > 1 or self.params.n_aug_support_samples > 1:
                    proto_local = self.get_sfc(proto_local, support_ys)
                    scores = self.forward_emd_1shot_loss(qry_local, proto_local)

                else:
                    scores = self.forward_emd_1shot_inference(qry_local, proto_local)
                _, pred = torch.max(scores, dim=1)
            elif classifier == 'LR':
                proto_local = torch.cat(proto_local, dim=0)
                qry_local = torch.cat(qry_local, dim=0)
                pred = self.LR(proto_local, support_ys, qry_local, query_ys)

            # print(np.mean(pred.cpu().numpy() == query_ys.numpy()))
            acc_epo = np.mean(pred.cpu().numpy() == query_ys.numpy())
            acc.append(acc_epo)
            print("\repisode {} acc: {:.2f} | avg_acc: {:.2f} +- {:.2f}, elapse : {:.2f}".format(i, acc_epo * 100,
                                                                                                 *mean_confidence_interval(
                                                                                                     acc, multi=100), (
                                                                                                             time.time() - tic) / 60),
                  end='')

        return mean_confidence_interval(acc)

    def LR(self,support_z,support_ys,query_z,query_ys):
        clf = LR(penalty='l2',
                 random_state=0,
                 C=1.0,
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

def bwlabel(bw):
    res = np.zeros_like(bw)
    for i in range(bw.shape[0]):
        label = 1
        # print(bw[i,:,:].size)
        equal_set = np.ones(bw[i, :, :].size) * -1

        for j in range(bw.shape[1]):
            for k in range(bw.shape[2]):
                if bw[i, j, k] == 1:
                    lab_min = label
                    if k > 0 and bw[i, j, k - 1] == 1:
                        if lab_min >= res[i, j, k - 1]:
                            lab_min = res[i, j, k - 1]
                        else:
                            equal_set[res[i, j, k - 1]] = lab_min
                        # lab_min = res[i, j, k - 1]

                    if j > 0 and bw[i, j - 1, k] == 1:
                        if lab_min >= res[i, j - 1, k]:
                            lab_min = res[i, j - 1, k]
                        else:
                            equal_set[res[i, j - 1, k]] = lab_min

                    if j > 0 and k > 0 and bw[i, j - 1, k - 1] == 1:
                        if lab_min >= res[i, j - 1, k - 1]:
                            lab_min = res[i, j - 1, k - 1]
                        else:
                            equal_set[res[i, j - 1, k - 1]] = lab_min

                    if j > 0 and k + 1 < bw.shape[2] and bw[i, j - 1, k + 1] == 1:
                        if lab_min >= res[i, j - 1, k + 1]:
                            lab_min = res[i, j - 1, k + 1]
                        else:
                            equal_set[res[i, j - 1, k + 1]] = lab_min

                    res[i, j, k] = lab_min
                    if lab_min == label:
                        label += 1

        label_set = np.zeros(equal_set.shape)
        count = 1
        for a in range(len(equal_set)):
            if equal_set[a] == -1 and a != 0:
                label_set[a] = count
                count += 1

        for j in range(bw.shape[1]):
            for k in range(bw.shape[2]):
                res[i, j, k] = find_root(equal_set, label_set, int(res[i, j, k]))
    # print(equal_set)
    return res

def find_root(a, label_set, i):
    while a[i] != -1:
        i = int(a[i])
    return label_set[i]

if __name__ == '__main__':
    print(random_sample(5,125,25))