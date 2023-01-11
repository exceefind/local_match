import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

class ADL(nn.Module):
    def __init__(self,drop_rate = 0.7,gama = 0.5):
        super(ADL, self).__init__()
        self.drop_rate = drop_rate
        self.gama = gama
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.sigmoid = torch.nn.Sigmoid()
        self.all_one = torch.tensor([1])



    def forward(self,feat_map,is_importance_area = False):
        device = feat_map.device
        # channel-wise Avgpool9oi654
        # map :  batch * h * w
        # drop_mask: mask importance 峰值位置
        att_map = torch.mean(feat_map,dim=1)
        # print(att_map)
        # avg_act: batch * 1
        avg_act = torch.mean(self.avg_pool(feat_map),dim=1)
        # print(avg_act)
        # select att_map >= avg_act
        # 高于平均值的面积
        # print(torch.where(att_map>=avg_act,1,0))
        rate_Sa = torch.sum(torch.sum(torch.where(att_map>=avg_act,1,0),dim=-1),dim=-1)/(att_map.shape[-2]*att_map.shape[-1])

        # drop (1-drop_gama) % area
        # rate_Sa = torch.ones_like(rate_Sa)
        drop_gama = 1 - rate_Sa * self.gama
        # drop_gama = 1 - self.gama
        # print(drop_gama)
        # 对attention att_map 采用sigmoid,得到 importance Map
        importance_map = self.sigmoid(att_map)

        # 找到前1-gama的位置
        value, ind = torch.sort(att_map.reshape(att_map.shape[0], -1), dim=1, descending=True)
        drop_ind = torch.floor((1-drop_gama) * att_map.shape[1] * att_map.shape[2])
        threshold = value[range(att_map.shape[0]),drop_ind.long()]
        # print(att_map)
        # print(torch.max(att_map.reshape(att_map.shape[0],-1),dim=-1)[0])
        # threshold = drop_gama * torch.max(att_map.reshape(att_map.shape[0],-1),dim=-1)[0]
        threshold = threshold.unsqueeze(1).unsqueeze(1).expand(att_map.shape)
        drop_mask = torch.where(att_map>=threshold,0,1)
        # print(drop_mask)
        # *************************
        # print(att_map)
        # *************************
        random_select =  torch.rand(drop_gama.shape[0])
        mask = torch.where(random_select>=self.drop_rate,1,0).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(feat_map.shape)
        importance_map = importance_map.unsqueeze(1).expand(feat_map.shape)
        # ************************
        # print(1 - drop_mask)
        # ************************

        if is_importance_area:

            return 1-drop_mask

        drop_mask = drop_mask.unsqueeze(1).expand(feat_map.shape)
        feat_map = torch.where(mask==1,importance_map.cpu()*feat_map.cpu(),drop_mask.cpu()*feat_map.cpu())
        feat_map = feat_map.to(device)
        # print(feat_map)

        #  torch.where(map>=avg_act,1,0)

        return feat_map

class ADL_variant(nn.Module):
    def __init__(self, drop_rate=0.7, gama=0.5):
        super(ADL_variant, self).__init__()
        self.drop_rate = drop_rate
        self.gama = gama
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.sigmoid = torch.nn.Sigmoid()
        self.all_one = torch.tensor([1])

    def forward(self, feat_map, is_importance_area=False):
        device = feat_map.device
        # channel-wise Avgpool
        # att_map : bs * 5 * 5
        # avg_act : bs * 1 * 1
        att_map = torch.mean(feat_map, dim=1)
        avg_act = torch.mean(self.avg_pool(feat_map), dim=1)
        # select att_map >= avg_act
        rate_Sa = torch.sum(torch.sum(torch.where(att_map >= avg_act, 1, 0), dim=-1), dim=-1) / (
                    att_map.shape[-2] * att_map.shape[-1])

        # drop (1-drop_gama) % area
        # rate_Sa = torch.ones_like(rate_Sa)
        drop_gama = 1 - rate_Sa * self.gama

        # 对attention att_map 采用sigmoid,得到 importance Map
        importance_map = self.sigmoid(att_map)

        # 找到前1-gama的位置
        value, ind = torch.sort(att_map.reshape(att_map.shape[0], -1), dim=1, descending=True)
        drop_ind = torch.floor((1 - drop_gama) * att_map.shape[1] * att_map.shape[2])
        threshold = value[range(att_map.shape[0]), drop_ind.long()]
        # threshold = drop_gama * torch.max(att_map.reshape(att_map.shape[0],-1),dim=-1)[0]
        threshold = threshold.unsqueeze(1).unsqueeze(1).expand(att_map.shape)
        fore_mask = torch.where(att_map >= threshold, 1, 0)

        # *************************
        # print(importance_map)
        # print(fore_mask)
        # ************************

        return fore_mask

class ADL_sig(nn.Module):
    def __init__(self,spatial_att=False,mask_ad=False,in_feature=640,gama=0.5):
        super(ADL_sig, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.spatial_att = spatial_att
        self.mask_ad = mask_ad
        self.gama = gama
        # self.scale = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        # ****************update 20221017******************
        # self.scale = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        # self.scale = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=False)
        if spatial_att:
            self.ca = channel_attention(in_channel=in_feature)
            self.conv = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3,stride=1,padding=1)

            # self.self_attention = self_attention_map(in_channels=in_feature)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self,feat_map,area_control=False):
        # *****************update: 20221018***************************
        if self.spatial_att:
            # feat_map = self.self_attention(feat_map)
            # att_map = torch.mean(feat_map, dim=1)
            # channel-wise attention :
            # print(self.ca(feat_map).shape)
            feat_map *= self.ca(feat_map)
            # spatial-wise attention :
            att_max = torch.max(feat_map,dim=1)[0].unsqueeze(1)
            att_mean = torch.mean(feat_map,dim=1).unsqueeze(1)
            att_mix = torch.cat([att_max,att_mean],dim=1)
            att_map = self.conv(att_mix).squeeze(1)
        # channel-wise Avgpool
        else:
            att_map = torch.mean(feat_map, dim=1)
        # if self.scale>4:
        # print(self.scale)
        # **********************************
        avg_act = torch.mean(self.avg_pool(feat_map), dim=1)
        rate_Sa = torch.sum(torch.sum(torch.where(att_map >= avg_act, 1, 0), dim=-1), dim=-1) / (
                att_map.shape[-2] * att_map.shape[-1])
        drop_gama = 1 - rate_Sa * self.gama
        value, ind = torch.sort(att_map.reshape(att_map.shape[0], -1), dim=1, descending=True)
        drop_ind = torch.floor((1 - drop_gama) * att_map.shape[1] * att_map.shape[2])
        threshold = value[range(att_map.shape[0]), drop_ind.long()]
        # threshold = drop_gama * torch.max(att_map.reshape(att_map.shape[0],-1),dim=-1)[0]
        threshold = threshold.unsqueeze(1).unsqueeze(1).expand(att_map.shape)
        drop_mask = torch.where(att_map >= threshold, 0, 1)
        # print(1-drop_mask)
        # print(ind.view(drop_mask.shape))
        # print(self.sigmoid(att_map))
        # **************************************************

        map_sig = self.sigmoid(att_map).unsqueeze(1).expand_as(feat_map)
        fore_feat = self.avg_pool(map_sig*feat_map)
        if self.mask_ad:
            drop_mask = drop_mask.unsqueeze(1).expand_as(map_sig)
            back_feat = self.avg_pool(drop_mask * (1 - map_sig) * feat_map)
        else:
            back_feat = self.avg_pool((1-map_sig)*feat_map)
        if area_control:
             return fore_feat,back_feat,map_sig

        return fore_feat,back_feat


class self_attention_map(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)

        return self.gamma * out

class channel_attention(nn.Module):
    def __init__(self,in_channel,hidden_c=8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveAvgPool2d(1)
        self.MLP = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // hidden_c, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channel // hidden_c, in_channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,feat_map):
        map_avg = self.MLP(self.avgpool(feat_map))
        map_max = self.MLP(self.maxpool(feat_map))
        return self.sigmoid(map_max+map_avg)


if __name__ == '__main__':
    adl = ADL_variant()
    a = torch.randn((3,4,5,5))
    b = torch.ones((3,4)).unsqueeze(-1).unsqueeze(-1)
    # print(a*b)
    # print( adl(a))
    # print(adl(a,is_importance_area=True))
