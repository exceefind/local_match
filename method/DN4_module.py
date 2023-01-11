import torch
import torch.nn as nn


class ImgtoClass_Metric(nn.Module):
    def __init__(self, neighbor_k=3):
        super(ImgtoClass_Metric, self).__init__()
        self.neighbor_k = neighbor_k

    # Calculate the k-Nearest Neighbor of each local descriptor
    def cal_cosinesimilarity(self, input1, input2):
        B, C, h, w = input1.size()
        Similarity_list = []

        for i in range(B):
            query_sam = input1[i]
            query_sam = query_sam.view(C, -1)
            query_sam = torch.transpose(query_sam, 0, 1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm

            if torch.cuda.is_available():
                inner_sim = torch.zeros(1, len(input2)).cuda()

            for j in range(len(input2)):
                support_set_sam = input2[j]
                support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
                support_set_sam = support_set_sam / support_set_sam_norm

                # cosine similarity between a query sample and a support category
                innerproduct_matrix = query_sam @ support_set_sam

                # choose the top-k nearest neighbors
                topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
                inner_sim[0, j] = torch.sum(topk_value)

            Similarity_list.append(inner_sim)

        Similarity_list = torch.cat(Similarity_list, 0)

        return Similarity_list

    def forward(self, x1, x2):

        Similarity_list = self.cal_cosinesimilarity(x1, x2)

        return Similarity_list




if __name__ == '__main__':
    q = torch.rand((75,512,5,5)).cuda()
    input2 = [torch.rand((5,512,5,5)).cuda() for i in range(5)]
    S = []
    for i in range(5):
        support_set_sam = input2[i]
        B, C, h, w = support_set_sam.size()
        support_set_sam = support_set_sam.permute(1, 0, 2, 3)
        support_set_sam = support_set_sam.contiguous().view(C, -1)
        S.append(support_set_sam)

    img2class = ImgtoClass_Metric().cuda()
    print(img2class(q,S).shape)