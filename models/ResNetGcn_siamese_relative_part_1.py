from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch import Tensor
import torchvision
import random
import math
__all__ = ['ResNet50Gcn_siamese_relative_part_1']
#from .resnet import ResNet, BasicBlock, Bottleneck

class Sinkhorn(nn.Module):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=10, epsilon=1e-4):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon

    def forward(self, s, nrows=None, ncols=None, exp=False, exp_alpha=20, dummy_row=False, dtype=torch.float32):
        batch_size = s.shape[0]

        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
            new_nrows = ncols
            for b in range(batch_size):
                s[b, nrows[b]:new_nrows[b], :ncols[b]] = self.epsilon
            nrows = new_nrows

        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
            col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        # for Sinkhorn stacked on last dimension
        if len(s.shape) == 4:
            row_norm_ones = row_norm_ones.unsqueeze(-1)
            col_norm_ones = col_norm_ones.unsqueeze(-1)

        s += self.epsilon

        for i in range(self.max_iter):
            if exp:
                s = torch.exp(exp_alpha * s)
            if i % 2 == 1:
                # column norm
                sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
            else:
                # row norm
                sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)

            tmp = torch.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        if dummy_row and dummy_shape[1] > 0:
            s = s[:, :-dummy_shape[1]]

        return s

class Voting(nn.Module):
    """
    Voting Layer computes a new row-stotatic matrix with softmax. A large number (alpha) is multiplied to the input
    stochastic matrix to scale up the difference.
    Parameter: value multiplied before softmax alpha
               threshold that will ignore such points while calculating displacement in pixels pixel_thresh
    Input: permutation or doubly stochastic matrix s
           ///point set on source image P_src
           ///point set on target image P_tgt
           ground truth number of effective points in source image ns_gt
    Output: softmax matrix s
    """
    def __init__(self, alpha=200, pixel_thresh=None):
        super(Voting, self).__init__()
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)  # Voting among columns
        self.pixel_thresh = pixel_thresh

    def forward(self, s, nrow_gt, ncol_gt=None):
        ret_s = torch.zeros_like(s)
        # filter dummy nodes
        for b, n in enumerate(nrow_gt):
            if ncol_gt is None:
                ret_s[b, 0:n, :] = \
                    self.softmax(self.alpha * s[b, 0:n, :])
            else:
                ret_s[b, 0:n, 0:ncol_gt[b]] =\
                    self.softmax(self.alpha * s[b, 0:n, 0:ncol_gt[b]])

        return ret_s

class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = int(d)
        self.A = Parameter(Tensor(self.d, self.d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d)

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        #M = torch.matmul(X, self.A)
        M = torch.matmul(X, (self.A + self.A.transpose(0, 1)) / 2)
        M = torch.matmul(M, Y.transpose(1, 2))
        return M

def sampler_fn(adj):
    n = adj.size(0)
    #print(adj.data)
    adj = adj.data>0
    n_max = adj.sum(dim=0).max() - 1
    nei = []
    for i in range(n):
        tmp = [j for j in range(n) if adj[i,j]>0 and j != i]
        if len(tmp) != n_max:
            while(len(tmp)<n_max):
                tmp += tmp
            random.shuffle(tmp)
            tmp = tmp[0:n_max]
        nei += tmp
    return nei

class BatchedGraphSAGEMean1(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True, mean=False, add_self=False):
        super(BatchedGraphSAGEMean1, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.aggregator = True
        #print(infeat,outfeat)
        self.W_x = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_x.weight, gain=nn.init.calculate_gain('relu'))

        self.W_neib = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_neib.weight, gain=nn.init.calculate_gain('relu'))

        self.W_relative = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_relative.weight, gain=nn.init.calculate_gain('relu'))

        if self.use_bn:
            self.bn = nn.BatchNorm1d(3*outfeat)
            #self.bn = nn.BatchNorm1d(16)

    def forward(self, x1, x2, adj1, adj2):
        #print(adj.shape)
        #print(x1.shape)
        #xshape = x1.shape
        #x1 = x1.view(x1.shape[0], x1.shape[1], -1)
        #x2 = x2.view(x2.shape[0], x2.shape[1], -1)
        #print(x1.shape)
        b = x1.size(0)
        parts = x1.size(2)
        #print(b)
        #print(len(adj))
        h_k_list1 = []
        h_k_list2 = []
        for i in range(b):
            # first graph in the pair
            sample_size1 = adj1[i].size(0)
            idx_neib1 = sampler_fn(adj1[i])
            x_neib1 = x1[i, idx_neib1, ].contiguous()
            x_neib1 = x_neib1.view(sample_size1, -1, x_neib1.size(1), x_neib1.size(2))
            x_neib1 = x_neib1.mean(dim=1)

            # second graph in the pair
            sample_size2 = adj2[i].size(0)
            idx_neib2 = sampler_fn(adj2[i])
            x_neib2 = x2[i, idx_neib2,].contiguous()
            x_neib2 = x_neib2.view(sample_size2, -1, x_neib2.size(1), x_neib2.size(2))
            x_neib2 = x_neib2.mean(dim=1)

            # calculate between graph message
            x1_valid = x1[i, :sample_size1, :]
            x2_valid = x2[i, :sample_size2, :]

            # concatenate part features
            x1_valid = x1_valid.view(x1_valid.shape[0], -1)
            x2_valid = x2_valid.view(x2_valid.shape[0], -1)
            # to verify that the cosine similarity is implemented correctly
            '''
            cos = nn.CosineSimilarity()
            for ii in range(sample_size1):
                for jj in range(sample_size2):
                    sim = cos(x1_valid[ii].unsqueeze(0), x2_valid[jj].unsqueeze(0))
                    print(sim)
            '''
            x1_valid_s = x1_valid.unsqueeze(2).expand(sample_size1, x1_valid.size(1), sample_size2)
            x2_valid_s = x2_valid.permute(1, 0).contiguous()
            x2_valid_s = x2_valid_s.unsqueeze(0).expand(sample_size1, x1_valid.size(1), sample_size2)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            sim = cos(x1_valid_s, x2_valid_s)
            att1 = F.softmax(sim, dim=1)
            att2 = F.softmax(sim.transpose(1,0).contiguous(), dim=1)
            mu1 = x1_valid - torch.matmul(att1, x2_valid)
            mu2 = x2_valid - torch.matmul(att2, x1_valid)
            #print(mu1.shape, mu2.shape)
            mu1 = mu1.view(mu1.shape[0], parts, -1)
            mu2 = mu2.view(mu2.shape[0], parts, -1)

            # calculate within graph and inter graph message
            h_k1 = torch.cat((self.W_x(x1[i, :sample_size1, :]), self.W_neib(x_neib1), self.W_relative(mu1)), 2).unsqueeze(0)
            h_k_junk1 = torch.cat((self.W_x(x1[i, sample_size1:, :]), self.W_x(x1[i, sample_size1:, :]), self.W_x(x1[i, sample_size1:, :])), 2).unsqueeze(
                0)

            h_k2 = torch.cat((self.W_x(x2[i, :sample_size2, :]), self.W_neib(x_neib2), self.W_relative(mu2)), 2).unsqueeze(0)
            h_k_junk2 = torch.cat((self.W_x(x2[i, sample_size2:, :]), self.W_x(x2[i, sample_size2:, :]), self.W_x(x2[i, sample_size2:, :])), 2).unsqueeze(
                0)

            h_k1 = torch.cat((h_k1, h_k_junk1), 1)
            h_k_list1.append(h_k1)

            h_k2 = torch.cat((h_k2, h_k_junk2), 1)
            h_k_list2.append(h_k2)

        h_k_f1 = torch.cat(h_k_list1, dim=0)
        h_k_f2 = torch.cat(h_k_list2, dim=0)
        #print(h_k_f.shape)

        h_k_f1 = F.normalize(h_k_f1, dim=3, p=2)
        h_k_f1 = F.relu(h_k_f1)
        h_k_f2 = F.normalize(h_k_f2, dim=3, p=2)
        h_k_f2 = F.relu(h_k_f2)
        #print(h_k_f1.shape)
        if self.use_bn:
            h_k_f1 = h_k_f1.view(-1, h_k_f1.size(2), h_k_f1.size(3))
            #self.bn = nn.BatchNorm1d(h_k.size(1))
            h_k_f1 = self.bn(h_k_f1.permute(0,2,1).contiguous())
            #print(h_k.shape)
            h_k_f1 = h_k_f1.permute(0, 2, 1)
            #print(h_k.shape)
            h_k_f1 = h_k_f1.view(b, -1, h_k_f1.size(1), h_k_f1.size(2))

            h_k_f2 = h_k_f2.view(-1, h_k_f2.size(2), h_k_f2.size(3))
            h_k_f2 = self.bn(h_k_f2.permute(0, 2, 1).contiguous())
            # print(h_k.shape)
            h_k_f2 = h_k_f2.permute(0, 2, 1)
            h_k_f2 = h_k_f2.view(b, -1, h_k_f2.size(1), h_k_f2.size(2))

        h_k_f1 = h_k_f1.contiguous()
        #print(h_k_f1.shape)
        h_k_f2 = h_k_f2.contiguous()
        #h_k_f1 = h_k_f1.view(xshape[0], xshape[1], xshape[2], -1)
        #h_k_f2 = h_k_f2.view(xshape[0], xshape[1], xshape[2], -1)
        return h_k_f1, h_k_f2


class ResNet50Gcn_siamese_relative_part_1(nn.Module):
    def __init__(self, num_classes, num_person_classes, loss={'xent'}, **kwargs):
        super(ResNet50Gcn_siamese_relative_part_1, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 2048
        self.part = 4

        # self.num_nodes = 3
        self.hidden_dim = 1024
        # self.adj = torch.ones((self.num_nodes, self.num_nodes))
        # self.adj.requires_gradient = False

        self.layers = nn.ModuleList([
            BatchedGraphSAGEMean1(self.feat_dim, self.hidden_dim),
            ])
        self.classifier = nn.Linear(3*self.part*self.hidden_dim, num_classes)
        #self.classifier_person = nn.Linear(3*self.hidden_dim, num_person_classes)
        self.classifier_person = nn.ModuleList([nn.Linear(3*self.hidden_dim, num_person_classes) for i in range(self.part)])

        self.affinity = Affinity(3*self.part*self.hidden_dim)
        self.voting_layer = Voting()
        self.bi_stochastic = Sinkhorn()

        '''
        self.layers = nn.ModuleList([
            BatchedGcnLayer(self.feat_dim, self.hidden_dim),
        ])
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.classifier_person = nn.Linear(self.hidden_dim, num_person_classes)
        '''

    def forward(self, x1, x2, adj1, adj2):
        feat1 = self.base(x1)
        feat2 = self.base(x2)
        #print(feat1.shape)
        
        #global_feat1 = self.gap(self.base(x1))
        #global_feat2 = self.gap(self.base(x2))
        '''
        f_p1 = global_feat1
        f_p1 = f_p1.view(f_p1.shape[0], -1)
        f_p2 = global_feat2
        f_p2 = f_p2.view(f_p2.shape[0] , -1)
        '''
        bs = len(adj1)

        part_feat1 = F.avg_pool2d(feat1, (int(feat1.size(-2) / self.part), feat1.size(-1))).squeeze()
        f1 = part_feat1.permute(0, 2, 1).contiguous().view(bs, int(part_feat1.shape[0] / bs), self.part, -1)
        #print(part_feat1.shape, f1.shape)
        part_feat2 = F.avg_pool2d(feat2, (int(feat2.size(-2) / self.part), feat2.size(-1))).squeeze()
        f2 = part_feat2.permute(0, 2, 1).contiguous().view(bs, int(part_feat2.shape[0] / bs), self.part, -1)
        #f1 = global_feat1.view(bs, int(global_feat1.shape[0] / bs), -1)
        #f2 = global_feat2.view(bs, int(global_feat2.shape[0] / bs), -1)

        for layer in self.layers:
            if isinstance(layer, BatchedGraphSAGEMean1):
                f1, f2 = layer(f1, f2, adj1, adj2)
       
        ns_src = []
        ns_tgt = []
        for adj in adj1:
            ns_src.append(adj.shape[0])
        for adj in adj2:
            ns_tgt.append(adj.shape[0])
        emb1 = f1.contiguous()
        emb1 = emb1.view(emb1.shape[0], emb1.shape[1], -1)
        emb2 = f2.contiguous()
        emb2 = emb2.view(emb2.shape[0], emb2.shape[1], -1)
        #print("emb1", emb1.shape, "emb2", emb2.shape)
        s = self.affinity(emb1, emb2)
        #print('affinity', s.shape)
        s = self.voting_layer(s, ns_src, ns_tgt)
        #print('voting', s.shape)
        s = self.bi_stochastic(s, ns_src, ns_tgt)        

        f_p1 = f1.contiguous()
        f_p1 = f_p1.view(f_p1.shape[0] * f_p1.shape[1], f_p1.shape[2], -1)
        #print(f_p1.shape)
        f_p2 = f2.contiguous()
        f_p2 = f_p2.view(f_p2.shape[0] * f_p2.shape[1], f_p2.shape[2], -1)

        # readout
        f1_list = []
        for i in range(f1.shape[0]):
            sample_size1 = adj1[i].size(0)
            f_tmp = torch.mean(f1[i, :sample_size1], 0)
            f_tmp = f_tmp.view(f_tmp.shape[0], -1)
            f1_list.append(f_tmp.unsqueeze(0))

        f1 = torch.cat(f1_list, 0)
        f1 = f1.view(f1.shape[0], -1)

        f2_list = []
        for i in range(f2.shape[0]):
            sample_size2 = adj2[i].size(0)
            f_tmp = torch.mean(f2[i, :sample_size2], 0)
            f_tmp = f_tmp.view(f_tmp.shape[0], -1)
            f2_list.append(f_tmp.unsqueeze(0))

        f2 = torch.cat(f2_list, 0)
        f2 = f2.view(f2.shape[0], -1)

        y_g1 = self.classifier(f1)
        y_g2 = self.classifier(f2)
        #y = self.classifier(f)
        y_p1_list = []
        f_p1_list = []
        for i in range(self.part):
            f_p1_list.append(f_p1[:, i, :])
            y_p1_list.append(self.classifier_person[i](f_p1[:,i,:]))

        y_p2_list = []
        f_p2_list = []
        for i in range(self.part):
            f_p2_list.append(f_p2[:, i, :])
            y_p2_list.append(self.classifier_person[i](f_p2[:,i,:]))

        if not self.training:
            return f1, f2, f_p1_list, f_p2_list

        #y_p1 = self.classifier_person(f_p1)
        #y_p2 = self.classifier_person(f_p2)

        #y_g1 = self.classifier(f1)
        #y_g2 = self.classifier(f2)

        return f1, f2, f_p1_list, f_p2_list, y_p1_list, y_p2_list, y_g1, y_g2, s
        
