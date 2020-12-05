from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
from bisect import bisect_right
import random


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=0.01,
            warmup_iters=20.,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
        # print(self.last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                # print(self.last_epoch)
                alpha = (self.last_epoch + 1) / self.warmup_iters
                # print(alpha)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
                # print(warmup_factor)
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22. 
- This version accepts distance matrix instead of raw features. 
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

Modified by Zhedong Zheng, 2018-1-12.
- replace sort with topK, which save about 30s.
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""

import numpy as np


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0)
    #original_dist = 2. - 2 * original_dist  # np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    # initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition(original_dist, range(1, k1 + 1))

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)

    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

def build_adj(pids):
    adj = []
    for i in range(pids.shape[0]):
        tmp_pid = pids[i]
        tmp_idx = np.argwhere(tmp_pid.numpy() == -1)
        #print(tmp_idx)
        if len(tmp_idx) > 0:
            adj.append(np.ones((tmp_idx[0][0], tmp_idx[0][0])))
        else:
            adj.append(np.ones((pids.shape[1], pids.shape[1])))
    return adj

def build_pairs(imgs, gids, pimgs, pids, adj_new):
    bs = imgs.size(0)
    ss = 2
    imgs1 = imgs
    gids1 = gids
    pimgs1 = pimgs
    pids1 = pids
    adj1 = adj_new

    idx = []
    siamese_target = np.zeros((bs)).astype(float)
    #siamese_target = np.zeros((bs))
    for i in range(bs//2):
        is_pair = random.randint(0,1)
        if is_pair:
            siamese_target[2*i] = 1
            siamese_target[2*i + 1] = 0
            idx.append(2*i + 1)
            tmp_idx = list(range(bs))
            tmp_idx.remove(2*i)
            tmp_idx.remove(2*i + 1)
            random.shuffle(tmp_idx)
            idx.append(tmp_idx[0])
        else:
            siamese_target[2 * i] = 0
            siamese_target[2 * i + 1] = 1
            tmp_idx = list(range(bs))
            tmp_idx.remove(2 * i)
            tmp_idx.remove(2 * i + 1)
            random.shuffle(tmp_idx)
            idx.append(tmp_idx[0])
            idx.append(2 * i)

    imgs2 = [imgs[i].unsqueeze(0) for i in idx]
    imgs2 = torch.cat(imgs2, 0)

    pimgs2 = [pimgs[i].unsqueeze(0) for i in idx]
    pimgs2 = torch.cat(pimgs2, 0)

    #print(idx)
    adj2 = [adj_new[i] for i in idx]

    siamese_target = torch.from_numpy(siamese_target)

    gids2 = torch.zeros_like(gids)
    for i, ind in enumerate(idx):
        gids2[i] = gids[ind]

    pids2 = torch.zeros_like(pids)
    for i, ind in enumerate(idx):
        pids2[i] = pids[ind]

    #print(imgs1.shape, imgs2.shape, gids1.shape, gids2.shape, pimgs1.shape, pimgs2.shape, pids1.shape, pids2.shape, siamese_target.shape)
    #print(idx, gids1, gids2, siamese_target)

    return imgs1, imgs2, gids1, gids2, pimgs1, pimgs2, pids1, pids2, adj1, adj2, siamese_target

def build_pairs_correspondence(imgs, gids, pimgs, pids, adj_new):
    bs = imgs.size(0)
    ss = 2
    imgs1 = imgs
    gids1 = gids
    pimgs1 = pimgs
    pids1 = pids
    adj1 = adj_new

    idx = []
    siamese_target = np.zeros((bs)).astype(float)
    #siamese_target = np.zeros((bs))
    for i in range(bs//2):
        is_pair = random.randint(0,1)
        if is_pair:
            siamese_target[2*i] = 1
            siamese_target[2*i + 1] = 0
            idx.append(2*i + 1)
            tmp_idx = list(range(bs))
            tmp_idx.remove(2*i)
            tmp_idx.remove(2*i + 1)
            random.shuffle(tmp_idx)
            idx.append(tmp_idx[0])
        else:
            siamese_target[2 * i] = 0
            siamese_target[2 * i + 1] = 1
            tmp_idx = list(range(bs))
            tmp_idx.remove(2 * i)
            tmp_idx.remove(2 * i + 1)
            random.shuffle(tmp_idx)
            idx.append(tmp_idx[0])
            idx.append(2 * i)

    imgs2 = [imgs[i].unsqueeze(0) for i in idx]
    imgs2 = torch.cat(imgs2, 0)

    pimgs2 = [pimgs[i].unsqueeze(0) for i in idx]
    pimgs2 = torch.cat(pimgs2, 0)

    #print(idx)
    adj2 = [adj_new[i] for i in idx]

    siamese_target = torch.from_numpy(siamese_target)

    gids2 = torch.zeros_like(gids)
    for i, ind in enumerate(idx):
        gids2[i] = gids[ind]

    pids2 = torch.zeros_like(pids)
    for i, ind in enumerate(idx):
        pids2[i] = pids[ind]

    #print("#############")
    #print(pids1)
    #print(pids2)  
    sinkhorn_target = np.zeros((pids1.shape[0], pids1.shape[1], pids2.shape[1])).astype(float)
    for i in range(pids1.shape[0]):
        if siamese_target[i] > 0:
            for j in range(pids1.shape[1]):
                if pids1[i][j] > -1:
                    idx = (pids2[i] == pids[i][j]).nonzero()
                    sinkhorn_target[i, j, idx] = 1
    sinkhorn_target = torch.from_numpy(sinkhorn_target).float()
    #for i in range(sinkhorn_target.shape[0]):
    #    print(sinkhorn_target[i])
                 

    #print(imgs1.shape, imgs2.shape, gids1.shape, gids2.shape, pimgs1.shape, pimgs2.shape, pids1.shape, pids2.shape, siamese_target.shape)
    #print(idx, gids1, gids2, siamese_target)

    return imgs1, imgs2, gids1, gids2, pimgs1, pimgs2, pids1, pids2, adj1, adj2, siamese_target, sinkhorn_target

def build_triplets(imgs, gids, pimgs, pids, adj_new):
    bs = imgs.size(0)
    idx1 = []
    idx2 = []
    siamese_target = np.zeros((bs)).astype(float)
    # siamese_target = np.zeros((bs))
    for i in range(bs):
        for j in range(bs):
            idx1.append(i)
            idx2.append(j)

    imgs1 = [imgs[i].unsqueeze(0) for i in idx1]
    imgs1 = torch.cat(imgs1, 0)

    imgs2 = [imgs[i].unsqueeze(0) for i in idx2]
    imgs2 = torch.cat(imgs2, 0)

    pimgs1 = [pimgs[i].unsqueeze(0) for i in idx1]
    pimgs1 = torch.cat(pimgs1, 0)

    pimgs2 = [pimgs[i].unsqueeze(0) for i in idx2]
    pimgs2 = torch.cat(pimgs2, 0)

    adj1 = [adj_new[i] for i in idx1]
    adj2 = [adj_new[i] for i in idx2]

    gids1 = []
    for i in idx1:
        gids1.append(gids[i])
    gids1 = torch.stack(gids1)

    gids2 = []
    for i in idx2:
        gids2.append(gids[i])
    gids2 = torch.stack(gids2)

    pids1 = []
    for i in idx1:
        pids1.append(pids[i].unsqueeze(0))
    pids1 = torch.cat(pids1, 0)

    pids2 = []
    for i in idx2:
        pids2.append(pids[i].unsqueeze(0))
    pids2 = torch.cat(pids2, 0)

    return imgs1, imgs2, gids1, gids2, pimgs1, pimgs2, pids1, pids2, adj1, adj2
