import os
import sys
import time
import datetime
import argparse
import os.path as osp
import transforms as T
import numpy as np


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

from dataset import CUHKGroup
from utils import AverageMeter, Logger, save_checkpoint, WarmupMultiStepLR, re_ranking, build_adj, build_pairs, build_pairs_correspondence
from samplers import RandomIdentitySampler
from losses import CrossEntropyLabelSmooth, TripletLoss, ContrastiveLoss, TripletLossFilter, PermutationLoss
from eval_metrics import evaluate, evaluate_person
import models
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='cuhk-group')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224,
                    help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=224,
                    help="width of an image (default: 112)")
# Optimization options
parser.add_argument('--max-epoch', default=200, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=8, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--gallery-batch', default=16, type=int, help="gallery batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=70, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=2,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
parser.add_argument('--xent-only', type=bool, default=False,
                    help="if this is True, only xent loss is used in training")
parser.add_argument('--data-root', type=str, default='/raid/yy1/data/cuhk/Image/SSM', help='root path of the images')

# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50gcn_siamese_part_half_sink', help="resnet503d, resnet50tp, resnet50ta, resnetrnn")
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])
parser.add_argument('--pretrained-model', type=str, default='', help='for test')


# Miscs
parser.add_argument('--print-freq', type=int, default=100, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--evaluate', type=bool, default=False, help="evaluation only")
parser.add_argument('--eval-step', type=int, default=200,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--save-dir', type=str, default='log1')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--warmup', action='store_true', help='use warming up scheduler')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    #print("Initializing dataset {}".format(args.dataset))
    # dataset = data_manager.init_dataset(name=args.dataset)

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_train_p = T.Compose([
        T.Random2DTranslation(256, 128),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test_p = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_file = 'data/cuhk_train.pkl'
    test_file = 'data/cuhk_test.pkl'
    gallery_file = 'data/cuhk_gallery.pkl'
    data_root = args.data_root
    dataset_train = CUHKGroup(train_file, data_root, True, transform_train, transform_train_p)
    dataset_test = CUHKGroup(test_file, data_root, False, transform_test, transform_test_p)
    dataset_query = CUHKGroup(test_file, data_root, False, transform_test, transform_test_p)
    dataset_gallery = CUHKGroup(gallery_file, data_root, False, transform_test, transform_test_p)


    pin_memory = True if use_gpu else False

    if args.xent_only:
        trainloader = DataLoader(
        dataset_train,
        batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )
    else:
        trainloader = DataLoader(
            dataset_train,
            sampler=RandomIdentitySampler(dataset_train, num_instances=args.num_instances),
            batch_size=args.train_batch, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True,
        )

    queryloader = DataLoader(
        dataset_test,
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    querygalleryloader = DataLoader(
        dataset_query,
        batch_size=args.gallery_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    galleryloader = DataLoader(
        dataset_gallery,
        batch_size=args.gallery_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    print("Initializing model: {}".format(args.arch))
    if args.xent_only:
        # model = models.init_model(name=args.arch, num_classes=dataset_train.num_train_gids, loss={'xent'})
        model = models.init_model(name=args.arch, num_classes=dataset_train.num_train_gids, loss={'xent'})
    else:
        # model = models.init_model(name=args.arch, num_classes=dataset_train.num_train_gids, loss={'xent', 'htri'})
        model = models.init_model(name=args.arch, num_classes=dataset_train.num_train_gids,
                                  num_person_classes=dataset_train.num_train_pids, loss={'xent', 'htri'})

    #criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset_train.num_train_gids, use_gpu=use_gpu)
    #criterion_xent_person = CrossEntropyLabelSmooth(num_classes=dataset_train.num_train_pids, use_gpu=use_gpu)
    
    if os.path.exists(args.pretrained_model):
        print("Loading checkpoint from '{}'".format(args.pretrained_model))
        checkpoint = torch.load(args.pretrained_model)
        model_dict = model.state_dict()
        pretrain_dict = checkpoint['state_dict']
        pretrain_dict = {k:v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)

    criterion_xent = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_xent_person = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_htri = TripletLoss(margin=args.margin)
    criterion_pair = ContrastiveLoss(margin=args.margin)
    criterion_htri_filter = TripletLossFilter(margin=args.margin)
    criterion_permutation = PermutationLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)

    if args.stepsize > 0:
        if args.warmup:
            scheduler = WarmupMultiStepLR(optimizer, [200, 400, 600])
        else:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test_gcn_person_batch(model, queryloader, querygalleryloader, galleryloader, args.pool, use_gpu)
        #test_gcn_batch(model, queryloader, querygalleryloader, galleryloader, args.pool, use_gpu)
        #test_gcn(model, queryloader, galleryloader, args.pool, use_gpu)
        #test(model, queryloader, galleryloader, args.pool, use_gpu)
        return

    start_time = time.time()
    best_rank1 = -np.inf
    for epoch in range(start_epoch, args.max_epoch):
        #print("==> Epoch {}/{}  lr:{}".format(epoch + 1, args.max_epoch, scheduler.get_lr()[0]))

        train_gcn(model, criterion_xent, criterion_xent_person, criterion_pair, criterion_htri_filter, criterion_htri, criterion_permutation, optimizer, trainloader, use_gpu)
        #train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)

        if args.stepsize > 0: scheduler.step()

        if args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            rank1 = test_gcn_person_batch(model, queryloader, querygalleryloader, galleryloader, args.pool, use_gpu)
            #rank1 = test_gcn(model, queryloader, galleryloader, args.pool, use_gpu=False)
            #rank1 = test(model, queryloader, galleryloader, args.pool, use_gpu)
            is_best = rank1 > best_rank1
            if is_best: best_rank1 = rank1
            
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train_gcn(model, criterion_xent, criterion_xent_person, criterion_pair, criterion_htri_filter, criterion_htri, criterion_permutation, optimizer, trainloader, use_gpu):
    model.train()
    losses = AverageMeter()

    num_nodes = 5
    #adj = torch.ones((num_nodes, num_nodes))
    #if use_gpu:
    #    adj = adj.cuda()
    #adj = Variable(adj)
    #adj.requires_gradient = False

    for batch_idx, (imgs, gids, pimgs, pids, _) in enumerate(trainloader):

        pids = torch.cat(pids, dim=0)
        pids = pids.view(num_nodes, -1).permute(1, 0)

        # build per image adj
        adj = build_adj(pids)
        #print(adj)
        adj_new = []
        for adj0 in adj:
            adj0 = torch.from_numpy(adj0)
            if use_gpu:
                adj0 = adj0.cuda()
                adj0 = Variable(adj0)
                adj0.requires_gradient = False
            #print(adj0)
            adj_new.append(adj0)
        #print(adj_new)

        #imgs1, imgs2, gids1, gids2, pimgs1, pimgs2, pids1, pids2, adj1, adj2, siamese_target = build_pairs(imgs, gids, pimgs, pids, adj_new)
        imgs1, imgs2, gids1, gids2, pimgs1, pimgs2, pids1, pids2, adj1, adj2, siamese_target, sinkhorn_target = build_pairs_correspondence(imgs, gids, pimgs, pids, adj_new)

        pids1 = pids1.reshape(-1)
        pids2 = pids2.reshape(-1)
        #print(pids)
        if use_gpu:
            pimgs1, pids1, pimgs2, pids2 = pimgs1.cuda(), pids1.cuda(), pimgs2.cuda(), pids2.cuda()
            gids1, gids2 = gids1.cuda(), gids2.cuda()
            siamese_target = siamese_target.cuda()
            sinkhorn_target = sinkhorn_target.cuda()
        pimgs1, pids1 = Variable(pimgs1), Variable(pids1)
        pimgs2, pids2 = Variable(pimgs2), Variable(pids2)
        b, s, c, h, w = pimgs1.size()
        #pimgs = pimgs.permute(1, 0, 2, 3, 4).contiguous()
        pimgs1 = pimgs1.view(b * s, c, h, w)
        pimgs2 = pimgs2.view(b * s, c, h, w)

        #features = model(pimgs, adj)
        if args.xent_only:
            outputs = model(pimgs, adj_new)
            #loss = criterion_xent(outputs, pids)
            loss = criterion_xent(outputs, gids)
        else:
            # combine hard triplet loss with cross entropy loss
            features1, features2, features_p1, features_p2, outputs_p1, outputs_p2, outputs_g1, outputs_g2, sinkhorn_matrix = model(pimgs1, pimgs2, adj1, adj2)
            #print(outputs.size(), features.size())
            #xent_loss = criterion_xent(outputs, pids)
            #htri_loss = criterion_htri(features, pids)
            #xent_loss = criterion_xent(outputs, gids)
            #htri_loss = criterion_htri(features, gids)
            xent_loss_p1 = torch.sum(torch.stack([criterion_xent_person(o_p1, pids1) for o_p1 in outputs_p1]))
            xent_loss_p2 = torch.sum(torch.stack([criterion_xent_person(o_p2, pids2) for o_p2 in outputs_p2]))
            htri_loss_p1 = torch.sum(torch.stack([criterion_htri_filter(f_p1, pids1) for f_p1 in features_p1]))
            htri_loss_p2 = torch.sum(torch.stack([criterion_htri_filter(f_p2, pids2) for f_p2 in features_p2]))
            #xent_loss_p2 = criterion_xent_person(outputs_p2, pids2)
            htri_loss_g1 = criterion_htri(features1, gids1)
            htri_loss_g2 = criterion_htri(features2, gids2)
            xent_loss_g1 = criterion_xent(outputs_g1, gids1)
            xent_loss_g2 = criterion_xent(outputs_g2, gids2)
            pair_loss = criterion_pair(features1, features2, siamese_target)
            permutation_loss = criterion_permutation(sinkhorn_matrix, sinkhorn_target, adj1, adj2, siamese_target)
            #loss = xent_loss_p1 + xent_loss_p2 + pair_loss + xent_loss_g1 + xent_loss_g2
            #loss = xent_loss_p1 + xent_loss_p2 + pair_loss + xent_loss_g1 + xent_loss_g2 + htri_loss_g1 + htri_loss_g2 + 0.2*htri_loss_p1 + 0.2*htri_loss_p2
            loss = xent_loss_p1 + xent_loss_p2 + pair_loss + xent_loss_g1 + xent_loss_g2 + htri_loss_p1 + htri_loss_p2 + htri_loss_g1 + htri_loss_g2 + 0.05*permutation_loss
            #loss = xent_loss_p1 + xent_loss_p2 + pair_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))

def test_gcn(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):
    model.eval()

    q_pids, q_camids = [], []
    g_pids, g_camids = [], []

    for batch_idx, (_, gids, pimgs, pids, camids) in enumerate(queryloader):
        q_pids.extend(gids)
        q_camids.extend(camids)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    max_qcam = camids + 1

    for batch_idx, (_, gids, pimgs, pids, camids) in enumerate(galleryloader):
        g_pids.extend(gids)
        g_camids.extend(camids + max_qcam)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    m, n = q_pids.shape[0], g_pids.shape[0]
    distmat = torch.zeros((m, m + n))
    
    g_camids = np.concatenate((q_camids, g_camids), axis=0)
    g_pids = np.concatenate((q_pids, g_pids), axis=0) 

    with torch.no_grad():
        for batch_idx, (_, gids, pimgs, pids, camids) in enumerate(queryloader):
            if use_gpu:
                pimgs = pimgs.cuda()
            pimgs = Variable(pimgs)
            # b=1, n=number of clips, s=16
            b, s, c, h, w = pimgs.size()
            #pimgs = pimgs.permute(1, 0, 2, 3, 4)
            assert(b==1)
            pimgs = pimgs.view(s, c, h, w)
            num_nodes = s
            adj = torch.ones((num_nodes, num_nodes))
            if use_gpu:
                adj = adj.cuda()
            adj = Variable(adj)


            for batch_idx_q, (_, gids_q, pimgs_q, pids_q, camids_q) in enumerate(queryloader):
                if use_gpu:
                    pimgs_q = pimgs_q.cuda()
                pimgs_q = Variable(pimgs_q)
                # pimgs = pimgs.permute(1, 0, 2, 3, 4)
                b, s, c, h, w = pimgs_q.size()
                pimgs_q = pimgs_q.view(s, c, h, w)
                assert (b == 1)
                num_nodes = s
                adj_q = torch.ones((num_nodes, num_nodes))
                if use_gpu:
                    adj_q = adj_q.cuda()
                adj_q = Variable(adj_q)
                features1, features2 = model(pimgs, pimgs_q, [adj], [adj_q])
                #dist = torch.pow(features1, 2).sum(dim=1, keepdim=True) + \
                #          torch.pow(features2, 2).sum(dim=1, keepdim=True).t()
                #dist.addmm_(1, -2, features1, features2.t())
                #print(dist)
                dist = F.pairwise_distance(features1, features2)
                #print(dist)
                distmat[batch_idx, batch_idx_q] = dist

            for batch_idx_g, (_, gids_g, pimgs_g, pids_g, camids_g) in enumerate(galleryloader):
                if use_gpu:
                    pimgs_g = pimgs_g.cuda()
                pimgs_g = Variable(pimgs_g)
                # pimgs = pimgs.permute(1, 0, 2, 3, 4)
                b, s, c, h, w = pimgs_g.size()
                pimgs_g = pimgs_g.view(s, c, h, w)
                assert (b == 1)
                num_nodes = s
                adj_g = torch.ones((num_nodes, num_nodes))
                if use_gpu:
                    adj_g = adj_g.cuda()
                adj_g = Variable(adj_g)
                features1, features2 = model(pimgs, pimgs_g, [adj], [adj_g])
                #dist = torch.pow(features1, 2).sum(dim=1, keepdim=True) + \
                #          torch.pow(features2, 2).sum(dim=1, keepdim=True).t()
                #dist.addmm_(1, -2, features1, features2.t())
                #print(dist)
                dist = F.pairwise_distance(features1, features2)
                #print(dist)
                distmat[batch_idx, batch_idx_g + m] = dist
    distmat = distmat.numpy()
    #print(distmat)

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    '''
    dist_qq = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
              torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
    dist_qq.addmm_(1, -2, qf, qf.t())
    dist_qq = dist_qq.numpy()

    dist_gg = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
    dist_gg.addmm_(1, -2, gf, gf.t())
    dist_gg = dist_gg.numpy()

    dist_re_rank = re_ranking(distmat, dist_qq, dist_gg)

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(dist_re_rank, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")
    '''

    return cmc[0]


def test_gcn_person_batch(model, queryloader, querygalleryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):
    model.eval()

    g_bs = 16

    q_pids, q_pids_p, q_camids, q_camids_p = [], [], [], []
    g_pids, g_pids_p, g_camids, g_camids_p = [], [], [], []
    for batch_idx, (_, gids, pimgs, pids, camids, _) in enumerate(queryloader):
        q_pids.extend(gids)
        q_pids_p.extend(pids)
        q_camids.extend(camids)
        q_camids_p.extend([camids] * len(pids))
    #print(camids)
    q_pids = np.asarray(q_pids)
    q_pids_p = np.asarray(q_pids_p)
    q_pids_p = np.squeeze(q_pids_p)
    q_camids = np.asarray(q_camids)
    q_camids_p = np.asarray(q_camids_p)
    max_qcam = camids + 1
    print(q_pids.shape, q_pids_p.shape, q_camids.shape, q_camids_p.shape)

    for batch_idx, (_, gids, pimgs, pids, camids, _) in enumerate(querygalleryloader):
        g_pids.extend(gids)
        #print(gids, pids, camids)
        tmp_pids = []
        for j in range(g_bs):
            tmp_pids.append([])
            for i in range(len(pids)):
                tmp_pids[j].append(pids[i][j])
        # tmp_pids -> list g_bs * 5
        for i in range(g_bs):
            g_pids_p.extend(tmp_pids[i])
            #print(camids)
            #print(camids[i].item())
            g_camids.extend([camids[i]])
            g_camids_p.extend([camids[i]]* len(tmp_pids[i]))
        #g_camids_p.extend([camids]* len(pids))
    #g_pids = np.asarray(g_pids)
    #g_pids_p = np.asarray(g_pids_p)
    #g_camids = np.asarray(g_camids)
    #g_camids_p = np.asarray(g_camids_p)
    #print(g_pids.shape, g_pids_p.shape, g_camids.shape, g_camids_p.shape)

    for batch_idx, (_, gids, pimgs, pids, camids, _) in enumerate(galleryloader):
        g_pids.extend(gids)
        #print(gids, pids, camids)
        tmp_pids = []
        for j in range(g_bs):
            tmp_pids.append([])
            for i in range(len(pids)):
                tmp_pids[j].append(pids[i][j])
        # tmp_pids -> list g_bs * 5
        for i in range(g_bs):
            g_pids_p.extend(tmp_pids[i])
            #print(camids)
            #print(camids[i].item())
            g_camids.extend([camids[i]])
            g_camids_p.extend([camids[i] + max_qcam]* len(tmp_pids[i]))
        #g_camids_p.extend([camids]* len(pids))
    g_pids = np.asarray(g_pids)
    g_pids_p = np.asarray(g_pids_p)
    g_camids = np.asarray(g_camids)
    g_camids_p = np.asarray(g_camids_p)
    print(g_pids.shape, g_pids_p.shape, g_camids.shape, g_camids_p.shape)

    m, n = q_pids.shape[0], g_pids.shape[0]
    distmat = torch.zeros((m, n))

    m, n = q_pids_p.shape[0], g_pids_p.shape[0]
    distmat_p = torch.zeros((m, n))
    p_start = 0
    p_end = 0

    with torch.no_grad():
        for batch_idx, (_, gids, pimgs, pids, camids, lenp) in enumerate(queryloader):
            #if batch_idx < 1720:
            #    continue
            start_time = time.time()
            if use_gpu:
                pimgs = pimgs.cuda()
            pimgs = Variable(pimgs)
            # b=1, n=number of clips, s=16
            b, s, c, h, w = pimgs.size()
            #pimgs = pimgs.permute(1, 0, 2, 3, 4)
            assert(b==1)
            pimgs = pimgs.repeat(g_bs, 1, 1, 1, 1)
            pimgs = pimgs.view(g_bs*s, c, h, w)
            #pimgs = pimgs.view(s, c, h, w)
            num_nodes = s
            adj = []
            adj0 = torch.ones((lenp, lenp))
            if use_gpu:
                adj0 = adj0.cuda()
                adj0 = Variable(adj0)
                adj0.requires_gradient = False
            for aa in range(g_bs):
                adj.append(adj0)
            p_start = batch_idx * s
            p_end = (batch_idx + 1) * s
            #print(p_start, p_end)
            #print(batch_idx, g_bs, s)
            g_start = 0
            g_end = 0

            for batch_idx_g, (_, gids_g, pimgs_g, pids_g, camids_g, lenp_g) in enumerate(querygalleryloader):
                if use_gpu:
                    pimgs_g = pimgs_g.cuda()
                pimgs_g = Variable(pimgs_g)
                # pimgs = pimgs.permute(1, 0, 2, 3, 4)
                b, s, c, h, w = pimgs_g.size()
                pimgs_g = pimgs_g.view(b*s, c, h, w)
                #pimgs_g = pimgs_g.view(s, c, h, w)
                assert (b == g_bs)
                num_nodes = s
                adj_g = []
                for aa in range(g_bs):
                    adj1 = torch.ones((lenp_g[aa], lenp_g[aa]))
                    if use_gpu:
                       adj1 = adj1.cuda()
                    adj1 = Variable(adj1)
                    adj1.requires_gradient = False
                    adj_g.append(adj1)
                features1, features2, features_p1, features_p2 = model(pimgs, pimgs_g, adj, adj_g)
                #print(features_p1[0].shape, features_p2[0].shape)
                features_p1 = torch.cat(features_p1, dim=1)
                features_p2 = torch.cat(features_p2, dim=1)
                #print(features_p1.shape)
                dist_p = torch.pow(features_p1, 2).sum(dim=1, keepdim=True) + \
                          torch.pow(features_p2, 2).sum(dim=1, keepdim=True).t()
                dist_p.addmm_(1, -2, features_p1, features_p2.t())
                #p_end = p_start + dist_p.shape[0]
                #assert (p_end - p_start) == dist_p.shape[0]
                #print(p_end-p_start, dist_p.shape[0])
                g_end = g_start + dist_p.shape[1]
                #print(dist_p.shape)
                #print(features_p1.shape, features_p2.shape)
                #print(distmat_p[p_start:p_end, g_start:g_end].shape)
                #distmat_p[p_start:p_end, g_start:g_end] = dist_p
                for i in range(g_bs):
                    distmat_p[p_start:p_end, g_start+i*s:g_start+(i+1)*s] = dist_p[i*s:(i+1)*s, i*s:(i+1)*s]
                #distmat_p[p_start:p_end, g_start:g_end] = dist_p
                assert(g_end == g_start+(i+1)*s)
                g_start = g_end
                #print(dist)
                dist = F.pairwise_distance(features1, features2)
                #print(dist.shape)
                distmat[batch_idx,batch_idx_g*g_bs:(batch_idx_g+1)*g_bs] = dist
                #distmat[batch_idx, batch_idx_g] = dist
            #print(g_end)
            max_batch_idx_g = batch_idx_g + 1
            for batch_idx_g, (_, gids_g, pimgs_g, pids_g, camids_g, lenp_g) in enumerate(galleryloader):
                if use_gpu:
                    pimgs_g = pimgs_g.cuda()
                pimgs_g = Variable(pimgs_g)
                # pimgs = pimgs.permute(1, 0, 2, 3, 4)
                b, s, c, h, w = pimgs_g.size()
                pimgs_g = pimgs_g.view(b*s, c, h, w)
                #pimgs_g = pimgs_g.view(s, c, h, w)
                assert (b == g_bs)
                num_nodes = s
                adj_g = []
                for aa in range(g_bs):
                    adj1 = torch.ones((lenp_g[aa], lenp_g[aa]))
                    if use_gpu:
                       adj1 = adj1.cuda()
                    adj1 = Variable(adj1)
                    adj1.requires_gradient = False
                    adj_g.append(adj1)
                features1, features2, features_p1, features_p2 = model(pimgs, pimgs_g, adj, adj_g)
                #print(features_p1[0].shape, features_p2[0].shape)
                features_p1 = torch.cat(features_p1, dim=1)
                features_p2 = torch.cat(features_p2, dim=1)
                #print(features_p1.shape)
                dist_p = torch.pow(features_p1, 2).sum(dim=1, keepdim=True) + \
                          torch.pow(features_p2, 2).sum(dim=1, keepdim=True).t()
                dist_p.addmm_(1, -2, features_p1, features_p2.t())
                #p_end = p_start + dist_p.shape[0]
                #assert (p_end - p_start) == dist_p.shape[0]
                #print(p_end-p_start, dist_p.shape[0])
                g_end = g_start + dist_p.shape[1]
                #print(dist_p.shape)
                #print(features_p1.shape, features_p2.shape)
                #print(distmat_p[p_start:p_end, g_start:g_end].shape)
                #distmat_p[p_start:p_end, g_start:g_end] = dist_p
                for i in range(g_bs):
                    distmat_p[p_start:p_end, g_start+i*s:g_start+(i+1)*s] = dist_p[i*s:(i+1)*s, i*s:(i+1)*s]
                #distmat_p[p_start:p_end, g_start:g_end] = dist_p
                assert(g_end == g_start+(i+1)*s)
                g_start = g_end
                #print(dist)
                dist = F.pairwise_distance(features1, features2)
                #print(dist.shape)
                distmat[batch_idx, (max_batch_idx_g + batch_idx_g)*g_bs:(max_batch_idx_g + batch_idx_g+1)*g_bs] = dist
            #print(g_end)
            #p_start = p_end
            #print(batch_idx)
            end_time = time.time()
            print("image {:04d}, time : {:f}".format(batch_idx, end_time - start_time))
    distmat = distmat.numpy()
    distmat_p = distmat_p.numpy()
    #print(distmat)

    print("Computing CMC and mAP")
    print(distmat.shape, q_pids.shape, g_pids.shape, q_camids.shape, g_camids.shape)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    #cmc_p, mAP_p = evaluate_person(distmat_p, q_pids_p, g_pids_p, q_camids_p, g_camids_p)

    print("Group Reid Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    print(distmat_p.shape, q_pids_p.shape, g_pids_p.shape, q_camids_p.shape, g_camids_p.shape)
    cmc_p, mAP_p = evaluate_person(distmat_p, q_pids_p, g_pids_p, q_camids_p, g_camids_p)
    print("Person Reid Results ----------")
    print("mAP: {:.1%}".format(mAP_p))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc_p[r - 1]))
    print("------------------")

    return cmc[0]

if __name__ == '__main__':
    main()
