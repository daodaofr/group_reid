import pickle
import numpy as np
import torch
import os
import random
from PIL import Image

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def relabel(labels):
    labels_all = []
    for label in labels:
        labels_all += label
    u_label = list(set(labels_all))
    if "-1" in u_label:
        u_label.remove("-1")
    new_label = []
    class_p = 0
    for label in labels:
        t_new_label = []
        for ll in label:
            if ll == "-1":
                t_new_label.append(-1)
            else:
                idx = u_label.index(ll)
                t_new_label.append(idx)
        class_p = max(class_p, max(t_new_label))
        new_label.append(t_new_label)

    return new_label, class_p+1

def relabel_gid(labels):
    labels_all = list(set(labels))
    new_label = []
    for label in labels:
        new_label.append(labels_all.index(label))
    #class_g = len(new_label)
    class_g = len(labels_all)
    return new_label, class_g

class CUHKGroup(object):

    def __init__(self, datafile, dataroot, rlb=False, transform=None, transform_p=None):
        super(CUHKGroup, self).__init__()
        with open(datafile, 'rb') as f:
            self.all_im_name, self.all_group_id, self.all_group_pid, self.all_group_bbox = pickle.load(f)
        self.dataroot = dataroot
        self.transform = transform
        self.transform_p = transform_p
        self.relabel = rlb
        self.max_num = 5

        if self.relabel:
            self.all_group_pid, self.num_train_pids = relabel(self.all_group_pid)
            self.all_group_id, self.num_train_gids = relabel_gid(self.all_group_id)
        #self.num_train_gids = len(set(self.all_group_id))


    def __len__(self):
        return len(self.all_im_name)

    def __getitem__(self, index):
        im_name = os.path.join(self.dataroot, self.all_im_name[index])
        group_id = self.all_group_id[index]
        group_pid = self.all_group_pid[index]
        group_bbox = self.all_group_bbox[index]
        tmp_pid = []
        #tmp_pid_shuffle = []
        len_p = self.max_num if len(group_pid) > self.max_num else len(group_pid)

        img = read_image(im_name)
        box_g = [[], [], [], []] 
        if self.relabel:
            imgs_p = []
            #imgs_p_shuffle = []
            while len(group_pid) < self.max_num:
                group_pid.append(-1)
                group_bbox.append(group_bbox[-1])
            if len(group_pid) > self.max_num:
                group_pid = group_pid[:self.max_num]
            for i, pid in enumerate(group_pid):
                tmp_pid.append(pid)
                tmp_bbox = group_bbox[i]
                tmp_pimg = img.crop((tmp_bbox[0], tmp_bbox[1], tmp_bbox[0] + tmp_bbox[2], tmp_bbox[1] + tmp_bbox[3]))
                box_g[0].append(tmp_bbox[0])
                box_g[1].append(tmp_bbox[1])
                box_g[2].append(tmp_bbox[0] + tmp_bbox[2])
                box_g[3].append(tmp_bbox[1] + tmp_bbox[3])
                # tmp_pimg.show()
                if self.transform_p is not None:
                    tmp_pimg = self.transform_p(tmp_pimg)
                tmp_pimg = tmp_pimg.unsqueeze(0)
                imgs_p.append(tmp_pimg)
            if -1 in tmp_pid:
                len_idx = tmp_pid.index(-1)
            else:
                len_idx = self.max_num
            rand_idx = list(range(len_idx))
            random.shuffle(rand_idx)
            imgs_p_shuffle = [imgs_p[i] for i in rand_idx]
            if len(imgs_p_shuffle) < self.max_num:
                for i in range(len_idx, self.max_num):
                    imgs_p_shuffle.append(imgs_p[i])
            tmp_pid_shuffle = [tmp_pid[i] for i in rand_idx]
            if len(tmp_pid_shuffle) < self.max_num:
                for i in range(len_idx, self.max_num):
                    tmp_pid_shuffle.append(tmp_pid[i])
            imgs_p_shuffle = torch.cat(imgs_p_shuffle, dim=0)
            #print(tmp_pid)
            #print(tmp_pid_shuffle)
            
            img = img.crop((min(box_g[0]), min(box_g[1]), max(box_g[2]), max(box_g[3])))
            if self.transform is not None:
                img = self.transform(img)
            return img, group_id, imgs_p_shuffle, tmp_pid_shuffle, index
        else:
            imgs_p = []
            #tmp_pid = []
            while len(group_pid) < self.max_num:
                group_pid.append("-1")
                group_bbox.append(group_bbox[-1])
            if len(group_pid) > self.max_num:
                group_pid = group_pid[:self.max_num]
            #print(group_pid)
            #print(group_bbox)
            for i, pid in enumerate(group_pid):
                tmp_pid.append(pid)
                #print(i)
                #print(len(group_bbox))
                tmp_bbox = group_bbox[i]
                tmp_pimg = img.crop((tmp_bbox[0], tmp_bbox[1], tmp_bbox[0] + tmp_bbox[2], tmp_bbox[1] + tmp_bbox[3]))
                box_g[0].append(tmp_bbox[0])
                box_g[1].append(tmp_bbox[1])
                box_g[2].append(tmp_bbox[0] + tmp_bbox[2])
                box_g[3].append(tmp_bbox[1] + tmp_bbox[3])
                if self.transform_p is not None:
                    tmp_pimg = self.transform_p(tmp_pimg)
                tmp_pimg = tmp_pimg.unsqueeze(0)
                imgs_p.append(tmp_pimg)
            imgs_p = torch.cat(imgs_p, dim=0)

            img = img.crop((min(box_g[0]), min(box_g[1]), max(box_g[2]), max(box_g[3])))
            if self.transform is not None:
                img = self.transform(img)

            return img, group_id, imgs_p, tmp_pid, index, len_p

