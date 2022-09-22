#!/usr/bin/env Python
# coding=utf-8
'''
MAP of Image to Text: 0.578, MAP of Text to Image: 0.572
'''
import torch
import numpy as np
import scipy.io as scio
import h5py
from data.base_dataset import BaseDataset

class WIKI(torch.utils.data.Dataset):
    def __init__(self, data, train=True):
        super().__init__()
        if train:
            self.labels = data.train_label_set
            self.index = data.train_index
            self.txt = data.train_txt_set
            self.img = data.train_img_set
        else:
            self.labels = data.test_label_set
            self.index = data.test_index
            self.txt = data.test_txt_set
            self.img = data.test_img_set

    def __getitem__(self, index):
        target = self.labels[index]
        txt = self.txt[index]
        img = self.img[index]
        return img, txt, target, index  # image-->(4096,)(ndarray)  txt-->(1386,)(ndarray)target-->(24,)(ndarray) index-->(0~70)

    def __len__(self):
        return len(self.labels)


