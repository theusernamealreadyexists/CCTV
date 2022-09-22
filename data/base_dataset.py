#!/usr/bin/env Python
# coding=utf-8
'''
MAP of Image to Text: 0.578, MAP of Text to Image: 0.572
'''
import torch
import numpy as np
import scipy.io as scio
import h5py

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, train_path, test_path):
        self.set_train = scio.loadmat(train_path)
        self.set_test = scio.loadmat(test_path)

        self.test_index = np.array(self.set_test['id_test'], dtype=np.int).squeeze()  # 1*2000
        self.test_label_set = np.array(self.set_test['label_test'], dtype=np.int8).squeeze()  # 2000*81
        self.test_txt_set = np.array(self.set_test['text_test'], dtype=np.float).squeeze() # 2000*512
        self.test_img_set = np.array(self.set_test['image_test'], dtype=np.float32).squeeze()  # 2000*4096

        self.train_index = np.array(self.set_train['id_train'], dtype=np.uint8).squeeze()  # 1*65000
        self.train_label_set = np.array(self.set_train['label_train'], dtype=np.int).squeeze()  # 65000*81
        self.train_txt_set = np.array(self.set_train['text_train'], dtype=np.float).squeeze()  # 65000*512
        self.train_img_set = np.array(self.set_train['image_train'], dtype=np.float32).squeeze()  # 65000*4096
