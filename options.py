#!/usr/bin/env Python
# coding=utf-8

import logging
import os.path as osp
import time
import numpy as np
import argparse

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for training")
    parser.add_argument('--scripts', type=str, default='/home/cs4007/data/zy/master/CCTV-new/new/scripts/config/MSCOCO.yml',
                        help='hyperparameters', )
    parser.add_argument('--lr_pretrain_img', type=float, default=3e-5)
    parser.add_argument('--lr_pretrain_txt', type=float, default=1e-3)
    parser.add_argument('--lr_img', type=float, default=1e-5)
    parser.add_argument('--lr_txt', type=float, default=1e-5)
    parser.add_argument('--cluster_number', type=int, default=10)
    parser.add_argument('--code_len', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='MSCOCO')
    parser.add_argument('--dataset_path', type=str, default='/home/cs4007/data/zy/dataset/CrossModal')
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--pre_num_epoch', type=int, default=200)
    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epoch_interval', type=int, default=2)
    parser.add_argument('--model_dir', type=str, default='./checkpoint', help='model save path')
    parser.add_argument('--model', type=str, default='CCTV')
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--result_dir', default='./result')
    parser.add_argument('--snapshot_dir', default='',help = 'result_dir/exp_name')
    parser.add_argument('--pretrain', type=bool, default=True, help='need pretrain or not')
    parser.add_argument('--pretrain_dir', type=str, default='')
    parser.add_argument('--random_train', type=bool, default=False)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--eval', type=bool, default=False, help='only eval or not')

    return parser.parse_args()

