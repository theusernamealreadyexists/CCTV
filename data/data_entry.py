import torch
from data.MSCOCO import MSCOCO
from data.WIKI import WIKI
from data.NUS_WIDE import NUS
from data.MIR_Flickr import MIR
import os.path as osp

import data.base_dataset

def get_dataset(args):

    if args.dataset == 'MSCOCO':
        train_path = osp.join(args.dataset_path, 'coco_train.mat')
        test_path = osp.join(args.dataset_path, 'coco_test.mat')
        dataset = data.base_dataset.BaseDataset(train_path, test_path)
        train_dataset = MSCOCO(dataset, True)
        test_dataset = MSCOCO(dataset, False)

    if args.dataset == 'WIKI':
        train_path = osp.join(args.dataset_path, 'wiki_train.mat')
        test_path = osp.join(args.dataset_path, 'wiki_test.mat')
        dataset = data.base_dataset.BaseDataset(train_path, test_path)
        train_dataset = WIKI(dataset, True)
        test_dataset = WIKI(dataset, False)

    if args.dataset == 'NUS':
        train_path = osp.join(args.dataset_path, 'nus_train.mat')
        test_path = osp.join(args.dataset_path, 'nus_test.mat')
        dataset = data.base_dataset.BaseDataset(train_path, test_path)
        train_dataset = NUS(dataset, True)
        test_dataset = NUS(dataset, False)

    if args.dataset == 'MIR':
        train_path = osp.join(args.dataset_path, 'mir_train.mat')
        test_path = osp.join(args.dataset_path, 'mir_test.mat')
        dataset = data.base_dataset.BaseDataset(train_path, test_path)
        train_dataset = MIR(dataset, True)
        test_dataset = MIR(dataset, False)

    return train_dataset, test_dataset


def get_loader(args):
    train_dataset, test_dataset = get_dataset(args)

    # base数据集用于初始化聚类中心以及验证时计算指标, shuffle = False
    static_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         pin_memory=False,
                                         num_workers=args.workers,
                                         drop_last=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.workers,
                                         drop_last=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.workers,
                                         drop_last=False)

    return static_loader, train_loader, test_loader
