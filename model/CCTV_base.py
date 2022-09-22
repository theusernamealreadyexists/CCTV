#!/usr/bin/env Python
# coding=utf-8
import torch.nn as nn
import models
from sklearn.cluster import KMeans
import os.path as osp
import scipy

from utils.loss import *
from utils.metrics import *


class CCTV(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        self.CodeNet_I = models.VAE_img(
            n_enc_img_1=2048, n_enc_img_2=1024,
            n_dec_img_1=1024, n_dec_img_2=2048,
            n_input=4096, n_z=args.code_len, n_clusters=args.cluster_number,
            alpha=1.0)
        self.CodeNet_T = models.VAE_txt(
            n_enc_txt_1=256, n_enc_txt_2=128,
            n_dec_txt_1=128, n_dec_txt_2=256,
            n_input=512, n_z=args.code_len, n_clusters=args.cluster_number,
            alpha=1.0)

        self.opt_I = torch.optim.Adam(self.CodeNet_I.parameters(), lr=args.lr_img, weight_decay=args.weight_decay)
        self.opt_T = torch.optim.Adam(self.CodeNet_T.parameters(), lr=args.lr_txt, weight_decay=args.weight_decay)

        self.warmup = {'beta': {'factor': 0.25,
                                'end_epoch': 93,
                                'start_epoch': 0},
                       'cross_reconstruction': {'factor': 2.37,
                                                'end_epoch': 75,
                                                'start_epoch': 21},
                       'distance': {'factor': 8.13,
                                    'end_epoch': 22,
                                    'start_epoch': 6}}

        self.CodeNet_I = self.CodeNet_I.cuda()
        self.CodeNet_T = self.CodeNet_T.cuda()
        self.logger = logger
        self.max_mapi2t = 0
        self.max_mapt2i = 0

    def train(self, args, logger, static_loader, train_loader, test_loader, static_dataset, test_dataset, simi):
        self.pretrain(args, train_loader)
        # cluster parameter initiate
        kmeans_img = KMeans(n_clusters=args.cluster_number, n_init=args.code_len)
        kmeans_txt = KMeans(n_clusters=args.cluster_number, n_init=args.code_len)
        mu_all_img = []
        mu_all_txt = []
        for index, (F_I, F_T, _, _) in enumerate(static_loader):
            F_I = F_I.cuda()
            F_T = F_T.float().cuda()
            mu_img, _ = self.CodeNet_I.vae_img_enc(F_I)
            mu_all_img.append(mu_img.detach().cpu())
            mu_txt, _ = self.CodeNet_T.vae_txt_enc(F_T)
            mu_all_txt.append(mu_txt.detach().cpu())
        kmeans_img.fit_predict(torch.cat(mu_all_img).numpy())
        kmeans_txt.fit_predict(torch.cat(mu_all_txt).numpy())

        # txt的分布几乎差不多
        cluster_centers_img = torch.tensor(kmeans_img.cluster_centers_, dtype=torch.float, requires_grad=True).cuda(
            non_blocking=True)
        cluster_centers_txt = torch.tensor(kmeans_txt.cluster_centers_, dtype=torch.float, requires_grad=True).cuda(
            non_blocking=True)

        self.CodeNet_I.state_dict()["cluster_layer_img"].copy_(cluster_centers_img)  # 观察这里随着学习是不是会改变
        self.CodeNet_T.state_dict()["cluster_layer_txt"].copy_(cluster_centers_txt)

        para_list = select_para(args)

        for epoch in range(args.num_epoch):
            self.CodeNet_I.train()
            self.CodeNet_T.train()

            self.CodeNet_I.set_kappa(epoch)
            self.CodeNet_T.set_kappa(epoch)

            # data_iter = tqdm(train_loader)

            for batch_idx, (F_I, F_T, _, idx) in enumerate(train_loader):
                F_I = F_I.cuda()
                F_T = F_T.float().cuda()
                self.opt_I.zero_grad()
                self.opt_T.zero_grad()

                mu_img, logvar_img, img_from_img, z_img, q_img = self.CodeNet_I(F_I)
                mu_txt, logvar_txt, txt_from_txt, z_txt, q_txt = self.CodeNet_T(F_T)
                code_img = torch.tanh(self.CodeNet_I.kappa * mu_img)
                code_txt = torch.tanh(self.CodeNet_T.kappa * mu_txt)

                img_from_txt = self.CodeNet_I.vae_img_dec(z_txt)
                txt_from_img = self.CodeNet_T.vae_txt_dec(z_img)

                # loss
                cada_loss, max_val1, max_val1, min_val1, min_val2, KLD= cada_loss_func(epoch, self.warmup, logvar_img,
                                                                                   logvar_txt, mu_img, mu_txt,
                                                                                   img_from_img,
                                                                                   txt_from_txt, img_from_txt,
                                                                                   txt_from_img, F_I, F_T)
                hash_loss = hash_loss_func(F_I, F_T, code_img, code_txt, para_list, args)
                dec_loss = dec_loss_func(q_img, q_txt)

                loss = para_list['ALPHA_DEC'] * dec_loss + para_list['ALPHA_HASH'] * hash_loss + para_list[
                    'ALPHA_CADA'] * cada_loss
                loss.backward()

                self.opt_I.step()
                self.opt_T.step()

                if (batch_idx + 1) % (len(static_dataset) // args.batch_size / args.epoch_interval) == 0:
                    self.logger.info(
                        'batch_idx: %d, Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f Loss3: %.4f Total Loss: '
                        '%.4f max_val1:%.4f, max_val1:%.4f, min_val1:%.4f, min_val2:%.4f, KLD:%.4f '
                        % (batch_idx, epoch + 1, args.num_epoch, batch_idx + 1, len(static_dataset) // args.batch_size,
                           cada_loss.item(), hash_loss.item(), dec_loss.item(), loss.item(), max_val1, max_val1, min_val1, min_val2, KLD))

            if (epoch + 1) % args.eval_interval == 0:
                self.eval(args, static_loader, test_loader, static_dataset, test_dataset, simi, epoch)

    def eval(self, args, static_loader, test_loader, static_dataset, test_dataset, simi, epoch=0):
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval()
        self.CodeNet_T.eval()

        if args.dataset == "WIKI":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(static_loader, test_loader,
                                                                   self.CodeNet_I, self.CodeNet_T,
                                                                   static_dataset, test_dataset,
                                                                   args.code_len, args.batch_size)

        else:
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(static_loader, test_loader, self.CodeNet_I,
                                                              self.CodeNet_T, static_dataset, test_dataset,
                                                              args.code_len, args.batch_size)

        mapi2t = calc_map(qu_BI, re_BT, qu_L, re_L)
        mapt2i = calc_map(qu_BT, re_BI, qu_L, re_L)
        self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (mapi2t, mapt2i))
        # i2t_pre, i2t_recall = pr(qu_BI, re_BT, qu_L, re_L)
        # t2i_pre, t2i_recall = pr(qu_BT, re_BI, qu_L, re_L)
        i2t_pre, i2t_recall = precision_recall(qu_BI, re_BT, simi)
        t2i_pre, t2i_recall = precision_recall(qu_BT, re_BI, simi)
        # self.logger.info('pre of Image to Text: ' + str(i2t_pre))
        # self.logger.info('recall of Image to Text:' + str(i2t_recall))
        # self.logger.info('pre of Text to Image: ' + str(t2i_pre))
        # self.logger.info('recall of Text to Image: ' + str(t2i_recall))
        self.logger.info('pre of Image to Text: ' + str(i2t_pre))
        self.logger.info('recall of Image to Text:' + str(i2t_recall))
        self.logger.info('pre of Text to Image: ' + str(t2i_pre))
        self.logger.info('recall of Text to Image: ' + str(t2i_recall))
        self.logger.info('--------------------------------------------------------------------')

        if mapi2t + mapt2i >= self.max_mapi2t + self.max_mapt2i:
            self.max_mapi2t = mapi2t
            self.max_mapt2i = mapt2i
            self.best_epoch = epoch
            self.logger.info('MAX MAP of Image to Text: %.3f, MAP of Text to Image: %.3f at epoch %d' % (
            self.max_mapi2t, self.max_mapt2i, epoch))
            hashcode_path = osp.join(args.snapshot_dir, 'CCTV_{}.mat'.format(args.code_len))
            scipy.io.savemat(hashcode_path, mdict={'code_test_I': qu_BI, 'code_test_T': qu_BT,
                                                   'code_train_I': re_BI, 'code_train_T': re_BT, })
            self.save_checkpoints(args)

    def pretrain(self, args, train_loader):
        img_path = '{}_{}_{}_img.pkl'.format(args.dataset, args.code_len, args.model)
        txt_path = '{}_{}_{}_txt.pkl'.format(args.dataset, args.code_len, args.model)

        self.CodeNet_I.pretrain_img(osp.join(args.pretrain_dir, img_path), train_loader, args)
        self.CodeNet_T.pretrain_txt(osp.join(args.pretrain_dir, txt_path), train_loader, args)

    def save_checkpoints(self, args):
        ckp_path = osp.join(args.model_dir, 'model', '{}.pth'.format(args.exp_name))
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, args):
        ckp_path = osp.join(args.model_dir, 'model', '{}.pth'.format(args.exp_name))
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        # self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])
