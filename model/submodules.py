#!/usr/bin/env Python
# coding=utf-8

import torch
import math
import torchvision
import torch.nn as nn
from torch.nn import Linear, ReLU, Tanh
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import itertools
import numpy as np
def reparameterize(mu, logvar):
    sigma = torch.exp(logvar/2)
    eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
    eps = eps.expand(sigma.size())
    return mu + sigma * eps

class VAE_encoder(nn.Module):
    def __init__(self, n_enc_img_1, n_enc_img_2, n_input, n_z):
        super(VAE_encoder, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, n_enc_img_1)
        self.enc_2 = Linear(n_enc_img_1, n_enc_img_2)
        self.mu_layer = Linear(n_enc_img_2, n_z)
        self.logvar_layer = Linear(n_enc_img_2, n_z)
        self.alpha = 1.0
    def forward(self, x):
        # encoder 必须加relu，否则就线性了
        enc_h1 = F.leaky_relu(self.enc_1(x))
        enc_h2 = F.leaky_relu(self.enc_2(enc_h1))
        mu = self.mu_layer(enc_h2)
        logvar = self.logvar_layer(enc_h2)
        return mu, logvar
    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class VAE_decoder(nn.Module):
    def __init__(self, n_dec_img_1, n_dec_img_2, n_input, n_z):
        super(VAE_decoder, self).__init__()
        # decoder
        self.dec_1 = Linear(n_z, n_dec_img_1)
        self.dec_2 = Linear(n_dec_img_1, n_dec_img_2)
        self.x_bar_layer = Linear(n_dec_img_2, n_input)
    def forward(self,z):
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)
        return x_bar

class VAE_img(nn.Module):
    def __init__(self, n_enc_img_1, n_enc_img_2, n_dec_img_1, n_dec_img_2, n_input, n_z, n_clusters, alpha,
                 pretrain_path_img='data/ae_coco_img_%d.pkl' % settings.CODE_LEN):  # code_len,
        super(VAE_img, self).__init__()
        self.alpha = alpha
        self.pretrain_path_img = pretrain_path_img
        self.vae_img_enc = VAE_encoder(
            n_enc_img_1=n_enc_img_1,
            n_enc_img_2=n_enc_img_2,
            n_input=n_input,
            n_z=n_z)
        self.vae_img_dec = VAE_decoder(
            n_dec_img_1 = n_dec_img_1,
            n_dec_img_2 = n_dec_img_2,
            n_input = n_input,
            n_z = n_z
        )
        self.cluster_layer_img = Parameter(torch.Tensor(n_clusters, n_z))  # , requires_grad=True
        torch.nn.init.xavier_normal_(self.cluster_layer_img.data)
        # self.fc_code = nn.Linear(n_z, code_len)

    def pretrain_img(self, path=''):
        if path == '':
            pretrain_vae_img(self.vae_img_enc, self.vae_img_dec)
        # load pretrain weights
        checkpoint = torch.load(self.pretrain_path_img, map_location='cuda:0')
        self.vae_img_enc.load_state_dict(checkpoint['enc'])
        self.vae_img_dec.load_state_dict(checkpoint['dec'])
        print('load pretrained ae from', path)

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(logvar/2)
        eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
        eps = eps.expand(sigma.size())
        return mu + sigma * eps

    def forward(self,x):
        mu, logvar = self.vae_img_enc(x)
        z = self.reparameterize(mu,logvar)
        x_bar = self.vae_img_dec(z)
        # cluster
        q = 1.0 / (1.0 + torch.sum(torch.pow(mu.unsqueeze(1) - self.cluster_layer_img, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return z, x_bar, q


class VAE_txt(nn.Module):
    def __init__(self, n_enc_txt_1, n_enc_txt_2, n_dec_txt_1, n_dec_txt_2, n_input, n_z, n_clusters, alpha,
                 pretrain_path_txt='data/ae_coco_txt_%d.pkl' % settings.CODE_LEN):  # code_len,
        super(VAE_txt, self).__init__()
        self.alpha = alpha
        # self.alpha_txt = 1.0
        self.pretrain_path_txt = pretrain_path_txt
        self.vae_txt_enc = VAE_encoder(
            n_enc_img_1=n_enc_txt_1,
            n_enc_img_2=n_enc_txt_2,
            n_input=n_input,
            n_z=n_z)
        self.vae_txt_dec = VAE_decoder(
            n_dec_img_1=n_dec_txt_1,
            n_dec_img_2=n_dec_txt_2,
            n_input=n_input,
            n_z=n_z
        )
        self.cluster_layer_txt = Parameter(torch.Tensor(n_clusters, n_z))  # , requires_grad=True
        torch.nn.init.xavier_normal_(self.cluster_layer_txt.data)
        # self.fc_code = nn.Linear(n_z, code_len)

    def pretrain_txt(self, path=''):
        if path == '':
            pretrain_vae_txt(self.vae_txt_enc, self.vae_txt_dec)
        # load pretrain weights
        checkpoint = torch.load(self.pretrain_path_txt, map_location='cuda:0')
        self.vae_txt_enc.load_state_dict(checkpoint['enc'])
        self.vae_txt_dec.load_state_dict(checkpoint['dec'])
        print('load pretrained ae from', path)

    def set_alpha_txt(self, epoch):
        self.alpha_txt = math.pow((1.0 * epoch + 1.0), 0.5)

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(logvar/2)
        eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
        eps = eps.expand(sigma.size())
        return mu + sigma * eps

    def forward(self, x):
        mu, logvar = self.vae_txt_enc(x)
        z = self.reparameterize(mu, logvar)
        x_bar = self.vae_txt_dec(z)
        # cluster
        q = 1.0 / (1.0 + torch.sum(torch.pow(mu.unsqueeze(1) - self.cluster_layer_txt, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # code_layer = self.fc_code(z)
        # code = torch.tanh(self.alpha_img * code_layer)
        # code = torch.tanh(self.alpha_txt * mu)
        return z, x_bar, q

def pretrain_vae_img(enc,dec):
    print(enc)
    print(dec)
    optimizer = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=settings.LR_IMG_pretrain_ae)
    # index_list = [i for i in range(len(F_I))]
    for epoch in range(400):
        total_loss = 0.
        for batch_idx, (F_I, _, _, _) in enumerate(kk.train_loader):
            F_I = Variable(F_I.float().to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
            # F_I = torch.Tensor(F_I).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            mu, log = enc(F_I)
            z = reparameterize(mu, log)
            x_bar = dec(z)
            loss = F.mse_loss(x_bar, F_I)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(batch_idx, ' ', labels, ' ', idx)
        print("img-->epoch {} loss={:.4f}".format(epoch, total_loss / (batch_idx+1)))
        state = {'enc':enc.state_dict(), 'dec':dec.state_dict()}
        torch.save(state, settings.pretrain_path_coco_img)
    print("img-->model saved to {}.".format(settings.pretrain_path_coco_img))

def pretrain_vae_txt(enc,dec):
    print(enc)
    print(dec)
    optimizer = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=settings.LR_IMG_pretrain_ae)
    # index_list = [i for i in range(len(F_I))]
    for epoch in range(400):
        total_loss = 0.
        for batch_idx, (_, F_T, _, _) in enumerate(kk.train_loader):
            F_T = Variable(F_T.float().to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
            # F_I = torch.Tensor(F_I).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            mu, log = enc(F_T)
            z = reparameterize(mu, log)
            x_bar = dec(z)
            loss = F.mse_loss(x_bar, F_T)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(batch_idx, ' ', labels, ' ', idx)
        print("txt-->epoch {} loss={:.4f}".format(epoch, total_loss / (batch_idx+1)))
        state = {'enc':enc.state_dict(), 'dec':dec.state_dict()}
        torch.save(state, settings.pretrain_path_coco_txt)
    print("txt-->model saved to {}.".format(settings.pretrain_path_coco_txt))
