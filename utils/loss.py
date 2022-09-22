import torch.nn.functional as F
import torch

import yaml

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def reconstruct_loss(img_from_img, txt_from_txt, F_I, F_T):
    reconstr_loss_img = F.mse_loss(img_from_img, F_I)
    reconstr_loss_txt = F.mse_loss(txt_from_txt, F_T)
    reconstruction_loss = reconstr_loss_img + reconstr_loss_txt
    return reconstruction_loss


def cross_reconstruction_loss(img_from_txt, txt_from_img, F_I, F_T):
    cross_rec_loss = F.mse_loss(img_from_txt, F_I) \
                     + F.mse_loss(txt_from_img, F_T)
    return cross_rec_loss


def cada_loss_func_woLcv(epoch, warmup, logvar_img, logvar_txt, mu_img, mu_txt, img_from_img, txt_from_txt, img_from_txt,
              txt_from_img, F_I, F_T):
    ##############################################
    # KL-Divergence
    ##############################################

    KLD = -(0.5 * torch.sum(1 + logvar_txt - mu_txt.pow(2) - logvar_txt.exp())) \
          - (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

    f2 = 1.0 * (epoch - warmup['beta']['start_epoch']) / (
            1.0 * (warmup['beta']['end_epoch'] - warmup['beta']['start_epoch']))
    f2 = f2 * (1.0 * warmup['beta']['factor'])
    beta = torch.cuda.FloatTensor([min(max(f2, 0), warmup['beta']['factor'])]).squeeze()

    max_val1 = torch.max(logvar_txt)
    max_val2 = torch.max(logvar_img)
    min_val1 = torch.min(logvar_txt)
    min_val2 = torch.min(logvar_img)
    # print('max_val: {}, {}'.format(max_val1, max_val2))

    ##############################################
    # Distribution Alignment
    ##############################################
    distance = torch.sqrt(torch.sum((mu_img - mu_txt) ** 2, dim=1) + \
                          torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_txt.exp())) ** 2,
                                    dim=1))
    distance = distance.sum()
    f3 = 1.0 * (epoch - warmup['distance']['start_epoch']) / (
            1.0 * (warmup['distance']['end_epoch'] - warmup['distance']['start_epoch']))
    f3 = f3 * (1.0 * warmup['distance']['factor'])
    distance_factor = torch.cuda.FloatTensor([min(max(f3, 0), warmup['distance']['factor'])])

    # Reconstruct loss
    rec_loss = reconstruct_loss(img_from_img, txt_from_txt, F_I, F_T)

    # corss_reconstrcution_loss
    # cross_rec_loss = cross_reconstruction_loss(img_from_txt, txt_from_img, F_I, F_T)

    loss_cada = rec_loss + beta * KLD
    if distance_factor.item() > 0.0:
        loss_cada += distance_factor.item() * distance
    return loss_cada, max_val1, max_val2, min_val1, min_val2, KLD

def cada_loss_func(epoch, warmup, logvar_img, logvar_txt, mu_img, mu_txt, img_from_img, txt_from_txt, img_from_txt,
              txt_from_img, F_I, F_T):
    ##############################################
    # KL-Divergence
    ##############################################

    KLD = -(0.5 * torch.sum(1 + logvar_txt - mu_txt.pow(2) - logvar_txt.exp())) \
          - (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

    f2 = 1.0 * (epoch - warmup['beta']['start_epoch']) / (
            1.0 * (warmup['beta']['end_epoch'] - warmup['beta']['start_epoch']))
    f2 = f2 * (1.0 * warmup['beta']['factor'])
    beta = torch.cuda.FloatTensor([min(max(f2, 0), warmup['beta']['factor'])]).squeeze()

    max_val1 = torch.max(logvar_txt)
    max_val2 = torch.max(logvar_img)
    min_val1 = torch.min(logvar_txt)
    min_val2 = torch.min(logvar_img)
    # print('max_val: {}, {}'.format(max_val1, max_val2))

    ##############################################
    # Distribution Alignment
    ##############################################
    distance = torch.sqrt(torch.sum((mu_img - mu_txt) ** 2, dim=1) + \
                          torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_txt.exp())) ** 2,
                                    dim=1))
    distance = distance.sum()
    f3 = 1.0 * (epoch - warmup['distance']['start_epoch']) / (
            1.0 * (warmup['distance']['end_epoch'] - warmup['distance']['start_epoch']))
    f3 = f3 * (1.0 * warmup['distance']['factor'])
    distance_factor = torch.cuda.FloatTensor([min(max(f3, 0), warmup['distance']['factor'])])

    # Reconstruct loss
    rec_loss = reconstruct_loss(img_from_img, txt_from_txt, F_I, F_T)

    # corss_reconstrcution_loss
    cross_rec_loss = cross_reconstruction_loss(img_from_txt, txt_from_img, F_I, F_T)

    loss_cada = rec_loss + cross_rec_loss + beta * KLD
    if distance_factor.item() > 0.0:
        loss_cada += distance_factor.item() * distance
    return loss_cada, max_val1, max_val2, min_val1, min_val2, KLD

def hash_loss_func(F_I, F_T, code_I, code_T, para_list, args):
    F_I = F.normalize(F_I)  # (32,4096)
    S_I = F_I.mm(F_I.t())  # (32,32)
    S_I = S_I * 2 - 1  # (32,32)

    F_T = F.normalize(F_T)  # (32,1386)
    S_T = F_T.mm(F_T.t())  # (32,32)
    S_T = S_T * 2 - 1  # (32,32)

    B_I = F.normalize(code_I)  # (32,64)
    B_T = F.normalize(code_T)  # (32,64)

    BI_BI = B_I.mm(B_I.t())  # (32,32)
    BT_BT = B_T.mm(B_T.t())  # (32,32)
    BI_BT = B_I.mm(B_T.t())  # (32,32)

    S_tilde = para_list['LAMBDA'] * S_I + (1 - para_list['LAMBDA']) * S_T  # (32,32)
    S = (1 - para_list['GAMMA']) * S_tilde + para_list['GAMMA'] * S_tilde.mm(S_tilde) / args.batch_size
    S = S * para_list['BETA']  # (32,32)

    loss1 = F.mse_loss(BI_BI, S)
    loss2 = F.mse_loss(BI_BT, S)
    loss3 = F.mse_loss(BT_BT, S)
    loss_CODE = para_list['LAMBDA1'] * loss1 + 1 * loss2 + para_list['LAMBDA2'] * loss3
    return loss_CODE


def dec_loss_func(q_img, q_txt):
    loss_func = torch.nn.KLDivLoss(size_average=False)
    q_img_log = q_img.log()+1e-8  # (50,24)
    p_img = target_distribution(q_img).detach()
    kl_loss_img = loss_func(q_img_log, p_img)/p_img.shape[0]

    q_txt_log = q_txt.log()+1e-8
    p_txt = target_distribution(q_txt).detach()
    kl_loss_txt = loss_func(q_txt_log, p_txt)/p_txt.shape[0]
    loss_dec = kl_loss_img + kl_loss_txt

    return loss_dec


def select_para(args):
    with open(args.scripts) as f:
        scripts = yaml.load(f, Loader=yaml.FullLoader)
    para_list = scripts[args.dataset]
    return para_list

