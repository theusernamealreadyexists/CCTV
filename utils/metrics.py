import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import torch

'''
假设原始相关向量为：[1,1,0,0,1,0]（1表示与它相关，0表示与它不相关），那么一共三个相关的。
按照预测的哈希进行汉明码排序后，为[0,0,1,1,1,0]。则MAP=(1/3+2/4+3/5)/3=0.4778。
再例，原始相关向量为：[1,1,0,0,1,0]。按照预测的哈希进行汉明码排序后，为[1,0,1,1,0]。
则MAP=(1/1+2/3+3/4)/3=0.9167。
————————————————
版权声明：本文为CSDN博主「低调流年的微凉」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40680309/article/details/115429280
'''


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    dist_h = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return dist_h.type(torch.int8)


def calc_map(qB, rB, query_L, retrieval_L):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    for iter in range(num_query):
        # 每个查询标签乘以检索标签的转置，只要有相同标签，该位置就是1
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)  # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.int8)
        tsum = int(torch.sum(gnd))  # 真实相关的数据个数
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter], rB)
        _, ind = torch.sort(hamm)  # 按照汉明距离排序，索引
        ind.squeeze_()
        gnd = gnd[ind]
        count = torch.arange(1, tsum + 1).type(torch.float32)
        tindex = torch.nonzero(gnd).squeeze().type(torch.float32) + 1.0
        map = map + torch.mean(count / (tindex))  # e.g. MAP=(1/3+2/4+3/5)/3=0.4778 分母是suoyin，分子是等差数列
    map = map / num_query  # mei ge yang ben
    return map


# 有待改进的地方：因为测试集和训练集的都是不打乱的，所以理论上来说可以不使用index来获取索引
def compress(train_loader, test_loader, model_I, model_T, train_dataset, test_dataset, code_len, batch_size):
    # 输出要全部是cpu上的tensor
    re_BI = torch.empty(len(train_dataset), code_len, dtype=torch.int8)
    re_BT = torch.empty(len(train_dataset), code_len, dtype=torch.int8)
    re_L = torch.empty(train_dataset.labels.shape[0], train_dataset.labels.shape[1], dtype=torch.int8)
    num_data = len(train_dataset)
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    i = 0
    for _, (data_I, data_T, target, _) in enumerate(train_loader):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        data_I = data_I.cuda()
        mu_img, _, _, _, _ = model_I(data_I)
        code_I = torch.tanh(model_I.kappa * mu_img)
        code_I = torch.sign(code_I)
        re_BI[ind, :] = code_I.cpu().data.type(torch.int8)  # GPU->cpu

        data_T = data_T.float().cuda()
        mu_txt, _, _, _, _ = model_T(data_T)
        code_T = torch.tanh(model_T.kappa * mu_txt)
        code_T = torch.sign(code_T)
        re_BT[ind, :] = code_T.cpu().data.type(torch.int8)  # GPU->cpu->numpy()

        re_L[ind, :] = target.type(torch.int8)

        i = i + 1

    i = 0
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    num_data = len(test_dataset)
    qu_BI = torch.empty(len(test_dataset), code_len, dtype=torch.int8)
    qu_BT = torch.empty(len(test_dataset), code_len, dtype=torch.int8)
    qu_L = torch.empty(test_dataset.labels.shape[0], test_dataset.labels.shape[1], dtype=torch.int8)
    for _, (data_I, data_T, target, _) in enumerate(test_loader):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        var_data_I = data_I.cuda()
        mu_img, _, _, _, _ = model_I(var_data_I)
        code_I = torch.tanh(model_I.kappa * mu_img)
        code_I = torch.sign(code_I)
        qu_BI[ind, :] = code_I.cpu().data.type(torch.int8)

        var_data_T = data_T.float().cuda()
        mu_txt, _, _, _, _ = model_T(var_data_T)
        code_T = torch.tanh(model_T.kappa * mu_txt)
        code_T = torch.sign(code_T)
        qu_BT[ind, :] = code_T.cpu().data.type(torch.int8)

        qu_L[ind, :] = target.type(torch.int8)

        i = i + 1

    # sio.savemat('./hashcodes_%s' % settings.DATASET + '/cctv_%d.mat' % settings.CODE_LEN,
    #             {'re_BI': re_BI, 're_BT': re_BT, 're_L': re_L, 'qu_BI': qu_BI, 'qu_BT': qu_BT, 'qu_L': qu_L})
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def compress_wiki(train_loader, test_loader, model_I, model_T, train_dataset, test_dataset, code_len, batch_size):
    # 全部是int8的cpu tensor
    re_BI = torch.empty(len(train_dataset), code_len, dtype=torch.int8)
    re_BT = torch.empty(len(train_dataset), code_len, dtype=torch.int8)
    re_L = torch.empty(train_dataset.labels.shape[0], 1, dtype=torch.int8)
    num_data = len(train_dataset)
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    i = 0
    for _, (data_I, data_T, target, _) in enumerate(train_loader):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        data_I = data_I.cuda()
        mu_img, _, _, _, _ = model_I(data_I)
        code_I = torch.tanh(model_I.alpha * mu_img)
        code_I = torch.sign(code_I)
        re_BI[ind, :] = code_I.cpu().data.type(torch.int8)  # GPU->cpu

        data_T = data_T.float().cuda()
        mu_txt, _, _, _, _ = model_T(data_T)
        code_T = torch.tanh(model_T.alpha * mu_txt)
        code_T = torch.sign(code_T)
        re_BT[ind, :] = code_T.cpu().data.type(torch.int8)  # GPU->cpu->numpy()
        re_L[ind, :] = target.unsqueeze(1).type(torch.int8)

        i = i + 1

    i = 0
    qu_BI = torch.empty(len(test_dataset), code_len, dtype=torch.int8)
    qu_BT = torch.empty(len(test_dataset), code_len, dtype=torch.int8)
    qu_L = torch.empty(test_dataset.labels.shape[0], 1, dtype=torch.int8)
    for _, (data_I, data_T, target, _) in enumerate(test_loader):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        var_data_I = data_I.cuda()
        mu_img, _, _, _, _ = model_I(var_data_I)
        code_I = torch.tanh(model_I.alpha * mu_img)
        code_I = torch.sign(code_I)
        qu_BI[ind, :] = code_I.cpu().data.type(torch.int8)

        var_data_T = data_T.float().cuda()
        mu_txt, _, _, _, _ = model_T(var_data_T)
        code_T = torch.tanh(model_T.alpha * mu_txt)
        code_T = torch.sign(code_T)
        qu_BT[ind, :] = code_T.cpu().data.type(torch.int8)

        qu_L[ind, :] = target.unsqueeze(1).type(torch.int8)

        i = i + 1
    qu_L = torch.from_numpy(np.eye(10)[qu_L - 1]).squeeze()
    re_L = torch.from_numpy(np.eye(10)[re_L - 1]).squeeze()

    # sio.savemat('./hashcodes_%s' % settings.DATASET + '/cctv_%d.mat' % settings.CODE_LEN,
    #             {'re_BI': re_BI, 're_BT': re_BT, 're_L': re_L, 'qu_BI': qu_BI, 'qu_BT': qu_BT, 'qu_L': qu_L})
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def pr(qB, rB, qL, rL, topK=-1):
    n_query = qB.shape[0]
    if topK == -1 or topK > rB.shape[0]:  # top-K 之 K 的上限
        topK = rB.shape[0]

    Gnd = (qL.mm(rL.transpose(0, 1)) > 0).type(torch.int8)
    _, Rank = torch.sort(calc_hammingDist(qB, rB))

    pre_list, recall_list = [], []
    for k in range(1, topK + 1):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all

        pre_list.append(torch.mean(p))
        recall_list.append(torch.mean(r))
    pre_list = [round(i.item(), 4) for i in pre_list]
    recall_list = [round(i.item(), 4) for i in recall_list]
    return pre_list, recall_list

def hamming_distance(X, Y):
    '''
    返回两个矩阵以行为pair的汉明距离
    :param X: (n, hash_len)
    :param Y: (m, hash_len)
    :return: (n, m)
    '''

    res = np.bitwise_xor(np.expand_dims(X, 1), np.expand_dims(Y, 0))
    res = np.sum(res, axis=2)
    return res

def precision_recall(q, r, similarity_matrix):
    q = q.numpy()
    r = r.numpy()
    pre_list = []
    recall_list = []
    query = q.copy()  # (2000,16)
    retrieval = r.copy()  # (18015,16)

    query[query >= 0] = 1
    query[query != 1] = 0
    retrieval[retrieval >= 0] = 1
    retrieval[retrieval != 1] = 0  # 将-1变为0

    query = query.astype(dtype=np.int8)
    retrieval = retrieval.astype(dtype=np.int8)

    distance = hamming_distance(query, retrieval)  # (2000,18015)
    distance_max = np.max(distance)  # 15
    distance_min = np.min(distance)  # 4

    for radius in range(int(distance_min), int(distance_max)):
        temp_distance = distance.copy()

        temp_distance[distance <= radius] = 1
        temp_distance[temp_distance > radius] = 0

        tp = np.sum(similarity_matrix * temp_distance)  # 7790,67581,723031
        precision = 0
        recall = 0
        if tp != 0:
            x = np.sum(temp_distance)  # 109
            y = np.sum(similarity_matrix) # 20848629
            precision = tp / x
            recall = tp / y
        pre_list.append(precision)
        recall_list.append(recall)

    pre_list = [round(i, 4) for i in pre_list]
    recall_list = [round(i, 4) for i in recall_list]

    return pre_list, recall_list

def similarity_matrix(train_label_set, test_label_set):
    query_label = np.expand_dims(test_label_set.astype(dtype=np.int8), 1)  # (2000, 1, 80)
    retrieval_label = np.expand_dims(train_label_set.astype(dtype=np.int8), 0)  # (1, 50000, 80)
    similarity_matrix = np.bitwise_and(query_label, retrieval_label)  # (2000,50000,80)
    similarity_matrix = np.sum(similarity_matrix, axis=2)  # (2000,50000)
    similarity_matrix[similarity_matrix >= 1] = 1  # (2000,50000)
    return similarity_matrix

def similarity_matrix_wiki(train_label_set, test_label_set):
    query_label = np.expand_dims(np.eye(10)[test_label_set.astype(dtype=np.int8)-1], 1).astype(int)  # (2000, 1, 80)
    retrieval_label = np.expand_dims(np.eye(10)[train_label_set.astype(dtype=np.int8)-1], 0).astype(int)  # (1, 50000, 80)
    similarity_matrix = np.bitwise_and(query_label, retrieval_label)  # (2000,50000,80)
    similarity_matrix = np.sum(similarity_matrix, axis=2)  # (2000,50000)
    similarity_matrix[similarity_matrix >= 1] = 1  # (2000,50000)
    return similarity_matrix



if __name__ == '__main__':
    qB = np.array([[1, -1, 1, 1],
                   [-1, -1, -1, 1],
                   [1, 1, -1, 1],
                   [1, 1, 1, -1]])
    rB = np.array([[1, -1, 1, -1],
                   [-1, -1, 1, -1],
                   [-1, -1, 1, -1],
                   [1, 1, -1, -1],
                   [-1, 1, -1, -1],
                   [1, 1, -1, 1]])
    query_L = np.array([[0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [1, 0, 0, 1],
                        [0, 1, 0, 1]])
    retrieval_L = np.array([[1, 0, 0, 1],
                            [1, 1, 0, 0],
                            [0, 1, 1, 0],
                            [0, 0, 1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 1, 0]])

    map = calc_map(qB, rB, query_L, retrieval_L)
    print(map)