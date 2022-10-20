import dgl
import numpy as np
import os
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from data.data import LoadData  # import dataset
import torch.nn as nn
import pandas as pd

import torch
from torch.nn import functional as F
from torch.autograd import Variable


def softmax_mse(input1, input2):
    assert input1.size() == input2.size()
    input_softmax = F.softmax(input1, dim=1)
    target_softmax = F.softmax(input2, dim=1)
    return torch.mean(F.mse_loss(input_softmax,
                                 target_softmax,
                                 reduction='none'),
                      dim=1)


def softmax_kl(input1, input2):
    assert input1.size() == input2.size()
    input_log_softmax = F.log_softmax(input1, dim=1)
    target_softmax = F.softmax(input2, dim=1)
    return torch.mean(F.kl_div(input_log_softmax,
                               target_softmax,
                               reduction='none'),
                      dim=1)


def proj(input1, input2):
    assert input1.size() == input2.size()
    input1 = F.normalize(input1, dim=1, p=2)
    input2 = F.normalize(input2, dim=1, p=2)
    return torch.mean(2 - 2 * (input1 * input2), dim=1)  ###2 - 2 * (input1 * input2).sum(dim=-1) ###recheck this one


def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    feat = input1.size()[1]
    return torch.sum((input1 - input2) ** 2, dim=1) / feat


def poly_kernel(input1, d=0.5, alpha=1.0, c=2.0):
    K_XX = torch.mm(input1, input1.t()) + c
    return K_XX.pow(d)


def smi(input1, input2):
    K_X = poly_kernel(input1)
    K_Y = poly_kernel(input2)
    n = K_X.size(0)
    phi = K_X * K_Y
    hh = torch.mean(phi, 1)
    Hh = K_X.mm(K_X.t()) * K_Y.mm(K_Y.t()) / n ** 2 + torch.eye(n)
    alphah = torch.matmul(torch.inverse(Hh), hh)
    smi = 0.5 * torch.dot(alphah, hh) - 0.5
    return smi  # , alphah


def sqrtcos (input1, input2):
    assert input1.size() == input2.size()
    num = torch.sum(torch.sqrt(abs(torch.mul(input1, input2))), dim=1)
    den = torch.sum(input1, dim=1)*torch.sum(input2, dim=1)
    return num/den


def ISC(input1, input2):
    assert input1.size() == input2.size()
    num = torch.sum(torch.sqrt(abs(torch.mul(input1, input2))),
                    dim=1)
    den = torch.sqrt(abs(torch.sum(input1, dim=1)))*\
          torch.sqrt(abs(torch.sum(input2, dim=1)))
    return num/den


def cosine(input1, input2):
    assert input1.size() == input2.size()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    out = cos(input1, input2)
    return out


class Similarity:

    def __init__(self, tresh=None):
        super().__init__()
        self.tresh = tresh

    def forward(self, g):
        positive_rate_0, positive_rate_1, positive_rate = self.apply_edge_processing(g)
        return positive_rate_0, positive_rate_1, positive_rate

    def apply_edge_processing(self, g):
        self.encode(g)
        self.experiment = 'sqrtcos'
        g.apply_edges(func=self.calc_similarity)
        mask = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['mask'])]
        g = dgl.remove_edges(g, mask)
        g.apply_edges(func=self.calc_label_dist)
        diff_from_1_idx = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['diff_from_1'])]
        diff_from_0_idx = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['diff_from_0'])]
        same_1_idx = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['same_1'])]
        same_0_idx = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['same_0'])]
        diff_idx = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['diff'])]
        same_idx = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['same'])]
        #assert len(diff_from_1_idx) == len(diff_from_0_idx), "Edge problem"
        assert len(diff_idx) == len(diff_from_1_idx) + len(diff_from_0_idx), "Edge problem"
        assert len(same_idx) == len(same_0_idx) + len(same_1_idx), "Edge problem"
        assert len(same_idx) + len(diff_idx) == g.number_of_edges(), "Edge problem"
        positive_rate_1 = len(same_1_idx)/(len(same_1_idx)+len(diff_from_1_idx))
        positive_rate_0 = len(same_0_idx)/(len(same_0_idx)+len(diff_from_0_idx))
        positive_rate = len(same_idx)/(len(diff_idx)+len(same_idx))
        return positive_rate_0, positive_rate_1, positive_rate

    def calc_similarity(self, edges):
        label_dist = torch.sum(
            torch.reshape(torch.cat((edges.dst['Label'], 2 * edges.src['Label'])), (2, -1)).t().float(), 1)
        diff_from_0 = label_dist == 1
        same_1 = label_dist == 3
        same_0 = label_dist == 0
        diff_from_1 = label_dist == 2
        diff = label_dist % 3 != 0
        same = label_dist % 3 == 0

        sex = torch.reshape(torch.cat((edges.dst['Sex'], edges.src['Sex'])), (2, -1)).t().float()
        age = torch.reshape(torch.cat((edges.dst['Age'], edges.src['Age'])), (2, -1)).t().float()
        # feat = torch.reshape(torch.cat((edges.dst['feat'], edges.src['feat'])), (2, 1024, -1)).t().float()

        age_inter = torch.ones(edges.dst['Age'].size(0)) * 5
        age_diff = abs(edges.dst['Age'] - edges.src['Age'])
        age_sim = (age_diff <= age_inter).int()
        sex_sim = torch.logical_not(torch.logical_xor(edges.dst['Sex'], edges.src['Sex']).int()).int()

        if self.experiment == 'cosine':
            cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
            feat_sim = cosine(edges.dst['feat'], edges.src['feat'])
        elif self.experiment == 'softmax_kl':
            feat_sim = softmax_kl(edges.dst['feat'], edges.src['feat'])

        elif self.experiment == 'symmetric_mse_loss':
            feat_sim = symmetric_mse_loss(edges.dst['feat'], edges.src['feat'])

        elif self.experiment == 'proj':
            feat_sim = proj(edges.dst['feat'], edges.src['feat'])

        elif self.experiment == 'softmax_mse':
            feat_sim = softmax_mse(edges.dst['feat'], edges.src['feat'])

        elif self.experiment == 'sqrtcos':
            feat_sim = sqrtcos(edges.dst['feat'], edges.src['feat'])

        elif self.experiment == 'ISC':
            feat_sim = ISC(edges.dst['feat'], edges.src['feat'])

        sim = (age_sim + sex_sim) * feat_sim

        sim -= sim.min(0, keepdim=True)[0]
        sim /= sim.max(0, keepdim=True)[0]
        data = {"similarity": np.array(sim),
                "same_1": np.array(same_1),
                "same_0": np.array(same_0),
                "same": np.array(same),
                "diff_from_0": np.array(diff_from_0),
                "diff_from_1": np.array(diff_from_1),
                "diff": np.array(diff)}
        df = pd.DataFrame(data)
        hue_vector = ["same"]#, "same_1", "same_0", "diff_from_0", "diff_from_1", "diff"]
        for hue in hue_vector:
            plt.figure()
            """sns.kdeplot(data=df, x='similarity', hue=hue, multiple="fill")
            plt.hlines(init_rate, df['similarity'].min(), df['similarity'].max())
            plt.xlim([df['similarity'].min(), df['similarity'].max()])"""
            #plt.savefig(exp_path + 'kde_' + hue)
            #plt.figure()
            plt.hist(df['similarity'])
            #sns.displot(data=df, x='similarity', hue=hue)
            #plt.ylim([0, np.median(np.histogram(df['similarity'])[0])*2])
            """plt.title('Distplot for similarity : {}'.format(self.experiment))"""
            #plt.ylim([0, 5000])
            plt.savefig('hist_' + self.experiment)
            plt.show()
        """plt.hist(np.array(sim), bins='auto')
        plt.ylim([0, 4000])
        plt.show()"""
        cond10 = sim <= 0.35
        cond11 = sim >= 0.22
        cond1 = torch.logical_and(cond10.int(), cond11.int())
        cond2 = sim >= 0.5
        mask = torch.logical_or(cond1.int(), cond2.int())

        return {'similarity ': sim, 'mask': mask}

    def calc_label_dist(self, edges):# TODO: add node feature

        label_dist = torch.sum(torch.reshape(torch.cat((edges.dst['Label'],  2*edges.src['Label'])), (2, -1)).t().float(), 1)
        diff_from_0 = label_dist == 1
        same_1 = label_dist == 3
        same_0 = label_dist == 0
        diff_from_1 = label_dist == 2
        diff = label_dist % 3 != 0
        same = label_dist % 3 == 0

        return {'label_dist': label_dist,
                'diff_from_1': diff_from_1, 'diff_from_0': diff_from_0,
                'same_1': same_1, 'same_0': same_0,
                'diff': diff, 'same': same}

    @staticmethod
    def encode(g):
        transformer = StandardScaler()
        g.ndata['age'] = torch.tensor(np.reshape(transformer.fit_transform(np.reshape(g.ndata['Age'], (-1, 1))), (-1)))


if __name__ == '__main__':
    DATASET_NAME = 'STOIC'
    root_path = os.getcwd()
    main_path = 'C:/Users/maxim/PycharmProjects/PrognosisGraph'
    local_path = main_path + '/similarity_analysis/'
    os.chdir(main_path)
    dataset = LoadData(DATASET_NAME)
    os.chdir(root_path)
    split_number = 0
    g = dataset.test[split_number][0][0]

    lst_0 = []
    lst_1 = []
    lst = []
    lst_tresh = [0]#np.arange(0.001, 0.6, 0.1)

    exp_name = 'cosine'
    exp_path = local_path + exp_name + '/'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    for tresh in lst_tresh:
        similarity = Similarity(tresh=tresh)
        # natural pos rate for split 0
        init_rate_0, init_rate_1, init_rate = 0.85, 0.15, 0.745
        positive_rate_0, positive_rate_1, positive_rate = similarity.forward(g)
        print('edge threshold :', tresh)
        print('positive_rate_0 : ', positive_rate_0)
        print('diff_rate_0 : ', positive_rate_0-init_rate_0)
        print('positive_rate_1 : ', positive_rate_1)
        print('diff_rate_1 : ', positive_rate_1-init_rate_1)
        print('positive_rate : ', positive_rate)
        print('diff_rate : ', positive_rate-init_rate)

        lst_0.append(positive_rate_0-init_rate_0)
        lst_1.append(positive_rate_1-init_rate_1)
        lst.append(positive_rate-init_rate)

    plt.plot(lst_tresh, lst_0)
    plt.plot(lst_tresh, lst_1)
    plt.plot(lst_tresh, lst)
    plt.legend(['diff_rate_0', 'diff_rate_1', 'diff_rate'])
    plt.xlabel(['threshold'])
    plt.title('positive rate vs threshold apply to cosine similarity')
    plt.savefig(exp_path + 'rate_tresh ')
    plt.show()





