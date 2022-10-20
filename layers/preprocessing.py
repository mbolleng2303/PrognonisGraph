import os

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import RemoveSelfLoop
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
#from similarity import *
import numpy as np
import dgl
import torch as th
#from dgl.nn.pytorch import EdgePredictor
import nets.STOIC.load_net
os.getcwd()
from layers.GATNet import GATNet
from layers.SIMNet import SIMNet
import dgl
import numpy as np
import os
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
#import seaborn as sns
#from data.data import LoadData  # import dataset
import torch.nn as nn
#import pandas as pd

import torch
from torch.nn import functional as F
#from torch.autograd import Variable


def softmax_mse(input1, input2):
    assert input1.size() == input2.size()
    input_softmax = F.softmax(input1, dim=1)
    target_softmax = F.softmax(input2, dim=1)
    feat = input1.size()[1]
    return torch.mean(F.mse_loss(input_softmax, target_softmax, reduction='none'), dim=1)


def softmax_kl(input1, input2):
    assert input1.size() == input2.size()
    input_log_softmax = F.log_softmax(input1, dim=1)
    target_softmax = F.softmax(input2, dim=1)
    return torch.mean(F.kl_div(input_log_softmax, target_softmax, reduction='none'),dim=1)


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
    input1 = F.normalize(input1, dim=1, p=2)
    input2 = F.normalize(input2, dim=1, p=2)
    num = torch.sum(torch.sqrt(abs(torch.mul(input1, input2))), dim = 1)
    denum = torch.sum(input1, dim=1)*torch.sum(input2, dim=1)
    return num/denum

def ISC(input1, input2):
    assert input1.size() == input2.size()
    """input1 -= input1.min(0, keepdim=True)[0]
    input1 /= input1.max(0, keepdim=True)[0]
    input2 -= input2.min(0, keepdim=True)[0]
    input2 /= input2.max(0, keepdim=True)[0]"""
    num = torch.sum(torch.sqrt(abs(torch.mul(input1, input2))), dim = 1)
    denum = torch.sqrt(abs(torch.sum(input1, dim=1)))*torch.sqrt(abs(torch.sum(input2, dim=1)))
    return num/denum

def custom(input1, input2):
    assert input1.size() == input2.size()
    out = np.zeros((input1.size(1)))
    for i in range(input1.size(1)):
        out[i] = softmax_kl(input1[:, i], input2[:, i])

    return out


class Preprocessing(nn.Module,):

    def __init__(self, tresh=None, similarity=None, split_number=0):
        super().__init__()
        self.tresh = tresh['value']
        self.experiment = tresh['exp']
        self.type = tresh['type']
        if similarity == 'Linear':
            self.method = nn.Linear(4, 1)
        if self.experiment == 'learn_cosine':
            out_dir = 'C:/Users/maxim/PycharmProjects/PrognosisGraph/out/'
            log_dir = out_dir+'logs/'
            version = "learn_cosine_2"
            last_epoch = [60, 56, 52, 51]
            # with open(out_dir +'configs/'+ 'config_'+version + '.txt', 'r') as f:
            net_params = {}
            root_ckpt_dir = out_dir + 'checkpoints/' + version
            check_point_dir = root_ckpt_dir + '/RUN_'+str(split_number)+'/'+'{}.pkl'.format("epoch_" + str(last_epoch[split_number]))
            check_pt = torch.load(check_point_dir)
            self.method = SIMNet(net_params)
            self.method.load_state_dict(check_pt)


        #else :
            #self.method = EdgePredictor('cos', in_feats, out_feats=3)
    def forward(self, g):
        """feature = g.ndata['feat']
        age = g.ndata['age']
        sex = g.ndata['Sex']"""
        g = self.apply_edge_processing(g)
        return g

    def apply_edge_processing(self, g):
        self.encode(g)
        #g.edata['e'] = self.method(g)
        if self.experiment in ['rate_1', 'rate_0', 'rate_group', 'rate_global', 'custom']:
            g.apply_edges(func=self.calc_label_connection)
            diff_from_1_idx = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['diff_from_1'])]
            diff_from_0_idx = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['diff_from_0'])]
            same_1_idx = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['same_1'])]
            same_0_idx = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['same_0'])]
            diff_idx = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['diff'])]
            same_idx = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['same'])]
            assert len(diff_from_1_idx) == len(diff_from_0_idx), "Edge problem"
            assert len(diff_idx) == len(diff_from_1_idx) + len(diff_from_0_idx), "Edge problem"
            assert len(same_idx) == len(same_0_idx) + len(same_1_idx), "Edge problem"
            assert len(same_idx) + len(diff_idx) == g.number_of_edges(), "Edge problem"


            if self.experiment == 'custom':
                "Positive label"
                tmp = self.tresh * (len(same_1_idx) + len(diff_from_1_idx))
                if tmp <= len(same_1_idx):
                    drop_eid = same_1_idx
                    noisy_drop_eid1 = drop_eid[torch.randint(len(drop_eid), (len(drop_eid) - int(tmp),))]
                else:
                    tmp = -(self.tresh - 1) * (len(same_1_idx) + len(diff_from_1_idx))
                    drop_eid = diff_from_1_idx
                    noisy_drop_eid1 = drop_eid[torch.randint(len(drop_eid), (len(drop_eid) - int(tmp),))]

                "Negative label"
                self.tresh = (1-self.tresh)  # enhancement
                tmp = self.tresh * (len(same_0_idx) + len(diff_from_0_idx))
                if tmp <= len(same_0_idx):
                    drop_eid = same_0_idx
                    noisy_drop_eid0 = drop_eid[torch.randint(len(drop_eid), (len(drop_eid) - int(tmp),))]
                else:
                    tmp = -(self.tresh - 1) * (len(same_0_idx) + len(diff_from_0_idx))
                    drop_eid = diff_from_0_idx
                    noisy_drop_eid0 = drop_eid[torch.randint(len(drop_eid), (len(drop_eid) - int(tmp),))]
                noisy_drop_eid = torch.cat((noisy_drop_eid1, noisy_drop_eid0))
                g = dgl.remove_edges(g, noisy_drop_eid)
            else:
                "Positive label"
                tmp = self.tresh*(len(same_1_idx) + len(diff_from_1_idx))
                if tmp <= len(same_1_idx):
                    drop_eid = same_1_idx
                    noisy_drop_eid1 = drop_eid[torch.randint(len(drop_eid), (len(drop_eid)-int(tmp),))]
                else:
                    tmp = -(self.tresh-1) * (len(same_1_idx) + len(diff_from_1_idx))
                    drop_eid = diff_from_1_idx
                    noisy_drop_eid1 = drop_eid[torch.randint(len(drop_eid), (len(drop_eid)-int(tmp),))]

                "Negative label"
                tmp = self.tresh * (len(same_0_idx) + len(diff_from_0_idx))
                if tmp <= len(same_0_idx):
                    drop_eid = same_0_idx
                    noisy_drop_eid0 = drop_eid[torch.randint(len(drop_eid), (len(drop_eid) - int(tmp),))]
                else:
                    tmp = -(self.tresh - 1) * (len(same_0_idx) + len(diff_from_0_idx))
                    drop_eid = diff_from_0_idx
                    noisy_drop_eid0 = drop_eid[torch.randint(len(drop_eid), (len(drop_eid) - int(tmp),))]

                "Global label"
                tmp = self.tresh * (len(same_idx) + len(diff_idx))
                if tmp <= len(same_idx):
                    drop_eid = same_idx
                    noisy_drop_eid2 = drop_eid[torch.randint(len(drop_eid), (len(drop_eid) - int(tmp),))]
                else:
                    tmp = -(self.tresh - 1) * (len(same_idx) + len(diff_idx))
                    drop_eid = diff_idx
                    noisy_drop_eid2 = drop_eid[torch.randint(len(drop_eid), (len(drop_eid) - int(tmp),))]

                if self.experiment == 'rate_1':
                    g = dgl.remove_edges(g, noisy_drop_eid1)
                elif self.experiment == 'rate_0':
                    g = dgl.remove_edges(g, noisy_drop_eid0)
                elif self.experiment == 'rate_group':
                    noisy_drop_eid = torch.cat((noisy_drop_eid1, noisy_drop_eid0))
                    g = dgl.remove_edges(g, noisy_drop_eid)
                elif self.experiment == 'rate_global':
                    g = dgl.remove_edges(g, noisy_drop_eid2)



                else:
                    print("indicate experiment")
                    raise NotImplementedError
        elif self.experiment == 'learn_cosine' :
            g.edata['similarity'] = self.method(g)
            sim = g.edata['similarity']
            sim -= sim.min(0, keepdim=True)[0]
            sim /= sim.max(0, keepdim=True)[0]
            mask = sim <= self.tresh
            mask = torch.arange(g.number_of_edges())[torch.squeeze(mask)]
            g = dgl.remove_edges(g, mask)
        else:
            g.apply_edges(func=self.calc_dist)
            mask = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['mask'])]
            #transform = RemoveSelfLoop()
            g = dgl.remove_edges(g, mask)
            #g = transform(g)
        return g

    def calc_dist(self, edges):

        age_inter = torch.ones(edges.dst['Age'] .size(0)) * 5
        age_diff = abs(edges.dst['Age'] - edges.src['Age'])
        age_sim = (age_diff <= age_inter).int()
        sex_sim = torch.logical_not(torch.logical_xor(edges.dst['Sex'], edges.src['Sex']).int())

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


        '''plt.hist(np.array(sim), bins='auto')
        plt.ylim([0, 4000])
        plt.show()'''
        if self.type not in ['supp_abs_from', 'sup_abs_to']:
            sim -= sim.min(0, keepdim=True)[0]
            sim /= sim.max(0, keepdim=True)[0]
            if self.type == 'supp_from':
                mask = sim >= self.tresh  # diff == 1
            elif self.type == 'supp_to':
                mask = sim <= self.tresh
            else :
                mask = sim != sim
        else:
            sim = abs(sim)
            if self.type == 'supp_abs_from':
                mask = sim >= self.tresh  # diff == 1
            elif self.type == 'sup_abs_to':
                mask = sim <= self.tresh


        return {'similarity': sim, 'mask': mask}


    def calc_label_connection(self, edges):
        dist = torch.sum(torch.reshape(torch.cat((edges.dst['Label'], 2 * edges.src['Label'])), (2, -1)).t().float(), 1)
        diff_from_0 = dist == 1
        same_1 = dist == 3
        same_0 = dist == 0
        diff_from_1 = dist == 2
        diff = dist % 3 != 0
        same = dist % 3 == 0

        return {'dist': dist,
                'diff_from_1': diff_from_1, 'diff_from_0': diff_from_0,
                'same_1': same_1, 'same_0': same_0,
                'diff': diff, 'same': same}

    @staticmethod
    def encode(g):
        transformer = StandardScaler()
        g.ndata['age'] = torch.tensor(np.reshape(transformer.fit_transform(np.reshape(g.ndata['Age'], (-1, 1))), (-1)))
