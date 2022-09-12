import dgl
import numpy as np
import os
import time
import random
import glob
import argparse, json
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from nets.STOIC.load_net import gnn_model  # import GNNs
from data.data import LoadData  # import dataset


class Similarity:

    def __init__(self, tresh=None):
        super().__init__()
        self.tresh = tresh

    def forward(self, g):
        positive_rate_0, positive_rate_1, positive_rate = self.apply_edge_processing(g)
        return positive_rate_0, positive_rate_1, positive_rate

    def apply_edge_processing(self, g):
        self.encode(g)
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
        assert len(diff_from_1_idx) == len(diff_from_0_idx), "Edge problem"
        assert len(diff_idx) == len(diff_from_1_idx) + len(diff_from_0_idx), "Edge problem"
        assert len(same_idx) == len(same_0_idx) + len(same_1_idx), "Edge problem"
        assert len(same_idx) + len(diff_idx) == g.number_of_edges(), "Edge problem"
        positive_rate_1 = len(same_1_idx)/(len(same_1_idx)+len(diff_from_1_idx))
        positive_rate_0 = len(same_0_idx)/(len(same_0_idx)+len(diff_from_0_idx))
        positive_rate = len(same_idx)/(len(diff_idx)+len(same_idx))
        return positive_rate_0, positive_rate_1, positive_rate

    def calc_similarity(self, edges):
        sex = torch.reshape(torch.cat((edges.dst['Sex'], edges.src['Sex'])), (2, -1)).t().float()
        age = torch.reshape(torch.cat((edges.dst['Age'], edges.src['Age'])), (2, -1)).t().float()
        feat = torch.reshape(torch.cat((edges.dst['feat'], edges.src['feat'])), (2, -1)).t().float()
        sim = torch.logical_xor(edges.dst['Sex'], edges.src['Sex']).int()
        mask = sim == 1 # >= self.tresh
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
    os.chdir(main_path)
    dataset = LoadData(DATASET_NAME)
    os.chdir(root_path)
    split_number = 0
    g = dataset.val[split_number][0][0]


    tresh = 0
    similarity = Similarity(tresh = tresh)

    positive_rate_0, positive_rate_1, positive_rate = similarity.forward(g)

    print('positive_rate_0 : ', positive_rate_0)
    print('positive_rate_1 : ', positive_rate_1)
    print('positive_rate : ', positive_rate)



