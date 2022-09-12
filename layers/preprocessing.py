import os

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler, StandardScaler
#from similarity import *
import numpy as np
import dgl
import torch as th
#from dgl.nn.pytorch import EdgePredictor
import nets.STOIC.load_net
os.getcwd()
from layers.GATNet import GATNet



class Preprocessing(nn.Module,):

    def __init__(self, tresh=None, similarity=None, split_number=0):
        super().__init__()
        if similarity == 'Linear':
            self.method = nn.Linear(4, 1)
        if similarity == 'gat':
            out_dir = 'C:/Users/maxim/PycharmProjects/PrognosisGraph/out/'
            log_dir = out_dir+'logs/'
            version = "GAT_PreGraph_GPU0_01h42m25s_on_Sep_09_2022"
            last_epoch = [0, 0, 0, 1]
            # with open(out_dir +'configs/'+ 'config_'+version + '.txt', 'r') as f:
            net_params = {'L': 3, 'hidden_dim': 512, 'out_dim': 256, 'residual': True, 'readout': 'mean', 'in_feat_dropout': 0, 'dropout': 0, 'batch_norm': True, 'sage_aggregator': 'mean', 'self_loop': False, 'gpu_id': 0, 'batch_size': 1, 'in_dim': 1024, 'n_classes': 2, 'total_param': 1351042}
            root_ckpt_dir = out_dir + 'checkpoints/' + version
            check_point_dir = root_ckpt_dir + '/RUN_'+str(split_number)+'/'+'{}.pkl'.format("epoch_" + str(last_epoch[split_number]))
            check_pt = torch.load(check_point_dir)
            self.method = GATNet(net_params)
            self.method.load_state_dict(check_pt)
        self.tresh = tresh['value']
        self.experiment = tresh['exp']

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
        g.apply_edges(func=self.calc_dist)
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

        """
        drop_eid = same_1_idx
            pos_rate = int((1 - self.tresh) * len(drop_eid))
        """
        return g

    def calc_dist(self, edges):# TODO: add node feature
        # vector = torch.reshape(torch.cat((edges.dst['age'], edges.dst['Sex'], edges.src['age'], edges.src['Sex'])), (4, -1)).t().float()
        # dist = self.method(vector)
        #predictor = EdgePredictor('cos', in_feats, out_feats=3)
        #predictor.reset_parameters()
        #predictor(h_src, h_dst).shape
        dist = torch.sum(torch.reshape(torch.cat((edges.dst['Label'],  2*edges.src['Label'])), (2, -1)).t().float(), 1)
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
