import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SIMNet import SIMNet
import dgl

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer import GraphSageLayer as GraphSageLayer
from layers.mlp_readout_layer import MLPReadout
from layers.preprocessing import Preprocessing

class GraphSageNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """

    def __init__(self, net_params, tresh=None):
        super().__init__()

        in_dim_node = 1024  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = 2
        if net_params['edge_feat']:
            self.layer_type = 'edge'
        else:
            self.layer_type = 'node'
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.readout = net_params['readout']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.tresh = tresh

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)# node feat is an integer
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                                    dropout, aggregator_type, batch_norm, residual) for _ in
                                     range(n_layers - 1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        if self.tresh['exp'] == 'SIMNet':
            global current_split
            split_number = current_split
            #self.preprocessor = SIMNet()
            out_dir = 'C:/Users/maxim/PycharmProjects/PrognosisGraph/out/'
            log_dir = out_dir + 'logs/'
            version = "SIMNet_1"
            last_epoch = [99, 0, 0, 1]
            root_ckpt_dir = out_dir + 'checkpoints/' + version
            check_point_dir = root_ckpt_dir + '/RUN_' + str(split_number) + '/' + '{}.pkl'.format(
                "epoch_" + str(last_epoch[split_number]))
            check_pt = torch.load(check_point_dir)
            self.preprocessor = SIMNet()
            self.preprocessor.load_state_dict(check_pt)
        elif tresh['exp'] != 'fully_connected':
            self.preprocessor = Preprocessing(tresh=tresh, split_number=net_params["split_num"])

    def forward(self, g):
        # input embedding
        if self.tresh['exp'] == 'SIMNet':
            """g = self.preprocessor(g)"""
            g.edata['similarity'] = self.preprocessor(g)
            mask = torch.arange(g.number_of_edges())[torch.logical_not(torch.squeeze(g.edata['similarity'].argmax(dim=1))).bool()]
            # transform = RemoveSelfLoop()
            g = dgl.remove_edges(g, mask)
            #g = dgl.sampling.select_topk(g, 100, 'similarity')
            #e = g.edata['similarity'][:, 1].float()

            #g = dgl.sampling.select_topk(g, 40, 'similarity')
            #e = g.edata['similarity'].float()
        elif self.tresh['exp'] != 'fully_connected':
            g = self.preprocessor(g)
            e = g.edata['similarity'].float()
            #g = dgl.sampling.select_topk(g, 50, 'feat')
        else:
            g = dgl.sampling.sample_neighbors(g, list(range(0, g.ndata['feat'].size()[0])), 40)
            #e = g.edata['feat'].float()

        #torch.set_default_dtype(torch.float64)
        #g = dgl.sampling.sample_neighbors(g, list(range(0, g.ndata['feat'].size()[0])), 30)
        #g = dgl.khop_graph(g, 1)
        #g = dgl.sampling.select_topk(g, self.tresh, 'feat')
        h = g.ndata['feat']
        h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)

        # e = self.embedding_e(np.reshape(e, (-1, 1)).float())
        #e = self.in_feat_dropout(e)

        # graphsage
        #g.ndata['h'] = h
        #g.edata['e'] = e
        if self.layer_type == 'edge':

            for conv in self.layers:
                h = conv(g, h, e)

            # output
            h_out = self.MLP_layer(h)

            return h_out
        else:
            for conv in self.layers:
                #torch.set_default_dtype(torch.float64)
                #g = dgl.sampling.select_topk(g, 7, 'e')
                h = conv(g, h)
            # output
            h_out = self.MLP_layer(h)

            return h_out



    def loss(self, pred, label):
        """
        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        """
        # weighted cross-entropy for unbalanced classes
        V = label.size(0)
        label_count = torch.bincount(label.long().argmax(dim=1))
        label_count = label_count[torch.nonzero(label_count, as_tuple=False)].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred.float(), label.long().argmax(dim=1))

        return loss


