import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer import GraphSageLayer as GraphSageLayer
from layers.mlp_readout_layer import MLPReadout
from layers.preprocessing import Preprocessing

class EdgePredict(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """
    def __init__(self, net_params, tresh):
        super().__init__()

        in_dim_node = 1024  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = 2
        self.layer_type = 'edge'
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.readout = net_params['readout']
        self.n_classes = n_classes
        self.device = net_params['device']

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                                    dropout, aggregator_type, batch_norm, residual) for _ in
                                     range(n_layers - 1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
        self.MLP_layer = MLPReadout(2 * out_dim, n_classes)
        self.preprocessor = Preprocessing()

    def forward(self, g):
        h = g.ndata['feat']
        h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h

        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
            return {'e': e}

        g.apply_edges(_edge_feat)

        return g.edata['e']



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
        label_count = torch.bincount(label.long())
        label_count = label_count[torch.nonzero(label_count, as_tuple=False)].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred.float(), label.long())

        return loss


