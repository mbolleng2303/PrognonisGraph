import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer, CustomGATLayer, CustomGATLayerEdgeReprFeat, CustomGATLayerIsotropic
from layers.mlp_readout_layer import MLPReadout


class GATNet(nn.Module):
    def __init__(self, net_params, tresh):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = 1#net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = 2#net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']

        self.layer_type = {
            "dgl": GATLayer,
            "edgereprfeat": CustomGATLayerEdgeReprFeat,
            "edgefeat": CustomGATLayer,
            "isotropic": CustomGATLayerIsotropic,
        }.get("dgl", GATLayer)

        self.embedding_h = nn.Linear(in_dim, hidden_dim * num_heads)

        if self.layer_type != GATLayer:
            self.edge_feat = net_params['edge_feat']
            self.embedding_e = nn.Linear(in_dim_edge, hidden_dim * num_heads)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([self.layer_type(hidden_dim * num_heads, hidden_dim, num_heads,
                                                     dropout, self.batch_norm, self.residual) for _ in
                                     range(n_layers - 1)])
        self.layers.append(self.layer_type(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(2 * out_dim, n_classes)

    def forward(self, g):
        h = self.embedding_h(g.ndata['feat'].float())
        h = self.in_feat_dropout(h)

        if self.layer_type == GATLayer:
            for conv in self.layers:
                h = conv(g, h)
        else:
            if not self.edge_feat:
                e = torch.ones_like(g.edata['feat']).to(self.device)
            e = self.embedding_e(e.float())

            for conv in self.layers:
                h, e = conv(g, h, e)

        g.ndata['h'] = h

        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
            return {'e': e}

        g.apply_edges(_edge_feat)

        return g.edata['e']

    def loss(self, pred, label):
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