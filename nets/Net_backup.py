import dgl
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from dgl.nn.pytorch import EdgePredictor

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout
from Abel.tab_transformer_pytorch import TabTransformer
from Abel.vit import ViT
from train.train_STOIC import compute_roc_auc

class Hyper(torch.nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def dist(self, u, v):
        sqdist = torch.sum((u - v) ** 2, dim=-1)
        squnorm = torch.sum(u ** 2, dim=-1)
        sqvnorm = torch.sum(v ** 2, dim=-1)
        x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + self.epsilon
        z = torch.sqrt(x ** 2 - 1)
        return torch.log(x + z)

    def forward(self, src, dst):

        return self.dist(src, dst)
def focal_loss(bce_loss, targets, gamma, alpha):
    """Binary focal loss, mean.

    Per https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5 with
    improvements for alpha.
    :param bce_loss: Binary Cross Entropy loss, a torch tensor.
    :param targets: a torch tensor containing the ground truth, 0s and 1s.
    :param gamma: focal loss power parameter, a float scalar.
    :param alpha: weight of the class indicated by 1, a float scalar.
    """
    p_t = torch.exp(-bce_loss)
    alpha_tensor = (1 - alpha) + targets * (2 * alpha - 1)  # alpha if target = 1 and 1 - alpha if target = 0
    f_loss = alpha_tensor * (1 - p_t) ** gamma * bce_loss
    return f_loss.mean()
"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    feat = input1.size()[1]
    return torch.sum((input1 - input2) ** 2, dim=1) / feat

class SBU_net(nn.Module):
    def __init__(self, net_params, tresh=None):
        super().__init__()
        unique_cat = tuple(np.squeeze(pd.read_csv('info.csv', index_col=0).values, axis=1))
        img_size = 32
        dim = 32
        info = pd.read_csv('info2.csv', index_col=0)
        self.num_cont = int(info['numerical_len'].values[0])
        self.num_cat = int(info['categorical_len'].values[0])
        n_classes = 2
        feature_dim = (self.num_cat*dim)+self.num_cont#net_params['feature_dim']
        similarity_dim = net_params['similarity_dim']
        mlp_dim = net_params['mlp_dim']
        if net_params['edge_feat']:
            self.layer_type = 'anisotropic'
        else:
            self.layer_type = 'isotropic'
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']
        n_transformer = net_params['n_transformers']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        n_head = net_params['n_heads']
        self.net_params = net_params
        self.n_classes = n_classes
        self.device = net_params['device']
        """self.tab_transformer = TabTransformer(categories = unique_cat,      # tuple containing the number of unique values within each category
                                                num_continuous = self.num_cont,                # number of continuous values
                                                dim = dim,                          # dimension, paper set at 32 this is just a test with 256
                                                dim_out = 2,                        # binary prediction, but could be anything
                                                depth = 6,                          # depth, paper recommended 6
                                                heads = 8,                          # heads, paper recommends 8
                                                patch_size= 8,                     #ike vit paper
                                                image_size= img_size,
                                                attn_dropout = 0.0,                 # post-attention dropout
                                                ff_dropout = 0.0,                   # feed forward dropout
                                                mlp_hidden_mults = (2, 2),          # relative multiples of each hidden dimension of the last mlp to logits
                                                mlp_act = nn.GELU())"""
        #self.norm_img = nn.LayerNorm(feature_dim)
        self.project_sim = nn.Linear(feature_dim, similarity_dim)#, bias=False
        #self.embedding_dist = nn.Linear(1, 1)
        self.embedding_e = nn.Linear(1, feature_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GraphSageLayer(feature_dim, feature_dim, F.relu,
                                                    dropout, aggregator_type, batch_norm, residual) for _ in
                                     range(n_layers - 1)])
        #self.layers.append(GraphSageLayer(feature_dim, feature_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
        self.MLP_layer = MLPReadout(feature_dim, n_classes)
        #self.MLP_layer_int = MLPReadout(feature_dim, n_classes)
        #self.norm_feat = nn.LayerNorm(feature_dim)
        self.norm_sim = nn.LayerNorm(67*67)#nn.Linear(1, 1)
        #self.norm_dist = nn.LayerNorm(400*400)
        """self.norm_img = nn.LayerNorm([img_size, img_size])
        self.batch_norm_img = nn.BatchNorm2d(1)
        self.batch_norm_cont = nn.BatchNorm1d(self.num_cont)"""
        self.to_tsne = []
        #self.pred_e = EdgePredictor('cos', similarity_dim, out_feats=1)
        self.hyper = Hyper()


    """def transformer_forward(self, g):
        x_categ = g.ndata['ehr'][:, self.num_cont:].long()  # category values, from 0 - max number of categories, in the order as passed into the constructor above
        x_cont = self.batch_norm_cont(g.ndata['ehr'][:, 0:self.num_cont].float())  # assume continuous values are already normalized individually
        x_image = self.batch_norm_img(g.ndata['img'].float())  #  g.ndata['img']# assume a gray level (1-channel) image of 256 \times 256 for example this can be the x-ray
        encoding, pred = self.tab_transformer(x_categ, x_cont, x_image)
        self.score_int = F.softmax(pred, dim=1)
        # print("Time taken: {:.4f}s".format(time.time() - t0))

        return encoding.detach(), pred"""

    def apply_edge_processing(self, g):
        g.apply_edges(func=self.calc_dist)
        #mask = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['mask'])].to(self.device)
        # transform = RemoveSelfLoop()
        #g.edata['similarity'][mask] = 0
        self.A = torch.reshape(g.edata['similarity'], (int(np.sqrt(g.edata['similarity'].size(0))), int(np.sqrt(g.edata['similarity'].size(0)))))
        #g = dgl.remove_edges(g, mask)
        return g

    def calc_dist(self, edges):
        cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
        """age_inter = torch.ones(edges.dst['Age'].size(0)) * 5
        age_diff = abs(edges.dst['Age'] - edges.src['Age'])
        age_sim = (age_diff <= age_inter).float()"""
        #torch.unsqueeze(age_sim, dim=1).float()
        self.Pi = edges.dst['h']
        self.Pj = edges.src['h']
        self.sim = torch.unsqueeze(self.hyper(self.Pi, self.Pj).float(), dim=1)#1-symmetric_mse_loss
        self.elab = torch.unsqueeze(torch.logical_xor(edges.dst['label'], edges.src['label']), dim =1).long()
        #sim = torch.relu(cosine(torch.relu(edges.src['h']*self.param_vec), torch.relu(edges.dst['h']), dim =1))
        #mask = self.sim <= 0.5
        return {'similarity': self.sim, 'label': self.elab}#, 'mask': mask}  # , 'mask': mask"""

    def representation_learning(self, encoded_feat, g):
        self.feat_sim = self.project_sim(encoded_feat)
        self.to_tsne.append(self.feat_sim.cpu().detach().numpy())
        g.ndata['h'] = self.feat_sim
        # Graph representation learning
        g = self.apply_edge_processing(g)

        g = dgl.sampling.select_topk(g.to('cpu'), 10, 'similarity', output_device=self.device, ascending=False)
        self.sim_smp = g.edata['similarity']
        self.elab_smp = g.ndata['label']
        e = g.edata['similarity'].expand(-1, encoded_feat.size(1))
        return g, e

    def forward_graph(self, encoded_feat, g, e):
        if self.layer_type == 'isotropic':
            h = encoded_feat
            for conv in self.layers:
                h = conv(g, h)
            # output
            self.h_out = self.MLP_layer(h)
            return self.h_out
        else:
            h = encoded_feat
            for conv in self.layers:
                h = conv(g, h, e)
            # output
            h_out = self.MLP_layer(h)
            return h_out

    def forward(self, g):
        # Embedding multi-modal feature
        self.to_tsne = []
        encoded_feat = g.ndata['feat']#self.transformer_forward(g)
        #DF = pd.DataFrame(encoded_feat.detach().numpy())
        #DF.to_csv("extract_feat.csv")
        extracted_feat = encoded_feat.clone().detach() #torch.tensor(np.array(pd.read_csv("extract_feat.csv"))[:,1:])
        self.to_tsne.append(encoded_feat.cpu().detach().numpy())
        g, e = self.representation_learning(extracted_feat, g)
        self.out = self.forward_graph(extracted_feat, g, e)
        self.to_tsne.append(g.ndata['label'].cpu().detach().numpy())

        return self.out, self.A.cpu(), self.to_tsne  #self.A.cpu()#encoded_feat#None#self.A

    def loss(self, pred, label):
        """#self.to_tsne.append(label.cpu().detach().numpy())
        scores_severity = []
        labels_lst = []
        batch_scores = self.score_int
        batch_labels = label
        for i in range(len(batch_scores)):
            score_value = float(batch_scores[i][1].item())
            lab_value = int(torch.argmax(batch_labels[i]).item())
            if i == 0:
                scores_severity = np.expand_dims(np.array(score_value), axis=0)
                labels_lst = np.expand_dims(np.array(lab_value), axis=0)
            else:
                scores_severity = np.concatenate((scores_severity, np.expand_dims(np.array(score_value), axis=0)),
                                                 axis=0)
                labels_lst = np.concatenate((labels_lst, np.expand_dims(np.array(lab_value), axis=0)), axis=0)

        tr_acc = compute_roc_auc(labels=labels_lst, prediction=scores_severity)"""
        scores_severity = []
        labels_lst = []
        batch_scores = self.out
        batch_labels = label
        for i in range(len(batch_scores)):
            score_value = float(batch_scores[i][1].item())
            lab_value = int(torch.argmax(batch_labels[i]).item())
            if i == 0:
                scores_severity = np.expand_dims(np.array(score_value), axis=0)
                labels_lst = np.expand_dims(np.array(lab_value), axis=0)
            else:
                scores_severity = np.concatenate((scores_severity, np.expand_dims(np.array(score_value), axis=0)),
                                                 axis=0)
                labels_lst = np.concatenate((labels_lst, np.expand_dims(np.array(lab_value), axis=0)), axis=0)

        gnn_acc = compute_roc_auc(labels=labels_lst, prediction=scores_severity)

        tr_acc = gnn_acc

        # weighted cross-entropy for unbalanced classes
        V = label.size(0)
        label_count = torch.bincount(label.long().argmax(dim=1))
        label_count = label_count[torch.nonzero(label_count, as_tuple=False)].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # class losss
        criterion = torch.nn.CrossEntropyLoss(weight=weight.to(self.device))#.to(self.device)weight=torch.tensor([0.14, 0.86]).to(self.device)weight=weight.to(self.device)
        loss_gnn = criterion(self.out.float(), label.float())#torch.tensor(0)#"""
        """criterion1 = torch.nn.CrossEntropyLoss(weight=weight.to(self.device))
        loss_tr = criterion1(self.score_int.float(), label.float())#.long() #.argmax(dim=1))"""
        loss_tr = loss_gnn
        #loss_focal = focal_loss(loss_bce, label, 10, 0.14)
        #loss_tr = loss_bce#loss_focal


        #adjacency losss
        N = self.A.size(0)
        d = torch.sum(self.A, dim=1)
        D = torch.diag(1/torch.sqrt(d))
        L = torch.eye(N).to(self.device) - torch.matmul(D, torch.matmul(self.A, D))
        smooth = nn.MSELoss()
        laplacian = torch.matmul(self.feat_sim.t(),
                     torch.matmul(L, self.feat_sim))
        n = self.feat_sim.size(1)
        loss_smooth = torch.tensor(0)#smooth(laplacian, torch.zeros_like(laplacian))# torch.tensor(0)#

        sparsity = nn.L1Loss()
        loss_sparsity = sparsity(self.A, torch.zeros_like(self.A)) #torch.tensor(0)#

        one = torch.ones(N, 1).to(self.device)
        loss_connectivity = -torch.matmul(one.t(), torch.log(torch.matmul(self.A, one)/N))/N#torch.tensor(0)#
        lambda_1 = 0.8
        lambda_2 = 0.2
        lambda_3 = 0.

        """label = self.elab
        V = label.size(0)
        label_count = torch.bincount(label.long()[:,0])
        label_count = label_count[torch.nonzero(label_count, as_tuple=False)].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        """
        """# class losss
        criterion3 = torch.nn.BCELoss(weight=weight.to(
            self.device))  # .to(self.device)weight=torch.tensor([0.14, 0.86]).to(self.device)weight=weight.to(self.device)
        loss_grl = criterion3(torch.concat((1-self.sim, self.sim), dim=1), torch.concat((1-label, label), dim=1).float())"""

        cosine_sim = nn.CosineEmbeddingLoss(margin=0)
        L2_loss = nn.L1Loss()
        contrast = L2_loss(self.sim, self.elab.float())  #cosine_sim(self.Pi, self.Pj, 2*self.elab-1)
        loss_grl = contrast# lambda_1*loss_sparsity #+ lambda_3*loss_smooth #+ lambda_2*loss_connectivity

        contrast1 = L2_loss(self.sim_smp, self.elab_smp.float())
        loss_connectivity =contrast1
        label = self.elab[:, 0]
        V = label.size(0)
        label_count = torch.bincount(label.long())
        label_count = label_count[torch.nonzero(label_count, as_tuple=False)].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        m = 1
        dist_loss = torch.sum(((1-label)/cluster_sizes[0])* symmetric_mse_loss(self.Pi, self.Pj)+(label)*(1/cluster_sizes[1])*torch.maximum(torch.zeros_like(label), m-symmetric_mse_loss(self.Pi, self.Pj)))
        comb_loss = {'smooth': loss_smooth.cpu().detach().numpy(),
                     'sparsity': loss_sparsity.cpu().detach().numpy(),
                     'connectivity': loss_connectivity.cpu().detach().numpy(),
                     'grl': loss_grl.cpu().detach().numpy(),
                     'transformer': loss_tr.cpu().detach().numpy(),
                     'gnn': loss_gnn.cpu().detach().numpy(),
                     'tr_acc': tr_acc,
                     'gnn_acc': gnn_acc}
        lambda_4 = 1
        return loss_gnn + loss_connectivity + dist_loss, comb_loss#+ loss_gnn + lambda_4*loss_grl

