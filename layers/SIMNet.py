import math
import pandas as pd
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import warnings
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    feat = input1.size()[1]
    return torch.sum((input1 - input2)**2, dim=1) / feat
class SIMNet(nn.Module):
    def __init__(self, net_params, tresh=None):
        super().__init__()
        self.n_classes = 2
        self.device = 'cpu'#net_params['device']
        self._hidden = 1024#net_params['hidden_dim']
        self.out = 128
        #self.tresh = net_params["threshold_value"]
        self.shared = nn.Linear(1024, self._hidden, bias=True)
        #self.shared3 = nn.Linear(2, 2, bias=True)
        #self.batch_norm =net_params['batch_norm']
        self.dropout_A = nn.Dropout(p=0.2)
        self.dropout_B = nn.Dropout(p=0.25)
        """if self.batch_norm:
            self.batch_norm_sim = nn.BatchNorm1d(self._hidden)"""
        #self.shared1 = nn.Linear(self._hidden, self._hidden//2, bias=True)
        #self.shared2 = nn.Linear(self._hidden//2, self.out, bias=True)
        """self.bilinear = nn.Bilinear(self._hidden, self._hidden, self.n_classes, bias=True)

        self.fusion = nn.Linear(self._hidden, self.n_classes, bias=True)"""
        #self.fusion = nn.Linear(self._hidden, self.n_classes, bias=True)

    def forward(self, g):
        g = self.encode(g)
        g.apply_edges(func=self.calc_dist)
        #sim = g.edata['similarity']#F.softmax(g.edata['similarity'], dim=1)
        #tresh = (sim >=self.tresh).long()
        #sim -= sim.min(0, keepdim=True)[0]
        #sim /= sim.max(0, keepdim=True)[0]
        """data = {"similarity": sim.detach().numpy(),
                "label": g.edata['dist'].detach().numpy()}
        df = pd.DataFrame(data)
        hue = "label"
        #for hue in hue_vector:
        #plt.figure()
        #sns.kdeplot(data=df, x='similarity', hue=hue, multiple="fill")
        #plt.hlines(init_rate, df['similarity'].min(), df['similarity'].max())
        #plt.xlim([df['similarity'].min(), df['similarity'].max()])
        #plt.savefig(exp_path + 'kde_' + hue)
        #plt.figure()
        sns.displot(data=df, x='similarity', hue=hue)
        #plt.ylim([0, np.median(np.histogram(df['similarity'])[0]) * 2])
        #plt.savefig(exp_path + 'hist_' + hue)
        plt.show()"""
        #y= F.one_hot(tresh, num_classes = 2)#y = torch.cat((sim, 1-sim), dim=0).view(2, -1).t()#F.one_hot(tresh, num_classes = 2)
        return g.edata['similarity']#F.softmax(g.edata['proba'], dim=1)

    def calc_dist(self, edges):
        # sim = torch.ones((len(edges), 2))
        input1 = edges.src['feat'].float()#torch.cat((edges.src['feat'], edges.src['Age'].unsqueeze(dim=1), edges.src['Sex'].unsqueeze(dim=1)), dim =1).float()
        input2 = edges.dst['feat'].float()#torch.cat((edges.dst['feat'], edges.dst['Age'].unsqueeze(dim=1), edges.dst['Sex'].unsqueeze(dim=1)), dim =1).float()
        """input1 = F.normalize(input1, p=2, dim=1)
        input2 = F.normalize(input2, p=2, dim=1)"""
        input1 = self.dropout_A(input1)
        input2 = self.dropout_B(input2)
        input1 = self.shared(input1)#self.shared2(torch.tanh(self.shared1(torch.tanh(self.shared(torch.tanh(input1))))))
        input2 = self.shared(input2)#self.shared2(torch.tanh(self.shared1(torch.tanh(self.shared(torch.tanh(input2))))))
        self.embeding1 = torch.sigmoid(input1)#torch.tanh(self.shared2(torch.tanh(self.shared1(torch.tanh(self.shared(input1))))))
        self.embeding2 = torch.sigmoid(input2)#torch.tanh(self.shared2(torch.tanh(self.shared1(torch.tanh(self.shared(input2))))))
        self.embeding1 = torch.cat((self.embeding1, edges.src['age'].unsqueeze(dim=1), edges.src['sex'].unsqueeze(dim=1)), dim =1).float()
        self.embeding2 = torch.cat(
            (self.embeding2, edges.dst['age'].unsqueeze(dim=1), edges.dst['sex'].unsqueeze(dim=1)), dim=1).float()
        #self.intermediate= torch.subtract(self.embeding1, self.embeding2)

        sim = F.cosine_similarity(self.embeding1, self.embeding2, dim=1)#torch.cat((self.shared(input1), self.shared(input2)), dim=1)
        #sim = symmetric_mse_loss(self.embeding1,self.embeding2)  # self.fusion((self.intermediate**2)/self.intermediate.size(1))
        proba = None#torch.relu(self.fusion(abs(self.embeding1-self.embeding2)))
        """if self.batch_norm:
            sim -= sim.min(0, keepdim=True)[0]
            sim /= sim.max(0, keepdim=True)[0]"""
        #sim = torch.relu(self.bilinear(self.embeding1, self.embeding2))
        #torch.cat((tmp, 1-P), dim=0).view(2, -1).t()
        # , 'proba' : proba
        return {'similarity': sim}

    def encode(self, g):
        transformer = MinMaxScaler()
        g.ndata['age'] = torch.tanh(
            torch.tensor(np.reshape(transformer.fit_transform(np.reshape(g.ndata['Age'], (-1, 1))), (-1))))
        g.ndata['sex'] = g.ndata['Sex']

        """res = self.shared3(torch.cat((g.ndata['Sex'].unsqueeze(dim=1),g.ndata['age'].unsqueeze(dim=1)),dim=1).float())
        g.ndata['sex'] =res[:,0]
        g.ndata['age'] = res[:,1]"""
        return g


    def loss(self, pred, label):
        """V = label.size(0)
        label_count = torch.bincount(label.long())
        label_count = label_count[torch.nonzero(label_count, as_tuple=False)].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        label_loss = criterion(pred.float(), label.long())"""
        """from sklearn.metrics import log_loss
        norm2 = torch.sum(self.intermediate**2, dim = 1)
        norm1 = abs(torch.sum(self.intermediate, dim = 1))
        
        P = (1+(math.e**-m))/(1+torch.exp(norm1-m))
        loss = criterion(torch.cat((P, 1-P), dim=0).view(2, -1).t(), label.long()))"""
        """label = (label*2)-1
        criterion = nn.CosineEmbeddingLoss(margin=-0.5)
        loss = criterion(self.embeding1, self.embeding2, label)"""

        """
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        label_loss = criterion(pred.float(), label.long())
        """
        V = label.size(0)
        label_count = torch.bincount(label.long())
        label_count = label_count[torch.nonzero(label_count, as_tuple=False)].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        m = 1
        """dist_loss = torch.sum((label/cluster_sizes[1])* symmetric_mse_loss(self.embeding1, self.embeding2)+(1-label)*(1/cluster_sizes[0])*torch.maximum(torch.zeros_like(label), m-symmetric_mse_loss(self.embeding1, self.embeding2)))
        loss = dist_loss #+ label_loss"""
        criterion = nn.CosineEmbeddingLoss(margin=0)
        label = 2*label-1
        loss = criterion(self.embeding1, self.embeding2, label)
        #loss = torch.sum((label/cluster_sizes[1])* (1-torch.cosine_similarity(self.embeding1, self.embeding2))+(1-label)*(1/cluster_sizes[0])*torch.max(torch.zeros_like(label), torch.cosine_similarity(self.embeding1, self.embeding2)))
        return loss

    """def __repr__(self):
        return '{}(in_channels={}, out_channels={}, aggregator={}, residual={})'.format(self.__class__.__name__,
                                                                                        self.in_channels,
                                                                                        self.out_channels,
                                                                                        self.aggregator_type,
                                                                                        self.residual)"""





