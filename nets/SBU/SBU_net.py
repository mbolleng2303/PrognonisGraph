import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""



class SBU_net(nn.Module):
    def __init__(self, net_params, tresh=None):
        super().__init__()

        imaging_dim = 1024
        ehr_dim = 1
        n_classes = 2
        self.ehr_data = ['Sex', 'Age']
        feature_dim = net_params['feature_dim']
        similarity_dim = net_params['similarity_dim']
        mlp_dim = net_params['mlp_dim']
        if net_params['edge_feat']:
            self.layer_type = 'anisotropic'
        else:
            self.layer_type = 'isotropic'
        if net_params['cls_token']:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
            nn.init.normal_(self.cls_token, std=1e-6)
            self.norm_token = nn.LayerNorm(feature_dim)
            n_tokens = len(self.ehr_data)+2
        else:
            n_tokens = len(self.ehr_data)+1
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
        self.embedding_ehr = nn.ModuleList([nn.Linear(ehr_dim, feature_dim) for _ in range(len(self.ehr_data))])
        self.norm_ehr = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(len(self.ehr_data))])
        self.embedding_img = nn.Linear(imaging_dim, feature_dim)
        self.norm_img = nn.LayerNorm(feature_dim)
        self.project_sim = nn.Linear(feature_dim, similarity_dim)#, bias=False
        self.embedding_dist = nn.Linear(1, 1)
        self.embedding_e = nn.Linear(1, feature_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=n_head, dropout=0)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_transformer)
        self.layers = nn.ModuleList([GraphSageLayer(feature_dim, feature_dim, F.relu,
                                                    dropout, aggregator_type, batch_norm, residual) for _ in
                                     range(n_layers - 1)])
        self.layers.append(GraphSageLayer(feature_dim, mlp_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
        self.MLP_layer = MLPReadout(mlp_dim, n_classes)
        self.MLP_layer_int = MLPReadout(feature_dim, n_classes)
        self.norm_feat = nn.LayerNorm(feature_dim)
        self.norm_sim = nn.LayerNorm(similarity_dim)
        self.norm_dist = nn.LayerNorm(400*400)

        self.pos_embed = nn.Parameter(torch.randn(n_tokens, 1, feature_dim) * .02)
        trunc_normal_(self.pos_embed, std=.02)
        self.param_vec = nn.Parameter(torch.zeros(1, feature_dim))
        nn.init.normal_(self.param_vec, std=1e-6)

    def transformer_forward(self, g):
        img = g.ndata['feat']
        ehr = [np.reshape(g.ndata[i], (-1, 1)).float() for i in self.ehr_data]
        img = self.embedding_img(img.float())
        img = self.in_feat_dropout(self.norm_img(F.relu(img)))
        sequence = torch.unsqueeze(img, 0)
        # e = self.embedding_e(np.reshape(g.edata['feat'], (-1, 1)).float())

        for i in range(len(ehr)):
            ehr[i] = torch.unsqueeze(self.embedding_ehr[i](ehr[i].float()), 0)
            ehr[i] = self.in_feat_dropout(self.norm_ehr[i](F.gelu(ehr[i])))
            sequence = torch.cat((sequence, ehr[i]), dim=0)
        # Encoding
        if self.net_params['cls_token']:
            sequence = torch.cat((sequence, self.norm_token(F.gelu(self.cls_token.expand(-1, 400, -1)))), dim=0)
            encoding = self.transformer_encoder(sequence)
            encoding = encoding[-1, :, :]
        else:
            encoding = self.transformer_encoder(sequence)
            encoding = torch.mean(encoding, dim=0)

        encoding = self.norm_feat(encoding)
        self.score_int = self.MLP_layer_int(encoding)

        return encoding

    def apply_edge_processing(self, g):
        g.apply_edges(func=self.calc_dist)
        #mask = torch.arange(g.number_of_edges())[torch.squeeze(g.edata['mask'])]
        # transform = RemoveSelfLoop()
        #g.edata['similarity'][mask] = 0
        # g = dgl.remove_edges(g, mask)
        return g

    def calc_dist(self, edges):
        cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
        age_inter = torch.ones(edges.dst['Age'].size(0)) * 5
        age_diff = abs(edges.dst['Age'] - edges.src['Age'])
        age_sim = (age_diff <= age_inter).float()
        #torch.unsqueeze(age_sim, dim=1).float()
        sim = torch.relu(torch.unsqueeze(cosine(edges.dst['h'], edges.src['h']), dim=1).float())
        #sim = torch.relu(cosine(torch.relu(edges.src['h']*self.param_vec), torch.relu(edges.dst['h']), dim =1))
        #mask = sim <= 0.5
        return {'similarity': sim}  # , 'mask': mask

    def representation_learning(self, encoded_feat, g):
        self.feat_sim = self.project_sim(F.normalize(encoded_feat, dim=1))
        g.ndata['h'] = self.feat_sim
        # Graph representation learning
        g = self.apply_edge_processing(g)
        e = g.edata['similarity'].expand(-1, encoded_feat.size(1))
        self.A = torch.reshape(g.edata['similarity'], (400, 400))
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
            self.h_out = self.MLP_layer(h)
            return self.h_out

    def forward(self, g):
        # Embedding multi-modal feature
        encoded_feat = self.transformer_forward(g)
        #DF = pd.DataFrame(encoded_feat.detach().numpy())
        #DF.to_csv("extract_feat.csv")
        extracted_feat = encoded_feat.detach() #torch.tensor(np.array(pd.read_csv("extract_feat.csv"))[:,1:])
        g, e = self.representation_learning(extracted_feat, g)
        out = self.forward_graph(extracted_feat, g, e)
        return out, self.A

    def loss(self, pred, label):
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

        tr_acc = compute_roc_auc(labels=labels_lst, prediction=scores_severity)

        # weighted cross-entropy for unbalanced classes
        V = label.size(0)
        label_count = torch.bincount(label.long().argmax(dim=1))
        label_count = label_count[torch.nonzero(label_count, as_tuple=False)].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        criterion = torch.nn.CrossEntropyLoss(weight=weight)

        loss_gnn = criterion(self.h_out.float(), label.long().argmax(dim=1))#torch.tensor(0)#
        loss_tr = criterion(self.score_int.float(), label.long().argmax(dim=1))

        N = self.A.size(0)

        d = torch.sum(self.A, dim=1)
        D = torch.diag(1/torch.sqrt(d))
        L = torch.eye(N) - torch.matmul(D, torch.matmul(self.A, D))
        smooth = nn.MSELoss()
        laplacian = torch.matmul(self.feat_sim.t(),
                     torch.matmul(L, self.feat_sim))
        n = self.feat_sim.size(1)
        loss_smooth = smooth(laplacian, torch.zeros_like(laplacian))# torch.tensor(0)#

        sparsity = nn.L1Loss()
        loss_sparsity = sparsity(self.A, torch.zeros_like(self.A)) #torch.tensor(0)#

        one = torch.ones(N, 1)
        loss_connectivity = -torch.matmul(one.t(), torch.log(torch.matmul(self.A, one)/N))/N#torch.tensor(0)#

        lambda_1 = 10
        lambda_2 = 0.2
        lambda_3 = 0.1
        lambda_4 = 1

        loss_grl = lambda_1*loss_sparsity + lambda_3*loss_smooth + lambda_2*loss_connectivity
        comb_loss = {'smooth': loss_smooth,
                     'sparsity': loss_sparsity,
                     'connectivity': loss_connectivity,
                     'grl': loss_grl,
                     'transformer': loss_tr,
                     'gnn': loss_gnn,
                     'tr_acc': tr_acc}
        return loss_tr + loss_gnn + lambda_4*loss_grl, comb_loss

