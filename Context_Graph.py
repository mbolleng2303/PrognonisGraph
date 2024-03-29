import pandas as pd


def get_vertices(a):
    edges = []
    feat = []
    for i in range(a.shape[1]):
        for j in range(a.shape[0]):
            if a[i, j] != 0:
                edges.append((i, j))
                feat.append(a[i, j])
    return edges, feat


class ContextGraph( ):
    def __init__(self, fold=0):
        self.nbr_graphs = 1
        self.graph = []
        self.fold = fold
        self._prepare()

    def _prepare(self):
        t0 = time.time()
        path = os.getcwd()
        fold = 'fold'+ str(self.fold)
        print("[I] Preparing ContextGraph..")
        df = pd.read_csv(os.getcwd() + '/data/STOIC/training_data_features_and_outputs_full.csv',
                         sep=",", header=None,
                         names=['PatientID', 'Subset', 'Age', 'Sex', 'Covid+', 'Severity', 'Covid+ & severe',
                                'Covid- & severe',
                                'Image_size_x', 'Image_size_y', 'Image_size_z', 'Spacing_x', 'Spacing_y', 'Spacing_z',
                                'cls_Covid_P_VS_N_feat', 'cls_Covid_P_VS_N_out', 'cls_Severe_VS_NSevere_feat_fold0',
                                'cls_Severe_VS_NSevere_out_fold0', 'cls_Severe_VS_NSevere_feat_fold1',
                                'cls_Severe_VS_NSevere_out_fold1', 'cls_Severe_VS_NSevere_feat_fold2',
                                'cls_Severe_VS_NSevere_out_fold2', 'cls_Severe_VS_NSevere_feat_fold3',
                                'cls_Severe_VS_NSevere_out_fold3', 'cls_Severe_VS_NSevere_feat_holdout0',
                                'cls_Severe_VS_NSevere_out_holdout0', 'cls_Severe_VS_NSevere_feat_holdout1',
                                'cls_Severe_VS_NSevere_out_holdout1'], skiprows=1, low_memory=False)

        sex_map = {'M': 1,
                   'F': 0}
        df['Sex'] = df['Sex'].map(sex_map)
        index = [k for k, x in enumerate(df['Subset'].values.tolist()) if
                 (x == fold and x != 'holdout0' and x != 'holdout1')]
        data = df.iloc[index]
        age = np.array(data['Age'].values.tolist())
        sex = np.array(data['Sex'].values.tolist())
        feature = np.array(list(map(eval, data['cls_Severe_VS_NSevere_feat_'+fold].values)))
        labels = np.array(data['Severity'].to_numpy(dtype=np.int8))
        g = dgl.DGLGraph()
        g.add_nodes(data.shape[0])
        g.ndata['feat'] = torch.tensor(feature)
        g.ndata['Age'] = torch.tensor(age)
        g.ndata['Sex'] = torch.tensor(sex)
        g.ndata['Label'] = torch.tensor(labels)
        A = np.ones((data.shape[0], data.shape[0]))
        edge = np.array(get_vertices(A)[0])
        edge_feat = np.array(get_vertices(A)[0])
        for src, dst in edge:
            g.add_edges(src.item(), dst.item())
        edge_feat = np.array(edge_feat)
        g.edata['feat'] = torch.tensor(edge_feat).long()
        self.graph.append(g)
        print("[I] Finished preparation after {:.4f}s".format(time.time() - t0))

    def __len__(self):
        return self.nbr_graphs

    def __getitem__(self, idx):
        return self.graph

"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import time
import argparse, json

import torch

from sklearn.metrics import roc_curve, auc

from tensorboardX import SummaryWriter


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from nets.load_net import gnn_model  # import GNNs
from data.data import LoadData  # import dataset

"""
    GPU Setup
"""


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""


def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""

def get_edge (age,sex,features):

    age_term = 1 if abs(age[0]-age[1]) <= 5 else 0
    sex_term =1 if sex[0]==sex[1] else 0
    feat_factor = cosine(features[0],features[1])
    edge = feat_factor*(sex_term+age_term)
    return edge
def cosine(input1, input2):

    return np.dot(input1, input2)/(np.linalg.norm(input1)*np.linalg.norm(input2))
def evalu(model, device, data_loader, epoch, infer = False, infer5=False):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            except :
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            #loss = model.loss(batch_scores, batch_labels)
            #epoch_test_loss += loss.detach().item()

        #epoch_test_loss /= (iter + 1)
        #epoch_test_acc /= (iter + 1)

    return batch_scores[-1],batch_labels[-1]
def compute_roc_auc(labels, prediction, path_to_save=None, stage=None):
    fpr, tpr, thresholds = roc_curve(labels, prediction, pos_label=1)
    roc_auc = auc(fpr, tpr)
    # print("TPR value: {}".format(tpr))
    # print("FPR value: {}".format(tpr))
    # print(thresholds)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ## locate the index of the largest g-mean
    #ix = np.argmax(gmeans)
    # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    return roc_auc
def infer(MODEL_NAME, DATASET_NAME, params, net_params, dirs, last_epoch):
    avg_test_acc = []
    t0 = time.time()
    dataset = LoadData(DATASET_NAME)
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        split_number = 3
        t0_split = time.time()
        log_dir = os.path.join(dirs, "RUN_" + str(split_number))
        writer = SummaryWriter(log_dir=log_dir)

        # setting seeds
        #random.seed(params['seed'])
        #np.random.seed(params['seed'])
        #torch.manual_seed(params['seed'])
        if device.type == 'cuda':
            torch.cuda.manual_seed(params['seed'])
        testset = dataset.test[0]
        # add curent node
        ref_graph = testset.graph_lists[0]

        ref_labels = testset.label_lists[0]
        save_labels = ref_labels

        """model = gnn_model(MODEL_NAME, net_params)
        check_point_dir = os.path.join(log_dir, '{}.pkl'.format("epoch_" + str(last_epoch[split_number])))
        check_pt = torch.load(check_point_dir)
        model.load_state_dict(check_pt)"""

        lst_score= []
        lst_label= []

        a=2
        context_graph = ContextGraph(fold=split_number).graph[0]
        save_graph = ref_graph
        for node_id in save_graph.nodes():
            feat = save_graph.ndata['feat'][node_id]
            current_graph= dgl.add_nodes(context_graph, 1, {'feat': feat, 'age': a, 'sex': s, 'predicted': torch.tensor([1])})
        """for p in range(len(patient[:,0,0])):#len(patient[:,0,0])
            res = [0, 0]

            res[int(patient[p, 0, 1])] = 1
            ref_labels = np.array(ref_labels)
            ref_labels = torch.tensor(np.reshape(np.append(ref_labels,res),[-1,2]))
            for i in range(len(age)):
                link[i]= get_edge([age[i],patient[p,0,2]],[sex[i],patient[p,0,3]],[feature[i,:],patient[p,:,0]])#cosine([age[i],patient[p,0,2]],[sex[i],patient[p,0,3]])#
            idx = np.where(link >0)
            vert = []
            last_node_idx = ref_graph.num_nodes()-1
            feat =torch.tensor(np.reshape(patient[p,:,0], (1024,1)).T)
            a = torch.tensor([patient[p,0,2]])
            s = torch.tensor([patient[p,0,3]])
            ref_graph= dgl.add_nodes(ref_graph, 1, {'feat': feat,'age': a, 'sex':s ,'predicted': torch.tensor([1])})
            idy = np.reshape([idx[i] for i in range (len(idx))],[-1])
            for id in idy :
                ref_graph = dgl.add_edges(ref_graph, torch.tensor([id]), torch.tensor([last_node_idx]),
                                          {'feat': torch.tensor([link[id]])})
                ref_graph = dgl.add_edges(ref_graph,torch.tensor([last_node_idx]),torch.tensor([id]),
                                          {'feat': torch.tensor([link[id]])})
            ref_graph = dgl.add_edges(ref_graph, torch.tensor([last_node_idx]), torch.tensor([last_node_idx]),
                                      {'feat': torch.tensor([2],dtype=torch.float64)})
            print("Test patients: ", p)




            # import train functions for all other GCNs
            from train.metrics import AUC as accuracy

            testset.graph_lists[0]= ref_graph
            testset.label_lists[0] = ref_labels
            test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last,
                                      collate_fn=dataset.collate)


            score,lab = evalu(model, device, test_loader, epoch=0, infer=False, infer5=False)
            #lst_label= torch.cat((torch.tensor(lst_label),lab),0)
            #lst_score=torch.cat((torch.tensor(lst_score),score),0)
            score_value = float(score[1].item())
            lab_value = int(torch.argmax(lab).item())
            if p == 0:
                scores_severity = np.expand_dims(np.array(score_value), axis=0)
                labels = np.expand_dims(np.array(lab_value), axis=0)
            else:
                scores_severity = np.concatenate((scores_severity, np.expand_dims(np.array(score_value), axis=0)),
                                                 axis=0)
                labels = np.concatenate((labels, np.expand_dims(np.array(lab_value), axis=0)), axis=0)

            #g = dgl.remove_nodes(ref_graph,, ntype='predicted')
            testset.graph_lists[0] = save_graph
            #ref_labels = ref_labels[0:len(ref_labels)-2]
            testset.label_lists[0] = save_labels

        scores_severity=scores_severity[covid_idx]
        labels = labels[covid_idx]
        AUC = compute_roc_auc(labels=labels, prediction=scores_severity)"""
        AUC=0

        print("Test Accuracy [LAST EPOCH]: {:.4f}".format(AUC))

    except KeyboardInterrupt:
        print('-' * 89)
    print('Exiting from training early because of KeyboardInterrupt')

    # Final test accuracy value averaged over 5-fold
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(AUC)) * 100, np.std(AUC) * 100))
    print("\nAll splits Test Accuracies:\n", AUC)


    writer.close()

    """
        Write the results in out/results folder
    """


def main():
    """
        USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = "C:/Users/maxim/PycharmProjects/PrognosisGraph/configs/STOIC_pipeline.json", help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--graph_norm', help="Please give a value for graph_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    #dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated == 'True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.graph_norm is not None:
        net_params['graph_norm'] = True if args.graph_norm == 'True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred == 'True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat == 'True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop == 'True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)

    net_params['in_dim'] = 1024  # node_dim (feat is an integer)
    net_params['n_classes'] = 2
    # RingGNN

    # RingGNN, 3WLGNN
    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        if net_params['pos_enc']:
            net_params['in_dim'] = net_params['pos_enc_dim']
        else:
            net_params['in_dim'] = 1


    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)

    version = "GraphSage_X_ray_1024_GPU0_17h10m30s_on_Mar_23_2022"
    last_epoch = [17, 17, 17, 17]
    #version = "GraphSage_X_ray_1024_GPU0_17h15m20s_on_Mar_13_2022"
    #last_epoch = [66, 66, 57, 93]
    root_ckpt_dir = out_dir + 'checkpoints/' + version
    infer(MODEL_NAME, DATASET_NAME, params, net_params, root_ckpt_dir, last_epoch)


main()


