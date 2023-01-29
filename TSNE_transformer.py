"""
    IMPORTING LIBS
"""

import numpy as np
import os
import time
import argparse, json
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import wandb
import pandas as pd
import seaborn as sns

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


def view_model_param(MODEL_NAME, net_params, tresh):
    model = gnn_model(MODEL_NAME, net_params, tresh)
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


def train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs, tresh):
    from train.train_STOIC import evaluate_network_sparse as evaluate_network
    from nets.SBU.SBU_net import SBU_net
    dataset = LoadData(DATASET_NAME)
    split_number = params['split_num']
    trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[
        split_number]
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=False,
                            collate_fn=dataset.collate)

    split_number = 0
    # self.preprocessor = SIMNet()
    out_dir = 'C:/Users/maxim/PycharmProjects/PrognosisGraph/out/'
    log_dir = out_dir + 'logs/'
    version = "REF_20fold_IS32"
    last_epoch = [6, 5, 6, 5]
    root_ckpt_dir = out_dir + 'checkpoints/' + version
    check_point_dir = root_ckpt_dir + '/RUN_' + str(split_number) + '/' + '{}.pkl'.format(
        "epoch_" + str(last_epoch[split_number]))
    check_pt = torch.load(check_point_dir)
    model = SBU_net(net_params)
    model.load_state_dict(check_pt)
    device = net_params['device']
    epoch = 0
    if device == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    model.to(device)
    epoch_val_loss, feature, val_comb_loss, A_val = evaluate_network(model, device, val_loader,
                                                                                   epoch)
    train_features, val_labels = next(iter(val_loader))
    #A_val = next(iter(val_loader))[0].ndata['feat'].detach().numpy()
    features, labels = A_val, next(iter(val_loader))[1].argmax(dim=1).detach().numpy()

    # We want to get TSNE embedding with 2 dimensions
    n_components = 2

    for i in range(1, 35):
        perp = i
        ##tsne = TSNE(n_components)
        tsne = TSNE(n_components=n_components, init='pca',
                    random_state=0, perplexity=perp, n_iter=5000)
        tsne_result = tsne.fit_transform(features)
        y = np.where(labels == 1, 'Positives', 'Negatives')
        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
        fig, ax = plt.subplots(1)
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=120)
        lim = (tsne_result.min() - 5, tsne_result.max() + 5)
        ax.set(title='tsne with ' + str(perp))
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.legend(loc=0)
        ax.axis('tight')
        plt.show()
def main():
    """
        USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.getcwd() + "/configs/SBU.json",
                        help="Please give a config.json file with training/model/data/param details")
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
    parser.add_argument('--feature_dim', help="Please give a value for feature_dim")
    parser.add_argument('--similarity_dim', help="Please give a value for feature_dim")
    parser.add_argument('--mlp_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--n_transformers', help="Please give a value for n_heads")
    parser.add_argument('--cls_token', help="Please give a value for cls_token")
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
    parser.add_argument('--cross_val', help="If you want to make cross validation")
    parser.add_argument('--split_num', help="To select the fold for training")
    parser.add_argument('--threshold_value', help="To select the threshold to apply on edge")
    parser.add_argument('--similarity', help="To select the edge similarity representation")
    parser.add_argument('--type_of_thresh', help="To select of to apply threshold")

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
    # dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.cross_val is not None:
        params['cross_val'] = bool(args.cross_val)
    if args.similarity is not None:
        params['similarity'] = args.similarity
    if args.type_of_thresh is not None:
        params['type_of_thresh'] = str(args.type_of_thresh)
    if args.threshold_value is not None:
        params['threshold_value'] = float(args.threshold_value)
    if args.split_num is not None:
        params['split_num'] = int(args.split_num)
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
    if args.feature_dim is not None:
        net_params['feature_dim'] = int(args.feature_dim)
    if args.similarity_dim is not None:
        net_params['similarity_dim'] = int(args.similarity_dim)
    if args.mlp_dim is not None:
        net_params['mlp_dim'] = int(args.mlp_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.cls_token is not None:
        net_params['cls_token'] = True if args.cls_token == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.n_transformers is not None:
        net_params['n_transformers'] = int(args.n_transformers)
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

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    def Merge(dict1, dict2):
        res = {**dict1, **dict2}
        return res

    params = Merge(params, net_params)
    """os.environ["WANDB_API_KEY"] = 'f6a1a7a209231118c99d5b6078fc26a2941ce3c3'
    wandb.login()"""

    wandb.init(config=params, allow_val_change=True)#, group = 'cross_val' , job_type="optimize"
    net_params = wandb.config
    params = wandb.config
    tresh=None
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params, tresh)
    train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs, tresh)

main()

