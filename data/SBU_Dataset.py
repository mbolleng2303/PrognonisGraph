import csv
import os
import pickle
import time

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp
import dgl


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.label_lists = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


def format_dataset(dataset):
    """
        Utility function to recover data,
        INTO-> dgl/pytorch compatible format
    """
    graphs = [data[0] for data in dataset]
    labels = [data[1] for data in dataset]

    for graph in graphs:
        # graph.ndata['feat'] = torch.FloatTensor(graph.ndata['feat'])
        graph.ndata['img'] = graph.ndata['img'].clone().detach().float()  # dgl 4.0
        # adding edge features for Residual Gated ConvNet, if not there
        if 'feat' not in graph.edata.keys():
            edge_feat_dim = graph.ndata['feat'].shape[1]  # dim same as node feature dim
            graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim)

    return DGLFormDataset(graphs, labels)


def get_all_split_idx(dataset):
    """
        - Split total number of graphs into 3 (train, val and test) in 3:1:1
        - Stratified split proportionate to original distribution of data with respect to classes
        - Using sklearn to perform the split and then save the indexes
        - Preparing 5 such combinations of indexes split to be used in Graph NNs
        - As with KFold, each of the 5 fold have unique test set.
    """
    root_idx_dir = './data/SBU/split/'
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}

    # If there are no idx files, do the split and store the files
    if not (os.path.exists(root_idx_dir + '_train.index')):
        print("[!] Splitting the data into train/val/test ...")

        # Using 5-fold cross val as used in RP-GNN paper
        k_splits = 5

        cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
        k_data_splits = []

        # this is a temporary index assignment, to be used below for val splitting
        for i in range(len(dataset.graph_lists)):
            dataset[i][0].a = lambda: None
            setattr(dataset[i][0].a, 'index', i)
        label_lst = [dataset.label_lists[i].argmax(dim=1)[0].numpy() for i in range(len(dataset.label_lists))]
        for indexes in cross_val_fold.split(dataset.graph_lists, label_lst):
            remain_index, test_index = indexes[0], indexes[1]

            remain_set = format_dataset([dataset[index] for index in remain_index])
            lab_lst = [remain_set.label_lists[i].argmax(dim=1)[0].numpy() for i in range(len(remain_set.label_lists))]
            # Gets final 'train' and 'val'
            train, val, _, __ = train_test_split(remain_set,
                                                 range(len(remain_set.graph_lists)),
                                                 test_size=0.25,
                                                 stratify=lab_lst)

            train, val = format_dataset(train), format_dataset(val)
            test = format_dataset([dataset[index] for index in test_index])

            # Extracting only idxs
            idx_train = [item[0].a.index for item in train]
            idx_val = [item[0].a.index for item in val]
            idx_test = [item[0].a.index for item in test]

            f_train_w = csv.writer(open(root_idx_dir + '_train.index', 'a+'))
            f_val_w = csv.writer(open(root_idx_dir + '_val.index', 'a+'))
            f_test_w = csv.writer(open(root_idx_dir + '_test.index', 'a+'))

            f_train_w.writerow(idx_train)
            f_val_w.writerow(idx_val)
            f_test_w.writerow(idx_test)

        print("[!] Splitting done!")

    # reading idx from the files
    for section in ['train', 'val', 'test']:
        with open(root_idx_dir + '_' + section + '.index', 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx


def get_vertices(a):
    edges = []
    feat = []
    for i in range(a.shape[1]):
        for j in range(a.shape[0]):
            if a[i, j] != 0:
                edges.append((i, j))
                feat.append(a[i, j])
    return edges, feat


class Data2Graph(torch.utils.data.Dataset):

    def __init__(self):
        self.nbr_graphs = 5
        self.graph_lists = []
        self.label_lists = []
        self._prepare()

    def _prepare(self):
        t0 = time.time()
        from data.SBU.explore import get_graph_from_data
        label_lists, graph_lists = get_graph_from_data()
        self.label_lists = label_lists
        self.graph_lists = graph_lists
        print("[I] Finished preparation after {:.4f}s".format(time.time() - t0))

    def __len__(self):
        return self.nbr_graphs

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.label_lists[idx]


class SBU_Dataset(torch.utils.data.Dataset):
    def __init__(self, name):
        t0 = time.time()
        self.name = name
        save_dir = os.getcwd() + '/data/SBU/save/'
        if not (os.path.exists(save_dir + 'graph_{}.pkl'.format(name))):
            if name == 'SBU':
                dataset = Data2Graph()
            else:
                print('not yes implemented')

            print("[!] Dataset: ", self.name)

            # this function splits data into train/val/test and returns the indices
            self.all_idx = get_all_split_idx(dataset)

            self.all = dataset
            self.train = [self.format_dataset([dataset[idx] for idx in self.all_idx['train'][split_num]]) for split_num
                          in
                          range(4)]
            self.val = [self.format_dataset([dataset[idx] for idx in self.all_idx['val'][split_num]]) for split_num in
                        range(4)]
            self.test = [self.format_dataset([dataset[idx] for idx in self.all_idx['test'][split_num]]) for split_num in
                         range(4)]
            self._save(save_dir)
        else:
            with open(save_dir + 'graph_{}.pkl'.format(name), "rb") as f:
                f = pickle.load(f)
                self.train = f[0]
                self.val = f[1]
                self.test = f[2]
        print("Time taken: {:.4f}s".format(time.time() - t0))

    def _save(self, save_dir):
        start = time.time()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + 'graph_{}.pkl'.format(self.name), 'wb') as f:
            pickle.dump([self.train, self.val, self.test], f)
        print(' data saved : Time (sec):', time.time() - start)

    def format_dataset(self, dataset):
        """
            Utility function to recover data,
            INTO-> dgl/pytorch compatible format
        """
        graphs = [data[0] for data in dataset]
        labels = [data[1] for data in dataset]

        return DGLFormDataset(graphs, labels)

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels




