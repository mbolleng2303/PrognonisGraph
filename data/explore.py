import dgl
from tqdm import tqdm
import pandas as pd
import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp
import dgl
import os
import shutil
from nets.SBU.Transformer import Transformer
import numpy as np
import tempfile
import warnings
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
from monai.data import ITKReader, PILReader, NumpyReader
from monai.transforms import (
    LoadImage, LoadImaged, EnsureChannelFirstd,
    Resized, EnsureTyped, Compose)
from monai.config import print_config
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from matplotlib import pyplot as plt
import matplotlib.cm as cm

def extract_feat(g):
    from train.train_STOIC import train_epoch_sparse as train_epoch, \
        evaluate_network_sparse as evaluate_network
    from nets.SBU.SBU_net import SBU_net
    net_params = {
        "L": 2,
        "n_transformers": 4,
        "n_heads": 4,
        "cls_token": False,
        "feature_dim": 256,
        "similarity_dim": 256,
        "mlp_dim": 256,
        "residual": True,
        "in_feat_dropout": 0,
        "dropout": 0,
        "batch_norm": True,
        "sage_aggregator": "mean",
        "edge_feat": True,
        'device' : 'cpu',
    }
    split_number = 0
    # self.preprocessor = SIMNet()
    out_dir = 'C:/Users/maxim/PycharmProjects/PrognosisGraph/out/'
    log_dir = out_dir + 'logs/'
    version = "REF_feat_BS67_IS128_2"
    last_epoch = [22, 13, 10, 9]
    root_ckpt_dir = out_dir + 'checkpoints/' + version
    check_point_dir = root_ckpt_dir + '/RUN_' + str(split_number) + '/' + '{}.pkl'.format(
        "epoch_" + str(last_epoch[split_number]))
    check_pt = torch.load(check_point_dir, map_location=torch.device('cpu'))
    model = Transformer(net_params)
    model.load_state_dict(check_pt)
    device = net_params['device']
    epoch = 0
    device = 'cpu'
    pred, encoding = model.forward(g)
    return encoding


def get_vertices(a):
    edges = []
    feat = []
    for i in range(a.shape[1]):
        for j in range(a.shape[0]):
            if a[i, j] != 0:
                edges.append((i, j))
                feat.append(a[i, j])
    return edges, feat


class Patient:
    def __init__(self, id, clinical_data, meta_data, data_dir, preprocess_data, img_mod='CR'):
        self.id = id
        self.loc = clinical_data[clinical_data['to_patient_id'] == id].index.values[0]
        self.ehr = preprocess_data.iloc[self.loc]
        self.img_mod = img_mod
        self.meta_data = meta_data.loc[(meta_data['Subject ID'] == id) & (meta_data['Modality'] == img_mod)]
        self.retain = True if self.meta_data.shape[0] != 0 else False
        self.data_dir = data_dir

    def get_img(self):
        if self.meta_data.shape[0] <= 2:
            img_path = self.meta_data['File Location'].iloc[0]
        elif self.meta_data.shape[0] > 2:
            """date_vector = pd.to_datetime(self.meta_data['Study Date'])
            min_date = date_vector.min()
            loc = date_vector[date_vector == min_date].reset_index"""
            img_path = self.meta_data['File Location'].iloc[0]
        filename = {"image": self.data_dir + img_path + '/1-1.dcm'}
        """transform = Compose([
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Resized(keys="image", spatial_size=[3000, 2500, 1]),
            EnsureTyped("image"),
        ])
        result = transform(filename)"""
        return filename #result['image']

    def get_ehr(self):
        return self.ehr

    def get_status(self):
        return self.ehr['last.status']

def get_img(img_path_lst):
    lst_result = []
    for i in range(len(img_path_lst)): #tqdm()
        filename = eval(img_path_lst[i])
        transform = Compose([
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Resized(keys="image", spatial_size=[1, 512, 512]),
            EnsureTyped("image"),
        ])
        result = transform(filename)['image']
        if i == 0:
            lst_result = result
        else:
            lst_result = torch.cat((lst_result, result), dim=0)
    return lst_result
def visualize_data(data, outcome):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isna().transpose(),
                cmap="YlGnBu",
                cbar_kws={'label': 'Missing Data'})
    plt.savefig("visualizing_missing_data_with_heatmap_Seaborn_Python.png", dpi=100)

def get_graph_from_data():
    label_lists = []
    graph_lists = []
    i = 0
    j = 0
    data_dir = "C:/Users/maxim/OneDrive/Bureau/PrognosisGraph/manifest-1628608914773"
    meta_data = pd.read_csv(data_dir + '/metadata.csv')
    clinical_data = pd.read_csv(data_dir + "/deidentified_overlap_tcia.csv.cleaned.csv_20210806.csv")
    clinical_stat = pd.read_csv(data_dir + "/deidentified_overlap_tcia.csv.cleaned.csv.template_20210806.csv").set_index('column_name')
    var_to_del = ['to_patient_id', 'visit_start_datetime']
    clinical_stat = clinical_stat.drop(var_to_del, axis=0)
    clinical_stat['column_type'][clinical_stat['column_type'] == 'int64'] = 'numeric'
    numerical_var = clinical_stat[clinical_stat['column_type'] == 'numeric'].index.values
    categorical_var = clinical_stat[clinical_stat['column_type'] != 'numeric'].index.values
    numerical_pipeline = make_pipeline(SimpleImputer(strategy='mean'),
                                       StandardScaler())  # ,VarianceThreshold(threshold=0)
    categorical_pipeline = make_pipeline(OrdinalEncoder(), KNNImputer())  # handle_unknown = 'ignore'
    preprocessor = make_column_transformer((numerical_pipeline, numerical_var),
                                           (categorical_pipeline, categorical_var))
    preprocess_data = preprocessor.fit_transform(clinical_data)
    #onehot_col = preprocessor.named_transformers_['pipeline-2'].named_steps['onehotencoder'].get_feature_names_out(categorical_var)
    feat_col = list(numerical_var) + list(categorical_var)
    preprocess_data = pd.DataFrame(preprocess_data, columns=list(numerical_var) + list(categorical_var))#+ list(onehot_col)
    patient_df = pd.DataFrame(columns=['id'] + feat_col + ['img'])

    var_to_keep_out = []#'age.splits', 'gender_concept_name'
    ehr = []
    img = []
    _id = []
    status = []
    for id in clinical_data['to_patient_id']:
        patient = Patient(id, clinical_data, meta_data, data_dir, preprocess_data, img_mod='CR')
        if patient.retain:
            _id.append(id)
            ehr.append(patient.get_ehr())
            img.append(patient.get_img())
            status.append(patient.get_status())
            j += 1
        else:
            #print('patient {} has no img'.format(id))
            i += 1
    print("number of discard patient = {}".format(i))
    print("number of retain patient (missing img )= {}".format(j))
    patient_df['id'] = _id
    patient_df['img'] = img
    patient_df[feat_col] = ehr
    patient_df['fold'] = np.zeros_like(_id)

    kf = StratifiedKFold(n_splits=20, shuffle=True) #KFold(n_splits=134*10, shuffle=True, random_state=42)
    idx = kf.split(_id, status)
    i = 0
    for fold in idx:
        patient_df.loc[fold[1], 'fold'] = i
        i+=1
    patient_df.to_csv('preprocess_data.csv')
    df = pd.read_csv('preprocess_data.csv', index_col=0)
    folds = df['fold'].unique()
    outcome = 'last.status'
    ehr_var = df.drop([outcome, 'fold', 'img', 'id'], axis=1).columns
    numerical_len = len(df.drop([outcome, 'fold', 'img', 'id'], axis=1)[numerical_var].columns)
    categorical_len = len(df.drop(['fold', 'img', 'id'], axis=1)[categorical_var].drop([outcome], axis=1).columns)#var_to_keep_outv
    unique_cat = []
    for cat_var in df.drop([ 'fold', 'img', 'id', ], axis=1)[categorical_var].drop([outcome], axis=1).columns:#var_to_keep_out
        unique_cat.append(len(df[cat_var].unique()))
    info_dict = {
                 'unique_cat': unique_cat}
    info = pd.DataFrame(info_dict['unique_cat'])
    info.to_csv('info.csv')
    info_dict ={'ehr_var': ehr_var,
                 'numerical_len': numerical_len,
                 'categorical_len': categorical_len}
    info2 = pd.DataFrame(info_dict)
    info2.to_csv('info2.csv')
    for i in tqdm(range(len(folds))):
        fold = folds[i]
        index = [k for k, x in enumerate(df['fold'].values.tolist()) if
                 x == fold]
        data = df.iloc[index]
        labels = np.array(data[outcome].to_numpy(dtype=np.int8))
        g = dgl.DGLGraph()
        g.add_nodes(data.shape[0])
        '''for var in df.drop([outcome, 'fold', 'img', 'id'], axis=1).columns:
            res = np.array(data[var].values.tolist())
            g.ndata[var] = torch.tensor(res)'''
        """g.ndata['age'] = torch.tensor(np.array(data['age.splits']))
        g.ndata['gender'] = torch.tensor(np.array(data['gender_concept_name']))""" # TODO: more generic
        #g.ndata['id'] = torch.tensor(np.array(data['id'].values.tolist()))
        g.ndata['label'] = torch.tensor(np.array(data[outcome].values.tolist()))
        g.ndata['ehr'] = torch.tensor(np.array(data[df.drop([outcome, 'fold', 'img', 'id'], axis=1).columns].values.tolist()))#var_to_keep_out
        g.ndata['img'] = get_img(data['img'].values.tolist())
        #g.ndata['feat'] = torch.tensor(extract_feat(g))
        A = np.ones((data.shape[0], data.shape[0]))
        edge = np.array(get_vertices(A)[0])
        edge_feat = np.array(get_vertices(A)[0])
        for src, dst in edge:
            g.add_edges(src.item(), dst.item())
        edge_feat = np.array(edge_feat)
        g.edata['feat'] = torch.tensor(edge_feat).long()
        # g = dgl.transform.remove_self_loop(g)
        onehot = [1, 1]
        tmp = []
        for i in labels:
            onehot[i] = 0
            tmp.append(onehot)
            onehot = [1, 1]
        label_lists.append(torch.tensor(tmp))
        graph_lists.append(g)
    return label_lists, graph_lists


def plot_stat():
    data_dir = "C:/Users/maxim/OneDrive/Bureau/PrognosisGraph/manifest-1628608914773"
    clinical_data = pd.read_csv(data_dir + "/deidentified_overlap_tcia.csv.cleaned.csv_20210806.csv")
    clinical_data = pd.DataFrame(clinical_data[4:]).reset_index(0,
                                                                drop=True)  # TODO: remove patient with high missing data and label null

    plt.figure(figsize=(10, 6))
    sns.heatmap(clinical_data.isna().transpose(),
                cmap="YlGnBu",
                cbar_kws={'label': 'Missing Data'})
    plt.show()


    plt.figure(figsize=(10,6))
    sns.displot(
        data=clinical_data.isna().melt(value_name="missing"),
        y="variable",
        hue="missing",
        multiple="fill",
        aspect=1.25
    )

    plt.show()

    plt.figure()
    sns.heatmap(clinical_data.corr())
    plt.show()

    plt.figure()
    plt.matshow(clinical_data.corr())
    plt.show()



    data = clinical_data
    plt.figure(figsize=(12, 8))
    corr = data.corr()
    mask = np.tri(*corr.shape).T
    sns.heatmap(corr.abs(), mask=mask, annot=True)
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.title('Correlation between data')
    plt.show()

    plt.figure()
    n_fts = len(data.columns)
    colors = cm.rainbow(np.linspace(0, 1, n_fts))
    data1 = data
    data.drop('last.status', axis=1).corrwith(data1['last.status']).sort_values(ascending=True).plot(kind='barh',
                                                                                         color=colors,
                                                                                         figsize=(12, 8))
    plt.title('Correlation to Target (outcome)')
    plt.show()

    plt.figure()
    sns.pairplot(clinical_data, kind="reg")
