"""
    Utility file to select GraphNN model as
    selected by the user
"""


from nets.STOIC.graphsage_net import GraphSageNet
from nets.STOIC.EdgePredict import EdgePredict
from nets.STOIC.GATNet import GATNet
from layers.SIMNet import SIMNet
from nets.STOIC.STOIC_V2_net import STOIC_V2
def GraphSage(net_params, tresh=None):

    return GraphSageNet(net_params, tresh)
def gatnet(net_params, tresh=None) :
    return GATNet(net_params, tresh)
def stoic_v2(net_params, tresh=None):
    return STOIC_V2(net_params, tresh)
def simnet(net_params, tresh=None):
    return SIMNet(net_params, tresh)
def gnn_model(MODEL_NAME, net_params, tresh=None):

    models = {
        'GAT': gatnet,
        'GraphSage': GraphSage,
        'EdgePredict' : EdgePredict,
        'SIMNet' : simnet,
        'STOIC_V2' : stoic_v2}

    return models[MODEL_NAME](net_params, tresh)