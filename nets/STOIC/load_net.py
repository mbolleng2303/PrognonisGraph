"""
    Utility file to select GraphNN model as
    selected by the user
"""


from nets.STOIC.graphsage_net import GraphSageNet
from nets.STOIC.EdgePredict import EdgePredict
from nets.STOIC.GATNet import GATNet
def GraphSage(net_params, tresh=None):

    return GraphSageNet(net_params, tresh)
def gatnet(net_params, tresh=None) :
    return GATNet(net_params, tresh)

def gnn_model(MODEL_NAME, net_params, tresh=None):

    models = {
        'GAT': gatnet,
        'GraphSage': GraphSage,
        'EdgePredict' : EdgePredict }

    return models[MODEL_NAME](net_params, tresh)