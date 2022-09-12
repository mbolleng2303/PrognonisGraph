import torch
from torch.nn import functional as F
from torch.autograd import Variable

def softmax_mse(input1, input2):
    assert input1.size() == input2.size()
    input_softmax = F.softmax(input1, dim=1)
    target_softmax = F.softmax(input2, dim=1)
    feat = input1.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / feat


def softmax_kl(input1, input2):
    assert input1.size() == input2.size()
    input_log_softmax = F.log_softmax(input1, dim=1)
    target_softmax = F.softmax(input2, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def MSE(input1, input2):
    assert input1.size() == input2.size()
    input1 = F.normalize(input1, dim=-1, p=2)
    input2 = F.normalize(input2, dim=-1, p=2)
    return torch.sum(2 - 2 * (input1 * input2))  ###2 - 2 * (input1 * input2).sum(dim=-1) ###recheck this one


def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    feat = input1.size()[1]
    return torch.sum((input1 - input2) ** 2) / feat


def poly_kernel(input1, d=0.5, alpha=1.0, c=2.0):
    K_XX = torch.mm(input1, input1.t()) + c
    return K_XX.pow(d)


def smi(input1, input2):
    K_X = poly_kernel(input1)
    K_Y = poly_kernel(input2)
    n = K_X.size(0)
    phi = K_X * K_Y
    hh = torch.mean(phi, 1)
    Hh = K_X.mm(K_X.t()) * K_Y.mm(K_Y.t()) / n ** 2 + torch.eye(n)
    alphah = torch.matmul(torch.inverse(Hh), hh)

    smi = 0.5 * torch.dot(alphah, hh) - 0.5
    return smi  # , alphah
