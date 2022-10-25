import torch
import torch.nn as nn
from tab_transformer_pytorch_cat import TabTransformer

cont_mean_std = torch.randn(10, 2)

model = TabTransformer(
    categories = (10, 5, 6, 5, 8),      # tuple containing the number of unique values within each category
    num_continuous = 10,                # number of continuous values
    dim = 256,                           # dimension, paper set at 32 this is just a test with 256
    dim_out = 2,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1,                   # feed forward dropout
    mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
)

x_categ = torch.randint(0, 5, (1, 5))     # category values, from 0 - max number of categories, in the order as passed into the constructor above
x_cont = torch.randn(1, 10)               # assume continuous values are already normalized individually
x_image = torch.rand((1, 1, 256, 256))    # assume a gray level (1-channel) image of 256 \times 256 for example this can be the x-ray

pred = model(x_categ, x_cont, x_image)

print('Done')