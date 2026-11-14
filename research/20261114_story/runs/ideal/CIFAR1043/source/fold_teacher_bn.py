"""Evaluation-time batch-norm folding for the adapted full-width GCN teacher."""
import copy
import torch
import torch.nn as nn


def fold_conv_bn(conv, bn):
    if bn.training or not bn.track_running_stats or not bn.affine:
        raise ValueError("Expected eval-mode affine BN with running statistics")
    if conv.bias is None:
        raise ValueError("Expected GraphConv bias")
    folded = copy.deepcopy(conv)
    with torch.no_grad():
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        folded.lin.weight.mul_(scale[:, None])
        folded.bias.copy_(scale * (conv.bias - bn.running_mean) + bn.bias)
    return folded


def fold_model(model):
    if model.training:
        raise ValueError("Model must be in eval mode before folding")
    clone = copy.deepcopy(model)
    for i, (conv, bn) in enumerate(zip(model.convs, model.norms)):
        clone.convs[i] = fold_conv_bn(conv, bn)
        clone.norms[i] = nn.Identity()
    return clone
