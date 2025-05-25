
# used implementation of Barlow Twins loss from https://github.com/RishikeshMagar/Crystal-Twins/blob/main/CT_transfer/loss/barlow_twins.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 15:41:13 2021

@author: rishikesh
"""
import torch
import numpy as np
from torch import nn


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    def __init__(self, device, params):
        super(BarlowTwinsLoss, self).__init__()

        self.device = device
        self.embed_size = params['model_params']['MLP2_dim']
        self.batch_size = params['batch_size']
        self.lambd = params['lambda_param']
        self.bn = nn.BatchNorm1d(self.embed_size, affine=False).to(self.device)

    def forward(self, z1, z2):

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2).to(self.device)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().to(self.device)
        off_diag = off_diagonal(c).pow_(2).sum().to(self.device)
        loss = on_diag + self.lambd * off_diag
        return loss.to(self.device)
