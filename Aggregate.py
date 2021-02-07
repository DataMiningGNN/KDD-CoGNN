# -*- coding: utf-8 -*-

import copy
import torch
import torch.nn.functional as F
from scipy import linalg
import numpy as np
import sys


def aggregate_att(w_clients, w_server, stepsize, metric, dp, device):
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param stepsize: step size for aggregation
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_next = copy.deepcopy(w_server)
    att, att_mat = {}, {}
    for k in w_server.keys():
        # w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()

    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            att[k][i] = torch.from_numpy(np.array(linalg.norm(\
                              w_server[k].cpu().numpy()-w_clients[i][k].cpu().numpy()))) ## ord=metric

    for k in w_next.keys():
        att[k] = F.softmax(att[k], dim=0)

    # att = att.cuda()
    for k in w_next.keys():
        att_weight = torch.zeros_like(w_server[k]).cuda()
        for i in range(0, len(w_clients)):
            att_weight = att_weight + torch.mul(w_server[k]-w_clients[i][k], att[k][i]).float()
        w_next[k] = w_server[k] - torch.mul(att_weight, stepsize) # + torch.mul(torch.randn(w_server[k].shape).cuda(), dp)

    return w_next


def average_weights_dp(w, dp,device):
    """
    Federated averaging
    :param w: list of client model parameters
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k] + w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w)) + torch.mul(torch.randn(w_avg[k].shape).cuda(), dp)
    return w_avg


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
