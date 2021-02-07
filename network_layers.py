from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init


###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True # default is linear model
        self.num_layers = num_layers

        # print('self.num_layers:', self.num_layers)
        # self.num_layers: 2

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        ### logits = F.relu(logits)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() # / float(y.size()[0])
            return logits, loss, acc
        else:
            return logits


class MLPTransformation(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = None, with_dropout=False):
        super(MLPTransformation, self).__init__()

        self.hidden_size = hidden_size

        if hidden_size is None:
            self.h1_weights = nn.Linear(input_size, output_size)
        else:
            self.h1_weights = nn.Linear(input_size, hidden_size)
            self.h2_weights = nn.Linear(hidden_size, output_size)

        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x):
        if self.hidden_size:
            h1 = self.h1_weights(x)
            h1 = F.relu(h1)
            if self.with_dropout:
                h1 = F.dropout(h1, training=self.training)
            out = self.h2_weights(h1)
        else:
            out = self.h1_weights(x)
            # out = F.relu(out)
            if self.with_dropout:
                out = F.dropout(out, training=self.training)

        return out      


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0):  ## , bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Linear(input_dim, output_dim) ## nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        
        # if bias:
        #     self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        # else:
        #     self.bias = None

    def forward(self, x, adj):
        # adj = F.sigmoid(adj)

        if self.dropout > 0.001:
            x = self.dropout_layer(x)

        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        # y = torch.matmul(y, self.weight)
        y = self.weight(y)
        # if self.bias is not None:
        #     y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            #print(y[0][0])
            
        # y = F.relu(y) ## add on 9-20
        # y = F.sigmoid(y) ## add on 9-20
        return y


class FTL_Classification(nn.Module):
    def __init__(self, args, G_rep_dim):
        super(FTL_Classification, self).__init__()
        print('Initializing FTL_Classification')
        self.G_rep_dim = G_rep_dim
        self.mlp = MLPClassifier(input_size=self.G_rep_dim, hidden_size=args.hidden_dim, num_class=args.num_target_class, with_dropout=args.dropout)

    def forward(self, graph_representations):
        output = self.mlp(graph_representations)
        return output