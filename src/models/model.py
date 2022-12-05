import math
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from ..makedata.make_data import *
from ..features.log import *

class GraphConvolution(Module):
    '''
    GCN layer
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    '''
    GCN model
    '''
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class FCN(nn.Module):
    '''
    FCN model
    '''
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(FCN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(nfeat, nhid),
            nn.ReLU(), 
            nn.Linear(nhid, nclass)
        )
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.fc(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


def train(epoch, model, features, adj, labels, idx_train, idx_val, task, to_train=True):
    t = time.time()
    optimizer = optim.Adam(model.parameters(),
                lr=0.01, weight_decay=5e-4)
    gcn_err, gcn_acc = [], []

    for e in epoch:
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            if to_train:
                loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                acc_val = accuracy(output[idx_val], labels[idx_val])
                towrite = "Epoch: {:04d}\n".format(e+1)
                towrite += "loss_train: {:.4f}\n".format(loss_train.item())
                towrite += "loss_val: {:.4f}\n\n".format(loss_val.item())
                log_curr("./data/out/", task, towrite, to_train=to_train)
        
                loss_val, acc_val = float(loss_val.to('cpu').numpy()), float(acc_val.to('cpu').numpy())
                gcn_err.append(loss_val)
                gcn_acc.append(acc_val)

    print(f"{task} optimization finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t))

    return (gcn_err, gcn_acc)

def test(model, features, adj, labels, idx_test):
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))
    
    return (loss_test, acc_test)

def run_gcn(path, dataset, feat_suf, edge_suf, epoch, task, to_train=True):
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path=path, dataset=dataset,\
                feat_suf=feat_suf, edge_suf=edge_suf)

    model = GCN(nfeat=features.shape[1],
            nhid=features.shape[1]//2, 
            nclass=labels.max().item() + 1,
            dropout=0.5) # hyperparameter

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    gcn_err, gcn_acc = train(epoch, model, features, adj, labels, idx_train,\
                             idx_val, task, to_train=to_train)

    return gcn_err, gcn_acc

def run_fcn(path, dataset, feat_suf, edge_suf, epoch, task, to_train=True):
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path=path, dataset=dataset,\
                feat_suf=feat_suf, edge_suf=edge_suf)

    model = FCN(nfeat=features.shape[1],
            nhid=features.shape[1]//2, 
            nclass=labels.max().item() + 1,
            dropout=0.5) # hyperparameter

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    fcn_err, fcn_acc = train(epoch, model, features, adj, labels, idx_train,\
                             idx_val, task, to_train=to_train)

    return fcn_err, fcn_acc