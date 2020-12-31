# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/12/29
@function: the implement of GCN by pytorch
"""
import math
import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, act=nn.ReLU(), bias=True, dropout_prob=0.0):
        """
        initialize class
        :param in_feature: the dimension of input feature
        :param out_feature: the dimension of output feature
        :param act: activation
        :param bias:  whether to add bias
        """
        super(GraphConvolution, self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.act=act if act is not None else None
        self.dropout=nn.Dropout(dropout_prob)
        self.weight=nn.Parameter(torch.FloatTensor(in_features, out_features),requires_grad=True)
        if bias:
            self.bias=nn.Parameter(torch.FloatTensor(out_features),requires_grad=True)
        else:
            self.register_parameter("bias",None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv=1.0/math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """
        compute output
        O=AXW, X as layer-i inputs, O as layer-i output as well as layer-i+1 input
        :param x: input feature of node in graph
        :param adj: adjacency matrix-A
        :return: output(logits)
        """
        support=torch.spmm(x, self.weight)
        output=torch.spmm(adj, support)
        if self.bias is not None:
            output=output+self.bias
        if self.act is not None:
            output=self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__+" ("+str(self.in_features)+" -> "+str(self.out_features)+") "


class GCN(nn.Module):
    def __init__(self,
                 feature_dim,
                 hidden_size=16,
                 num_classes=7,
                 dropout_prob=0.0):
        super(GCN, self).__init__()
        self.layer1=GraphConvolution(in_features=feature_dim,
                                     out_features=hidden_size,
                                     act=nn.ReLU(),
                                     dropout_prob=dropout_prob)
        self.layer2=GraphConvolution(in_features=hidden_size,
                                     out_features=num_classes,
                                     act=None,
                                     dropout_prob=dropout_prob)
        self.softmax=nn.Softmax(-1)
        self.ce=nn.CrossEntropyLoss(reduction='none')

    def forward(self, X, adj, labels=None, labels_mask=None):
        output=self.layer1(X, adj)
        logits=self.layer2(output,adj)
        if labels is not None:
            loss=self.ce(logits, torch.argmax(labels,dim=-1))
            loss=torch.sum(loss*labels_mask)/labels_mask.sum()
            return loss, logits
        else:
            output=self.softmax(logits)
            return output






if __name__ == '__main__':
    inputs = torch.FloatTensor([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6,5]])
    adjs = torch.FloatTensor([
        [0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ])
    labels=torch.FloatTensor([[1,0],[1,0],[0,1],[1,0],[0,1],[0,1]])
    labels_mask=torch.FloatTensor([True,True,True,True,False,False])

    model=GCN(2,6,2)
    loss, logits=model(inputs, adjs, labels, labels_mask)
    print(loss)
    print(logits)

