# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/12/30
@function: load data and preprocess. when train, gcn need all nodes and edges, but, calulate loss only need train_labels
"""
import sys
import logging
import pickle as pkl
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sample_mask(idx, size):
    mask = np.zeros(size)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(source="cora", path="./data"):
    """
    Load input data --cora
    ind.cora.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object; [140, 1433]
    ind.cora.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object; [1000, 1433]
    ind.cora.allx => the feature vectors of both labeled and unlabeled training instances (a superset of x) as scipy.sparse.csr.csr_matrix object; [1708, 1433]
    ind.cora.y => the one-hot labels of the labeled training instances as numpy.ndarray object; [140, 7]
    ind.cora.ty => the one-hot labels of the test instances as numpy.ndarray object; [1000, 7]
    ind.cora.ally => the labels for instances in ind.cora.allx as numpy.ndarray object; [1708, 7]
    ind.cora.graph => a dict in the format {index: [index or neighbor_nodes]} as collection.defaultdict object;
    ind.cora.test.index => the indices of test instances in graph, for the inductive setting as list object.
    :param source: dataset name
    :param path: save path, dir name
    :return: adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    """
    logger.info(f"==>loading {source} dataset from {path}.")
    suffixs = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for i in range(len(suffixs)):
        with open(f"{path}/ind.{source}.{suffixs[i]}", "rb") as file:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(file, encoding="latin1"))
            else:
                objects.append(pkl.load(file))

    x, y, tx, ty, allx, ally, graph = objects

    test_idx_reorder = []
    with open(f"{path}/ind.{source}.test.index", encoding="utf8") as file:
        for line in file:
            test_idx_reorder.append(int(line.strip()))
    test_idx_range = np.sort(test_idx_reorder)

    # total data = allx+tx
    features = sp.vstack([allx, tx]).tolil()  # [2708, 1433]
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack([ally, ty])  # [2708, 7]
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # adjacency matrix
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # get train/eval/test indices
    idx_test = test_idx_range.tolist()  # 1708-2707
    idx_train = range(len(y))  # 0-139
    idx_val = range(len(y), len(y) + 500)  # 140-639

    # mask
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # get train/eval/test label
    y_train = np.zeros(labels.shape)  # [2708, 7]
    y_val = np.zeros(labels.shape)  # [2708, 7]
    y_test = np.zeros(labels.shape)  # [2708, 7]
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def preprocess_features(features):
    """Row normalize feature matrix and convert to tuple representation
    1.x/x.sum(), dim=1
    2.coordination  row, col and value, as sp.coo_matrix
    """
    logger.info("==>preprocess features.")
    rowsum=np.array(features.sum(1)) # [2708,1]
    r_inv=np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)]=0.0
    r_mat_inv=sp.diags(r_inv) # [2708, 2708]
    features=r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def preprocess_adj(adj):
    """preprocess adjacency matrix and convert to tuple representation.
    1. normalize
    2. convert to tuple
    """
    logger.info("==>preprocess adj.")
    adj=adj+sp.eye(adj.shape[0])
    adj=sp.coo_matrix(adj)
    rowsum=np.array(adj.sum(1))
    d_inv_sqrt=np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)]=0.0
    d_mat_inv_sqrt=sp.diags(d_inv_sqrt)
    adj_norm=adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_norm)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx=mx.tocoo()
        coords = np.vstack((mx.row, mx.col))
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i]=to_tuple(sparse_mx[i])
    else:
        sparse_mx=to_tuple(sparse_mx)
    return sparse_mx




if __name__ == '__main__':
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data()

    features=preprocess_features(features)

    adj=preprocess_adj(adj)
    print(adj)