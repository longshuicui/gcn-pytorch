# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/12/30
@function:
"""
import time
import logging
import argparse
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from model import GCN
from utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--save_path", type=str, default="./model.pt")


    args = parser.parse_args()
    return args


def accuracy(y_true, y_pred, mask):
    correct_prediction= (torch.argmax(y_true,dim=-1)) == (torch.argmax(y_pred,dim=-1))
    acc=torch.sum(correct_prediction.float()*mask)/mask.sum()
    return acc.item()


def main(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset, args.path)
    features = preprocess_features(features)
    adj = preprocess_adj(adj)

    feature_dim = features[2][1]
    num_classes = y_train.shape[1]
    model=GCN(feature_dim=feature_dim, num_classes=num_classes, hidden_size=args.hidden_size).to(device)
    model.load_state_dict()

    # add L2 to weight except bias
    weight_p, bias_p=[], []
    for name, p in model.named_parameters():
        if "bias" in name:
            bias_p+=[p]
        else:
            weight_p+=[p]

    optimizer=torch.optim.Adam(params=[{"params":weight_p,"weight_decay":args.weight_decay},
                                       {"params":bias_p,"weight_decay":0.}],
                               lr=args.lr)

    inputs = torch.sparse_coo_tensor(indices=features[0], values=features[1], size=features[2], dtype=torch.float32).to(device)
    A = torch.sparse_coo_tensor(indices=adj[0], values=adj[1], size=adj[2], dtype=torch.float32).to(device)
    for epoch in range(args.epochs):
        model.train()
        train_labels = torch.from_numpy(y_train).to(device)
        train_labels_mask = torch.from_numpy(train_mask).to(device)
        train_loss, train_logits=model(inputs, A, train_labels, train_labels_mask)
        train_acc=accuracy(train_labels, train_logits,train_labels_mask)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        eval_labels=torch.from_numpy(y_val).to(device)
        eval_labels_mask=torch.from_numpy(val_mask).to(device)
        eval_loss, eval_logits=model(inputs, A, eval_labels, eval_labels_mask)
        eval_acc=accuracy(eval_labels, eval_logits, eval_labels_mask)
        logger.info(f"Epoch:{epoch+1}/{args.epochs}, "
                    f"train loss:{train_loss:.4f}, "
                    f"train acc:{train_acc:.4f}, "
                    f"eval loss: {eval_loss:.4f}, "
                    f"eval acc:{eval_acc:.4f}.")

    torch.save({"structure":model,"state_dict":model.state_dict()},f=args.save_path)


def infer(args):
    items=torch.load(args.save_path)
    model=items["structure"]
    state_dict=items["state_dict"]
    model.load_state_dict(state_dict=state_dict)

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset, args.path)
    features = preprocess_features(features)
    adj = preprocess_adj(adj)

    inputs = torch.sparse_coo_tensor(indices=features[0], values=features[1], size=features[2], dtype=torch.float32)
    A = torch.sparse_coo_tensor(indices=adj[0], values=adj[1], size=adj[2], dtype=torch.float32)
    output=model(inputs, A)

    embeddings=output.detach().numpy()[test_mask]
    y=y_test.argmax(axis=-1)[test_mask]

    plot_embeddings(embeddings, np.arange(y.shape[0]), y)
    # y_pred=torch.argmax(output, dim=-1)
    # print(y_pred[test_mask])
    #
    # y_test=torch.from_numpy(y_test)
    # test_mask=torch.from_numpy(test_mask)
    # acc=accuracy(y_test,output, test_mask)
    # print(acc)


def plot_embeddings(embeddings, X, Y):
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i], [])
        color_idx[Y[i]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    args=add_arguments()
    # main(args)
    infer(args)