#!/usr/bin/env python3
# run_baselines.py (Aligned with Train Version GTLGT)

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from torch import optim
import random

# ===== 导入你 train 里的模型 =====
from model import (
    GCNNet,
    GATEncoder,
    SimpleTransformerEncoder,
    Classifier,
    GTLGT
)

# ---------------------------
# Utility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed()

def load_planetoid(name, data_root='./data'):
    path = os.path.join(data_root, name)
    dataset = Planetoid(root=path, name=name)
    return dataset[0]

def adjacency_from_data(data):
    return to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]

# ---------------------------
# Single Graph Mode
# ---------------------------
def run_single(data, args):

    results = {}

    # ===== GCN Baseline =====
    gcn = GCNNet(data.num_features, args.hid_dim, args.embed_dim).to(device)
    clf = Classifier(args.embed_dim, int(data.y.max())+1).to(device)

    opt = optim.Adam(list(gcn.parameters())+list(clf.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(args.epochs):
        gcn.train(); clf.train()
        opt.zero_grad()
        z = gcn(data.x.to(device), data.edge_index.to(device))
        loss = loss_fn(clf(z)[data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        opt.step()

        gcn.eval(); clf.eval()
        with torch.no_grad():
            z = gcn(data.x.to(device), data.edge_index.to(device))
            preds = clf(z).argmax(dim=-1).cpu()
            acc = accuracy_score(data.y[data.test_mask].cpu(), preds[data.test_mask])
            best_acc = max(best_acc, acc)

    results['GNN'] = {'ACC': best_acc, 'AE': 0.0, 'CS': args.epochs}

    # ===== GTLGT（正式版本）=====
    encoder = GATEncoder(data.num_features, args.hid_dim, args.embed_dim).to(device)
    transformer = SimpleTransformerEncoder(args.embed_dim).to(device)
    classifier = Classifier(args.embed_dim, int(data.y.max())+1).to(device)

    model = GTLGT(
        encoder=encoder,
        transformer=transformer,
        classifier=classifier,
        decoder_type="simple",
        embed_dim=args.embed_dim,
        n_nodes=data.num_nodes
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()

        logits, z_raw, z_final = model(data.x.to(device), data.edge_index.to(device))
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits, _, _ = model(data.x.to(device), data.edge_index.to(device))
            preds = logits.argmax(dim=-1).cpu()
            acc = accuracy_score(data.y[data.test_mask].cpu(), preds[data.test_mask])
            best_acc = max(best_acc, acc)

    results['GTLGT'] = {'ACC': best_acc, 'AE': 0.0, 'CS': args.epochs}

    return results

# ---------------------------
# Transfer Mode (Using OT)
# ---------------------------
def run_transfer(src, tgt, args):

    results = {}

    encoder = GATEncoder(src.num_features, args.hid_dim, args.embed_dim).to(device)
    transformer = SimpleTransformerEncoder(args.embed_dim).to(device)
    classifier = Classifier(args.embed_dim, int(src.y.max())+1).to(device)

    model = GTLGT(
        encoder=encoder,
        transformer=transformer,
        classifier=classifier,
        decoder_type="simple",
        embed_dim=args.embed_dim,
        n_nodes=max(src.num_nodes, tgt.num_nodes)
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # train on source
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits, _, _ = model(src.x.to(device), src.edge_index.to(device))
        loss = loss_fn(logits[src.train_mask], src.y[src.train_mask].to(device))
        loss.backward()
        opt.step()

    # get embeddings
    model.eval()
    with torch.no_grad():
        _, zA_raw, zA = model(src.x.to(device), src.edge_index.to(device))
        _, zB_raw, zB = model(tgt.x.to(device), tgt.edge_index.to(device))

    # compute OT matrix
    model.coupling.compute_P(zA, zB)

    A = adjacency_from_data(src).to(device)
    B = adjacency_from_data(tgt).to(device)
    AE = model.coupling(A, B, zA, zB).item()

    # evaluate transfer accuracy
    with torch.no_grad():
        logits, _, _ = model(tgt.x.to(device), tgt.edge_index.to(device))
        preds = logits.argmax(dim=-1).cpu()
        acc = accuracy_score(tgt.y.cpu(), preds)

    results['GTLGT_transfer'] = {
        'ACC': acc,
        'AE': AE,
        'CS': args.epochs
    }

    return results

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--mode', choices=['single','transfer'], default='single')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--data_root', type=str, default='./data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    if args.mode == 'single':
        data = load_planetoid(args.dataset, args.data_root)
        results = run_single(data, args)
    else:
        src_name, tgt_name = args.dataset.split(',')
        src = load_planetoid(src_name.strip(), args.data_root)
        tgt = load_planetoid(tgt_name.strip(), args.data_root)
        results = run_transfer(src, tgt, args)

    print("Results:", results)
