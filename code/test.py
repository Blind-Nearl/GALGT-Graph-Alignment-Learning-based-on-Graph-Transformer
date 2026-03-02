"""
train.py (OT-Sinkhorn version with Kendall weighting on L_cls & L_recon only)
Usage example:
    python train.py --mode transfer --model gt --dataset p1,p1 --epoch 200 --lr 0.005
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from dataload import load_dataset
from model import (
    GCNNet, GATEncoder, SimpleTransformerEncoder,
    Classifier, GTLGT
)

# ----------------------------
# Utilities / Metrics
# ----------------------------
def accuracy(logits, labels):
    """ logits: [N, num_classes], labels: [N] """
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item()

def cosine_similarity_matrix(z1, z2):
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)
    return (z1 @ z2.t()).mean().item()

# ----------------------------
# Training: single-graph
# ----------------------------
def train_single(model, data, lr, epochs, early_stopping, cs_threshold=0.1, device=None):
    device = device or torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = 0
    cs_epoch = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits, z_raw, z_final = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc_test = accuracy(logits[data.test_mask], data.y[data.test_mask])

        if acc_test > best_acc:
            best_acc = acc_test
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping:
                print(f"[Single] Early stopping at epoch {epoch}")
                break

        if cs_epoch is None and loss.item() <= cs_threshold:
            cs_epoch = epoch

        # 🔴 每个 epoch 都打印（原先是每10轮）
        print(f"[Single][Epoch {epoch:03d}] loss={loss.item():.6f} acc_test={acc_test:.4f}")

    return best_acc, cs_epoch or epochs

# ----------------------------
# Training: transfer (OT-based alternating minimization)
# ----------------------------
def train_transfer_ot(model, data_src, data_tgt, lr, epochs, early_stopping, cs_threshold=0.1, device=None,
                      weight_align=1.0):
    """
    Alternating minimization:
      - Step A: fix P, optimize W with Kendall weighting on L_cls & L_recon only.
      - Step B: fix W, compute P via OTCoupling.compute_P (Sinkhorn).
    L_align uses fixed weight_align and is NOT part of Kendall uncertainty learning.
    """
    device = device or torch.device("cpu")
    model.to(device)

    # === Learnable log variances for cls and recon (Kendall) ===
    log_var_cls = torch.nn.Parameter(torch.tensor(0.0, device=device))
    log_var_recon = torch.nn.Parameter(torch.tensor(0.0, device=device))

    # Optimizer includes model parameters + log_vars
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [log_var_cls, log_var_recon],
        lr=lr, weight_decay=5e-4
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_cs_val = -np.inf
    patience = 0
    cs_epoch = None
    AE_final = None
    acc_final = None

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()

        # ---------- Step A: optimize W (with Kendall on cls & recon) ----------
        optimizer.zero_grad()

        logits_s, z_s_raw, z_s = model(data_src.x.to(device), data_src.edge_index.to(device))
        logits_t, z_t_raw, z_t = model(data_tgt.x.to(device), data_tgt.edge_index.to(device))

        A_s = to_dense_adj(data_src.edge_index)[0].to(device)
        A_t = to_dense_adj(data_tgt.edge_index)[0].to(device)

        # Classification loss (source)
        loss_cls = criterion(logits_s[data_src.train_mask], data_src.y[data_src.train_mask])

        # Reconstruction loss
        A_hat_s = model.reconstruct_adj(z_s)
        A_hat_t = model.reconstruct_adj(z_t)
        loss_recon = 0.5 * (F.mse_loss(A_hat_s, A_s) + F.mse_loss(A_hat_t, A_t))

        # Alignment loss (OT coupling) —— NO GRAD through P, and NOT in Kendall
        if model.coupling.P_current is None:
            model.coupling.compute_P(z_s.detach(), z_t.detach())
        loss_align = model.coupling(A_s, A_t, z_s.detach(), z_t.detach())

        # ✅ Kendall weighting ONLY on L_cls and L_recon
        precision_cls = torch.exp(-log_var_cls)
        precision_recon = torch.exp(-log_var_recon)

        total_loss = (
            precision_cls * loss_cls + log_var_cls +
            precision_recon * loss_recon + log_var_recon +
            weight_align * loss_align  # ← fixed weight, no uncertainty
        )

        total_loss.backward()
        optimizer.step()

        # ---------- Step B: update P via Sinkhorn (no grad) ----------
        model.eval()
        with torch.no_grad():
            _, _, z_s = model(data_src.x.to(device), data_src.edge_index.to(device))
            _, _, z_t = model(data_tgt.x.to(device), data_tgt.edge_index.to(device))
            P_new = model.coupling.compute_P(z_s, z_t)
            diff = A_s @ P_new - P_new @ A_t
            AE_val = torch.norm(diff, p='fro') ** 2
            AE_final = AE_val.item()

            acc_tgt = accuracy(logits_t[data_tgt.test_mask].to(device),
                               data_tgt.y[data_tgt.test_mask].to(device))
            acc_final = acc_tgt

        # CS detection (based on total_loss including all terms)
        if cs_epoch is None and total_loss.item() <= cs_threshold:
            cs_epoch = epoch

        # 🔴 每个 epoch 都打印（原先是每10轮）
        print(f"[Epoch {epoch:03d}] total_loss={total_loss.item():.6f} | Acc_tgt={acc_final:.4f} | "
              f"AE={AE_final:.4f} | log_var_cls={log_var_cls.item():.3f}, log_var_recon={log_var_recon.item():.3f}")

        # Early stopping based on cross-domain similarity
        cs_signal = cosine_similarity_matrix(z_s, z_t)
        if cs_signal > best_cs_val:
            best_cs_val = cs_signal
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping:
                print(f"[Transfer] Early stopping at epoch {epoch}")
                break

    total_time = time.time() - start_time
    print(f"[Training finished in {total_time:.1f}s]")
    return acc_final or 0.0, cs_epoch or epochs, AE_final or float("nan")


# ----------------------------
# Main
# ----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.mode == "single":
        data, = load_dataset(args.dataset, mode="single")
        data = data.to(device)
        n_nodes = data.num_nodes
        data_src = data_tgt = None
    else:
        data_src, data_tgt = load_dataset(args.dataset, mode="transfer")
        data_src, data_tgt = data_src.to(device), data_tgt.to(device)
        n_nodes = data_src.num_nodes

    in_dim = (data.x.size(1) if args.mode == "single" else data_src.x.size(1))
    n_classes = int(((data.y.max() if args.mode == "single" else data_src.y.max()).item()) + 1)

    # Model selection
    if args.model == "gnn":
        encoder = GCNNet(in_dim, hid_dim=64, out_dim=args.embed_dim)
        transformer = None
        decoder_type = "simple"
    elif args.model == "gaae":
        encoder = GATEncoder(in_dim, hid_dim=16, out_dim=args.embed_dim)
        transformer = None
        decoder_type = "simple"
    elif args.model == "gt":
        encoder = GATEncoder(in_dim, hid_dim=16, out_dim=args.embed_dim)
        transformer = SimpleTransformerEncoder(dim=args.embed_dim, nhead=4, nlayers=2)
        decoder_type = "gt"
    else:
        raise ValueError("Unknown model")

    classifier = Classifier(args.embed_dim, n_classes)
    model = GTLGT(encoder, transformer, classifier, decoder_type=decoder_type, embed_dim=args.embed_dim, n_nodes=n_nodes)
    model.to(device)

    if args.mode == "single":
        acc, cs = train_single(model, data, args.lr, args.epoch, args.early_stopping, args.cs_threshold, device=device)
        print(f"Single Result — ACC={acc:.4f}, CS_epoch={cs}")
    else:
        acc_final, cs_epoch, AE_final = train_transfer_ot(
            model, data_src, data_tgt,
            lr=args.lr, epochs=args.epoch,
            early_stopping=args.early_stopping,
            cs_threshold=args.cs_threshold,
            device=device,
            weight_align=args.w_align
        )
        print("\n====== FINAL RESULT ======")
        print(f"Target Accuracy = {acc_final:.4f}")
        print(f"CS (epoch where total_loss <= {args.cs_threshold}) = {cs_epoch}")
        print(f"AE (||A_s P - P A_t||_F^2) = {AE_final:.4f}")
        print("=====================================")


# ----------------------------
# Argument parser
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="transfer", choices=["single", "transfer"])
    parser.add_argument("--model", type=str, default="gt", choices=["gnn", "gaae", "gt"])
    parser.add_argument("--dataset", type=str, default="p1,p1", help="src,tgt for transfer")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--early_stopping", type=int, default=50)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--cs_threshold", type=float, default=0.1)
    # Only alignment weight remains; cls/recon are auto-weighted by Kendall
    parser.add_argument("--w_align", type=float, default=1.0, help="Fixed weight for OT alignment loss")

    args = parser.parse_args()
    main(args)