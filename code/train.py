"""
train.py (OT-Sinkhorn + optional Frank-Wolfe for P; Kendall weighting on L_cls & L_recon)
Usage example:
    python train.py --mode transfer --model gt --dataset p1,p1 --epoch 200 --lr 0.005
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
import matplotlib.colors as colors

from dataload import load_dataset
from model import (
    GCNNet, GATEncoder, SimpleTransformerEncoder,
    Classifier, GTLGT,
    GraphBridge, compute_mmd,
    ZeroG,
    GraphLoRA,LoRALayer,
    CrossTREs
)


# ----------------------------
# Utilities
# ----------------------------
def plot_P_heatmap(P, title, save_path, log_scale=True):
    """Plot and save heatmap of matrix P with much better visibility."""
    P_np = P.detach().cpu().numpy()

    plt.figure(figsize=(7, 6))

    if log_scale:
        # 加 1e-9 避免 log(0)
        plt.imshow(P_np + 1e-9,
                   cmap="magma",
                   norm=colors.LogNorm(vmin=(P_np[P_np>0].min()+1e-12),
                                       vmax=P_np.max()))
    else:
        # 自动缩小颜色范围（对比提升）
        vmax = np.percentile(P_np, 99)   # 去掉极端高值
        plt.imshow(P_np, cmap="magma", vmin=0, vmax=vmax)

    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()


def accuracy(logits, labels):
    """ logits: [N, num_classes], labels: [N] """
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item()


def cosine_similarity_matrix(z1, z2):
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)
    return (z1 @ z2.t()).mean().item()


# ----------------------------
# Frank-Wolfe helpers
# ----------------------------
def frank_wolfe_update(P, grad, step_size=0.1):
    """
    Simple FW-like step: move toward negative gradient direction and re-project
    to (approximate) doubly-stochastic by row/col normalization.
    Note: This is a heuristic projection, not exact FW linear oracle.
    """
    P_new = P - step_size * grad  # gradient descent direction (we minimize AE)
    # clamp to positive and normalize rows/cols
    P_new = torch.clamp(P_new, min=1e-9)
    P_new = P_new / (P_new.sum(dim=1, keepdim=True) + 1e-9)
    P_new = P_new / (P_new.sum(dim=0, keepdim=True) + 1e-9)
    return P_new


def ae_gradient_wrt_P(A_s, A_t, P):
    """
    Analytical gradient of AE = ||A_s P - P A_t||_F^2 wrt P:
    Let D = A_s P - P A_t
    d/dP AE = 2 * (A_s^T D - D A_t^T)
    returns gradient tensor same shape as P
    """
    D = A_s @ P - P @ A_t
    grad = 2.0 * (A_s.t() @ D - D @ A_t.t())
    return grad


# ----------------------------
# Single-graph training
# ----------------------------
def train_single(model, data, lr, epochs, early_stopping, cs_threshold=0.1, device=None, log_every=10):
    device = device or torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = 0
    cs_epoch = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        output = model(data.x.to(device), data.edge_index.to(device))

        if args.model == 'gnn':
            logits = output
        else:
            logits = output[0]

        loss = criterion(logits[data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc_test = accuracy(logits[data.test_mask], data.y[data.test_mask].to(device))

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

        if epoch % log_every == 0 or epoch == 1:
            print(f"[Single][Epoch {epoch:03d}] loss={loss.item():.4f} acc_test={acc_test:.4f}")

    return best_acc, cs_epoch or epochs


# ----------------------------
# Transfer training (OT or FW)
# ----------------------------
def train_transfer_ot(model, data_src, data_tgt, lr, epochs, early_stopping, cs_threshold=0.1, device=None,
                      weight_align=1.0, p_update_method="sink", log_every=10, compare_P=False, out_dir="."):
    """
    Alternating minimization:
      - Step A: fix P, optimize W with Kendall weighting on L_cls & L_recon only.
      - Step B: fix W, update P via Sinkhorn (OT) or Frank-Wolfe (fw) according to p_update_method.
    """
    device = device or torch.device("cpu")
    model.to(device)

    # Kendall learnable log-variances (for cls and recon)
    log_var_cls = torch.nn.Parameter(torch.tensor(0.0, device=device))
    log_var_recon = torch.nn.Parameter(torch.tensor(0.0, device=device))

    optimizer = torch.optim.Adam(list(model.parameters()) + [log_var_cls, log_var_recon],
                                 lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_cs_val = -np.inf
    patience = 0
    cs_epoch = None
    AE_final = None
    acc_final = None

    losses_log = []
    ae_log = []
    acc_log = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # forward pass (current W)
        logits_s, z_s_raw, z_s = model(data_src.x.to(device), data_src.edge_index.to(device))
        logits_t, z_t_raw, z_t = model(data_tgt.x.to(device), data_tgt.edge_index.to(device))

        A_s = to_dense_adj(data_src.edge_index)[0].to(device)
        A_t = to_dense_adj(data_tgt.edge_index)[0].to(device)

        # 1) classification loss
        loss_cls = criterion(logits_s[data_src.train_mask], data_src.y[data_src.train_mask].to(device))

        # 2) reconstruction loss (unless ablation disables)
        if getattr(model, "ablation", None) == "no_gaae" or getattr(model, "ablation", None) == "no_gaae":
            loss_recon = torch.tensor(0.0, device=device)
        else:
            A_hat_s = model.reconstruct_adj(z_s)
            A_hat_t = model.reconstruct_adj(z_t)
            loss_recon = 0.5 * (F.mse_loss(A_hat_s, A_s) + F.mse_loss(A_hat_t, A_t))

        # 3) alignment loss (requires P_current; compute P_current if missing via Sinkhorn)
        if model.coupling.P_current is None:
            # initial P via sinkhorn
            model.coupling.compute_P(z_s.detach(), z_t.detach())

        # AE is computed by coupling.forward which reads P_current
        loss_align = model.coupling(A_s, A_t, z_s.detach(), z_t.detach())

        # 4) Kendall weighting (ablation option may disable)
        if getattr(model, "ablation", None) == "no_kuw":
            total_loss = loss_cls + loss_recon + weight_align * loss_align
        else:
            precision_cls = torch.exp(-log_var_cls)
            precision_recon = torch.exp(-log_var_recon)
            total_loss = (
                precision_cls * loss_cls + log_var_cls +
                precision_recon * loss_recon + log_var_recon +
                weight_align * loss_align
            )
    


        # backward on W + log_vars
        total_loss.backward()
        optimizer.step()

        # Step B: update P (no grad for W)
        model.eval()
        with torch.no_grad():
            _, _, z_s = model(data_src.x.to(device), data_src.edge_index.to(device))
            _, _, z_t = model(data_tgt.x.to(device), data_tgt.edge_index.to(device))

            if p_update_method == "sink":
                # Sinkhorn: compute and store P_current
                P_new = model.coupling.compute_P(z_s, z_t)
                model.coupling.P_current = P_new

            elif p_update_method == "fw":
                # Frank-Wolfe / gradient step on P using analytical gradient of AE
                # Use current P_current as starting point; if None, initialize as uniform
                if model.coupling.P_current is None:
                    n = z_s.size(0)
                    P0 = torch.ones((n, n), device=device) / n
                    model.coupling.P_current = P0

                P = model.coupling.P_current.clone().to(device)
                # compute gradient of AE wrt P
                grad_P = ae_gradient_wrt_P(A_s, A_t, P)
                # update P
                P_new = frank_wolfe_update(P, grad_P, step_size=0.1)
                model.coupling.P_current = P_new

            else:
                raise ValueError("Unknown P update method; choose 'sink' or 'fw'")

            P_curr = model.coupling.P_current
            diff = A_s @ P_curr - P_curr @ A_t
            AE_val = torch.norm(diff, p='fro') ** 2
            AE_final = AE_val.item()

            # compute target accuracy using logits_t (from Step A)
            acc_tgt = accuracy(logits_t[data_tgt.test_mask].to(device), data_tgt.y[data_tgt.test_mask].to(device))
            acc_final = acc_tgt

        # logging and CS detection
        losses_log.append(total_loss.item())
        ae_log.append(AE_final)
        acc_log.append(acc_final)

        if cs_epoch is None and total_loss.item() <= cs_threshold:
            cs_epoch = epoch

        if epoch % log_every == 0 or epoch == 1:
            # log numeric info
            if getattr(model, "ablation", None) == "no_kuw":
                print(f"[Epoch {epoch:03d}] total_loss={total_loss.item():.6f} | Acc_tgt={acc_final:.4f} | AE={AE_final:.6f}")
            else:
                print(f"[Epoch {epoch:03d}] total_loss={total_loss.item():.6f} | Acc_tgt={acc_final:.4f} | AE={AE_final:.6f} | log_var_cls={log_var_cls.item():.3f}, log_var_recon={log_var_recon.item():.3f}")

        # early stopping on cross-domain similarity signal
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

    # optional: compare P matrices (computes both P_sink and P_fw heatmaps)
    if compare_P:
        # ensure directory
        os.makedirs(out_dir, exist_ok=True)
        prefix = os.path.join(out_dir, f"compare_{data_src.__class__.__name__}_{data_tgt.__class__.__name__}")
        # compute sink P
        with torch.no_grad():
            _, _, z_s = model(data_src.x.to(device), data_src.edge_index.to(device))
            _, _, z_t = model(data_tgt.x.to(device), data_tgt.edge_index.to(device))
            P_sink = model.coupling.compute_P(z_s, z_t)
            plot_P_heatmap(P_sink, "P via Sinkhorn", f"{prefix}_P_sink.png")

            # compute FW P (from same embeddings)
            # compute grad and update a few steps from uniform to get a representative P_fw
            n = z_s.size(0)
            P_fw = torch.ones((n, n), device=device) / n
            for _ in range(5):
                grad = ae_gradient_wrt_P(A_s.to(device), A_t.to(device), P_fw)
                P_fw = frank_wolfe_update(P_fw, grad, step_size=0.1)
            plot_P_heatmap(P_fw, "P via Frank-Wolfe", f"{prefix}_P_fw.png")

            # diff
            diff = (P_sink - P_fw).abs()
            plot_P_heatmap(diff, "|Sinkhorn - FW|", f"{prefix}_P_diff.png")

        print(f"[Compare_P] Saved {prefix}_P_sink.png, {prefix}_P_fw.png, {prefix}_P_diff.png")

    return acc_final or 0.0, cs_epoch or epochs, AE_final or float("nan")

def train_transfer_mmd(model, data_src, data_tgt, lr, epochs, device=None, mmd_lambda=1.0, cs_threshold=0.1):
    device = device or torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    A_s = to_dense_adj(data_src.edge_index)[0].to(device)
    A_t = to_dense_adj(data_tgt.edge_index)[0].to(device)
    AE = torch.norm(A_s - A_t, p='fro') ** 2

    cs_epoch = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        output_s = model(data_src.x, data_src.edge_index)
        output_t = model(data_tgt.x, data_tgt.edge_index)

        logits_s = output_s[0] if isinstance(output_s, tuple) else output_s
        z_s = output_s[-1] if isinstance(output_s, tuple) else None
        logits_t = output_t[0] if isinstance(output_t, tuple) else output_t
        z_t = output_t[-1] if isinstance(output_t, tuple) else None

        loss_cls = criterion(logits_s[data_src.train_mask], data_src.y[data_src.train_mask])
        loss_mmd = compute_mmd(z_s, z_t)
        total_loss = loss_cls + mmd_lambda * loss_mmd

        total_loss.backward()
        optimizer.step()

        if cs_epoch is None and total_loss.item() <= cs_threshold:
            cs_epoch = epoch

        if epoch % 10 == 0 or epoch == 1:
            acc_t = accuracy(logits_t[data_tgt.test_mask], data_tgt.y[data_tgt.test_mask])
            print(f"[MMD][Epoch {epoch}] loss={total_loss.item():.4f} Acc={acc_t:.4f} AE={AE.item():.4f}")

    acc_final = accuracy(logits_t[data_tgt.test_mask], data_tgt.y[data_tgt.test_mask])
    return acc_final, cs_epoch or epochs, AE.item()


def train_transfer_ce(model, data_src, data_tgt, lr, epochs, device=None, cs_threshold=0.1):
    device = device or torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    A_s = to_dense_adj(data_src.edge_index)[0].to(device)
    A_t = to_dense_adj(data_tgt.edge_index)[0].to(device)
    AE = torch.norm(A_s - A_t, p='fro') ** 2

    cs_epoch = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        output_s = model(data_src.x, data_src.edge_index)
        output_t = model(data_tgt.x, data_tgt.edge_index)

        logits_s = output_s if isinstance(output_s, torch.Tensor) else output_s[0]
        logits_t = output_t if isinstance(output_t, torch.Tensor) else output_t[0]

        total_loss = criterion(logits_s[data_src.train_mask], data_src.y[data_src.train_mask])

        total_loss.backward()
        optimizer.step()

        if cs_epoch is None and total_loss.item() <= cs_threshold:
            cs_epoch = epoch

        if epoch % 10 == 0 or epoch == 1:
            acc_t = accuracy(logits_t[data_tgt.test_mask], data_tgt.y[data_tgt.test_mask])
            print(f"[CE][Epoch {epoch}] loss={total_loss.item():.4f} Acc={acc_t:.4f} AE={AE.item():.4f}")

    acc_final = accuracy(logits_t[data_tgt.test_mask], data_tgt.y[data_tgt.test_mask])
    return acc_final, cs_epoch or epochs, AE.item()



# ----------------------------
# Main
# ----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Load dataset
    # -------------------------
    if args.mode == "single":
        data, = load_dataset(args.dataset, mode="single")
        data = data.to(device)
        n_nodes = data.num_nodes
    else:
        data_src, data_tgt = load_dataset(args.dataset, mode="transfer")
        data_src, data_tgt = data_src.to(device), data_tgt.to(device)
        n_nodes = data_src.num_nodes

    in_dim = (data.x.size(1) if args.mode == "single" else data_src.x.size(1))
    n_classes = int(((data.y.max() if args.mode == "single" else data_src.y.max()).item()) + 1)

    # -------------------------
    # Build model
    # -------------------------

    if args.model in ["gaae", "gt"]:

        encoder = GATEncoder(in_dim, hid_dim=16, out_dim=args.embed_dim)

        transformer = None
        decoder_type = "simple"

        if args.model == "gt":
            transformer = SimpleTransformerEncoder(
                dim=args.embed_dim,
                nhead=args.nhead,
                nlayers=args.gt_layers,
                dim_ff=args.dim_ff,
                dropout=args.dropout
            ).to(device)
            decoder_type = "gt"

        classifier = Classifier(args.embed_dim, n_classes)

        model = GTLGT(
            encoder,
            transformer,
            classifier,
            decoder_type=decoder_type,
            embed_dim=args.embed_dim,
            n_nodes=n_nodes,
            ablation=args.ablation
        ).to(device)

    elif args.model == "graphbridge":

        encoder = GATEncoder(in_dim, 16, args.embed_dim)
        model = GraphBridge(
            encoder=encoder,
            embed_dim=args.embed_dim,
            num_classes=n_classes
        ).to(device)

    elif args.model == "zerog":

        encoder = GATEncoder(in_dim, 16, args.embed_dim)
        model = ZeroG(
            encoder=encoder,
            embed_dim=args.embed_dim,
            num_classes=n_classes
        ).to(device)

    elif args.model == "graphlora":

        encoder = GATEncoder(in_dim, 16, args.embed_dim)
        model = GraphLoRA(
            encoder=encoder,
            embed_dim=args.embed_dim,
            num_classes=n_classes
        ).to(device)

    elif args.model == "crosstres":

        encoder = GATEncoder(in_dim, 16, args.embed_dim)
        model = CrossTREs(
            encoder=encoder,
            embed_dim=args.embed_dim,
            num_classes=n_classes
        ).to(device)

    elif args.model == "gnn":

        model = GCNNet(in_dim, hid_dim=64, out_dim=n_classes).to(device)

    else:
        raise ValueError("Unknown model")

    # -------------------------
    # Training
    # -------------------------
    if args.mode == "single":

        acc, cs = train_single(
            model, data,
            args.lr, args.epoch, args.early_stopping,
            cs_threshold=args.cs_threshold,
            device=device,
            log_every=args.log_every
        )

        print(f"Single Result — ACC={acc:.4f}, CS_epoch={cs}")

    else:

        # ================= 主方法 =================
        if args.model in ["gaae", "gt"]:

            acc_final, cs_epoch, AE_final = train_transfer_ot(
                model, data_src, data_tgt,
                lr=args.lr,
                epochs=args.epoch,
                early_stopping=args.early_stopping,
                cs_threshold=args.cs_threshold,
                device=device,
                weight_align=args.w_align,
                p_update_method=args.P_update,
                log_every=args.log_every,
                compare_P=args.compare_P,
                out_dir=args.out_dir
            )

        # ================= MMD baselines =================
        elif args.model in ["graphbridge", "graphlora", "crosstres"]:

            acc_final, cs_epoch, AE_final = train_transfer_mmd(
                model, data_src, data_tgt,
                args.lr, args.epoch,
                device=device,
                mmd_lambda=args.mmd_lambda,
                cs_threshold=args.cs_threshold
            )

        # ================= CE baselines =================
        elif args.model in ["zerog", "gnn"]:

            acc_final, cs_epoch, AE_final = train_transfer_ce(
                model, data_src, data_tgt,
                args.lr, args.epoch,
                device=device,
                cs_threshold=args.cs_threshold
            )

        else:
            raise ValueError("Unknown model")

        # ============ 统一输出 ============
        print("\n====== FINAL RESULT ======")
        print(f"Target Accuracy = {acc_final:.4f}")
        print(f"CS (Convergence Epoch) = {cs_epoch}")
        print(f"AE (||A_s P - P A_t||_F^2) = {AE_final:.6f}")
        print("=====================================")




# ----------------------------
# Argument parser
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="transfer", choices=["single", "transfer"])
    parser.add_argument("--model", type=str, default="gt", choices=["gnn", "gaae", "gt",'graphbridge','zerog',"graphlora","crosstres"])
    parser.add_argument("--dataset", type=str, default="p1,p1", help="src,tgt for transfer")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--early_stopping", type=int, default=50)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--cs_threshold", type=float, default=0.1)
    parser.add_argument("--w_align", type=float, default=1.0, help="Fixed weight for OT alignment loss")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["no_gt", "no_gaae", "no_kuw", "full"],
                        help="Ablation mode")
    parser.add_argument("--P_update", type=str, default="sink", choices=["sink", "fw"],
                        help="Update method for P: sink (Sinkhorn) or fw (Frank-Wolfe)")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Print log every N epochs (default=10). Use 1 for logging every epoch.")
    parser.add_argument("--compare_P", action="store_true",
                        help="Compare Sinkhorn vs FW P matrices and save heatmaps at end.")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Directory to save P heatmaps when --compare_P is used.")
    parser.add_argument('--gt_layers', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dim_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--mmd_lambda', type=float, default=1.0)

    args = parser.parse_args()
    main(args)
