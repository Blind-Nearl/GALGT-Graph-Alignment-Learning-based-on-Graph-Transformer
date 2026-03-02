"""
model.py  (Stable OT Version)
-----------------------------
Implement:
 - GCNNet
 - GATEncoder
 - SimpleTransformerEncoder
 - AdjacencyDecoder
 - GraphTransformerDecoder
 - Classifier
 - OTCoupling (Sinkhorn Optimal Transport)
 - GTLGT (full model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ot       # POT library: pip install pot
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_adj

# ==============================================================
# Baseline GCN
# ==============================================================
class GCNNet(nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=128, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ==============================================================
# GAT Encoder
# ==============================================================
class GATEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim=32, out_dim=128, heads=4, dropout=0.5):
        super().__init__()
        self.gat1 = GATConv(in_dim, hid_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hid_dim * heads, out_dim, heads=1, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x


# ==============================================================
# Simple Decoder (inner product)
# ==============================================================
class AdjacencyDecoder(nn.Module):
    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize

    def forward(self, z):
        if self.normalize:
            z = F.normalize(z, p=2, dim=-1)
        A_logits = z @ z.t()
        return torch.sigmoid(A_logits)


# ==============================================================
# Graph Transformer Decoder
# ==============================================================
class GraphTransformerDecoder(nn.Module):
    def __init__(self, dim, nhead=4, nlayers=1, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.W_d = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, 1)

    def forward(self, z):
        h = self.decoder(z.unsqueeze(0)).squeeze(0)
        zi = h.unsqueeze(1).expand(-1, h.size(0), -1)
        zj = h.unsqueeze(0).expand(h.size(0), -1, -1)
        rel = torch.tanh(self.W_d(zi - zj))
        A_logits = self.out_proj(rel).squeeze(-1)
        return torch.sigmoid(A_logits)


# ==============================================================
# Transformer Encoder
# ==============================================================
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, dim=128, nhead=4, nlayers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)

    def forward(self, z):
        return self.encoder(z.unsqueeze(0)).squeeze(0)


# ==============================================================
# Classifier
# ==============================================================
class Classifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, z):
        return self.fc(z)


# ==============================================================
# Stable OT (Sinkhorn) Coupling Layer
# ==============================================================
# ==============================================================
# Stable OT (Sinkhorn) Coupling Layer —— FIXED VERSION
# ==============================================================
class OTCoupling(nn.Module):
    """
    Sinkhorn Optimal Transport with convergence safeguards.
    """
    def __init__(self, n_nodes, eps=0.1, max_iter=5000, tol=1e-5):
        super().__init__()
        self.n = n_nodes
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol

        # uniform marginals
        self.register_buffer('mu', torch.full((n_nodes,), 1.0 / n_nodes))
        self.register_buffer('nu', torch.full((n_nodes,), 1.0 / n_nodes))

        self.P_current = None

    @torch.no_grad()
    def compute_P(self, z_s, z_t):
        """
        Compute OT matrix using stabilized Sinkhorn with fallback.
        """
        device = z_s.device
        C = torch.cdist(z_s, z_t, p=2) ** 2  # [n, n]
        C_np = C.detach().cpu().numpy()
        mu_np = self.mu.cpu().numpy()
        nu_np = self.nu.cpu().numpy()

        try:
            # First attempt: stabilized Sinkhorn with high iter
            P = ot.sinkhorn(
                mu_np, nu_np, C_np,
                reg=self.eps,
                numItermax=self.max_iter,
                stopThr=self.tol,
                method='sinkhorn_stabilized',
                verbose=False
            )
        except Exception as e:
            print(f"[OT Warning] Stabilized Sinkhorn failed: {e}. Trying default...")
            try:
                P = ot.sinkhorn(
                    mu_np, nu_np, C_np,
                    reg=self.eps,
                    numItermax=self.max_iter,
                    stopThr=self.tol,
                    verbose=False
                )
            except Exception as e2:
                print(f"[OT Error] All Sinkhorn attempts failed: {e2}. Using uniform coupling.")
                P = np.ones_like(C_np) / (C_np.shape[0] * C_np.shape[1])

        P = torch.tensor(P, dtype=torch.float32, device=device)
        self.P_current = P
        return P

    def forward(self, A_s, A_t, z_s, z_t):
        assert self.P_current is not None, "Call compute_P before forward!"
        P = self.P_current
        diff = A_s @ P - P @ A_t
        AE = torch.norm(diff, p='fro') ** 2
        return AE


# ==============================================================
# GTLGT Model
# ==============================================================
class GTLGT(nn.Module):
    def __init__(self, encoder, transformer, classifier,
                 decoder_type="simple",
                 embed_dim=128,
                 n_nodes=None,
                 ablation="full"):

        super().__init__()
        self.ablation = ablation
        self.encoder = encoder
        self.transformer = transformer
        self.classifier = classifier

        # OT-based coupling
        self.coupling = OTCoupling(n_nodes=n_nodes,eps=0.2, max_iter=5000, tol=1e-4)

        if decoder_type == "gt":
            self.decoder = GraphTransformerDecoder(dim=embed_dim)
        elif decoder_type == "simple":
            self.decoder = AdjacencyDecoder()
        elif decoder_type == "none":
            self.decoder = None  # for no_gaae
        else:
            raise ValueError("Unknown decoder type")

    def forward(self, x, edge_index):
        z_raw = self.encoder(x, edge_index)

        if self.transformer is not None:
            z_final = self.transformer(z_raw)
        else:
            z_final = z_raw  # no_gt

        logits = self.classifier(z_final)
        return logits, z_raw, z_final


    def reconstruct_adj(self, z):
        if self.decoder is None:   # no_gaae
            return None
        return self.decoder(z)

# ============================
# GraphBridge Baseline
# ============================

class GraphBridge(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes):
        super(GraphBridge, self).__init__()

        self.encoder = encoder

        self.bridge = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        z_bridge = self.bridge(z)
        logits = self.classifier(z_bridge)
        return logits, z, z_bridge


# ============================
# MMD Alignment Loss
# ============================

def compute_mmd(x, y, sigma=1.0):

    def rbf(a, b):
        dist = torch.cdist(a, b, p=2)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))

    Kxx = rbf(x, x)
    Kyy = rbf(y, y)
    Kxy = rbf(x, y)

    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


class ZeroG(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        logits = self.classifier(z)
        return logits, z
    
class LoRALayer(nn.Module):
    def __init__(self, dim, rank=4):
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)

    def forward(self, x):
        return self.B(self.A(x))


class GraphLoRA(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes, rank=4):
        super().__init__()
        self.encoder = encoder

        # 冻结 encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.lora = LoRALayer(embed_dim, rank=rank)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        z = z + self.lora(z)
        logits = self.classifier(z)
        return logits, z
    
class CrossTREs(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        logits = self.classifier(z)
        return logits, z

    def structure_loss(self, edge_index_s, edge_index_t):
        A_s = to_dense_adj(edge_index_s)[0]
        A_t = to_dense_adj(edge_index_t)[0]
        return F.mse_loss(A_s, A_t)