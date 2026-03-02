# ================================================================
# dataload.py
# 离线加载所有 npz 格式图数据 (真实 + 合成)
# 支持单图学习 (single) 与迁移学习 (transfer)
# ================================================================
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from scipy import sparse

# ------------------------------------------------------------
# 自动定位 data 文件夹路径
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.path.join(BASE_DIR, "data")

# ------------------------------------------------------------
# 创建 train/val/test 掩码
# ------------------------------------------------------------
def make_masks(data, train_ratio=0.6, val_ratio=0.2):
    n = data.num_nodes
    idx = np.arange(n)
    np.random.shuffle(idx)
    tr = torch.zeros(n, dtype=torch.bool)
    val = torch.zeros(n, dtype=torch.bool)
    te = torch.zeros(n, dtype=torch.bool)
    tr[idx[:int(train_ratio * n)]] = True
    val[idx[int(train_ratio * n):int((train_ratio + val_ratio) * n)]] = True
    te[idx[int((train_ratio + val_ratio) * n):]] = True
    data.train_mask, data.val_mask, data.test_mask = tr, val, te
    return data

# ------------------------------------------------------------
# 通用 npz 文件读取函数
# ------------------------------------------------------------
def load_npz_graph(path):
    """从 .npz 文件加载图为 PyG Data 对象（兼容 CSR 格式）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 文件不存在: {path}")

    data_npz = np.load(path, allow_pickle=True)

    # 节点特征
    if 'x' in data_npz:
        x = data_npz['x']
    elif 'features' in data_npz:
        x = data_npz['features']
    else:
        raise KeyError(f"{path} 中未找到特征字段 x/features")

    # 标签
    if 'y' in data_npz:
        y = data_npz['y']
    elif 'labels' in data_npz:
        y = data_npz['labels']
    else:
        raise KeyError(f"{path} 中未找到标签字段 y/labels")

    # ✅ 新增：读取CSR稀疏邻接矩阵格式
    if all(k in data_npz for k in ['adj_data', 'adj_indices', 'adj_indptr', 'adj_shape']):
        A = sparse.csr_matrix(
            (data_npz['adj_data'], data_npz['adj_indices'], data_npz['adj_indptr']),
            shape=data_npz['adj_shape']
        )
        edge_index = np.vstack(A.nonzero())
    elif 'edge_index' in data_npz:
        edge_index = data_npz['edge_index']
    elif 'adj' in data_npz:
        adj = data_npz['adj']
        if adj.ndim != 2:
            raise ValueError(f"{path} 中 adj 维度异常: {adj.shape}")
        edge_index = np.vstack(np.nonzero(adj))
    else:
        raise KeyError(f"{path} 中未找到 edge_index/adj/adj_data 等字段")

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long)
    )

    # 如果有 train_mask 就加载
    if 'train_mask' in data_npz:
        mask = data_npz['train_mask']
        if mask.size > 0:
            data.train_mask = torch.tensor(mask, dtype=torch.bool)

    return data
# ------------------------------------------------------------
# 真实数据加载
# ------------------------------------------------------------
def load_real_dataset(name: str, root: str = DEFAULT_DATA_ROOT):
    """
    从 data/<name>/processed/<name>.npz 读取真实数据集
    """
    name = name.lower()
    file_path = os.path.join(root, name, "processed", f"{name}.npz")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 未找到真实数据文件: {file_path}")

    data = load_npz_graph(file_path)
    data = make_masks(data)
    print(f"✅ Loaded real dataset: {name} | nodes={data.num_nodes}, edges={data.num_edges}")
    return data

# ------------------------------------------------------------
# 合成数据加载
# ------------------------------------------------------------
def load_synthetic_dataset(name: str, root: str = DEFAULT_DATA_ROOT):
    """
    从 data/p1/p1_A.npz, p1_B.npz 等读取合成图对
    """
    name = name.lower()
    dir_path = os.path.join(root, name)
    pathA = os.path.join(dir_path, f"{name}_A.npz")
    pathB = os.path.join(dir_path, f"{name}_B.npz")

    if not (os.path.exists(pathA) and os.path.exists(pathB)):
        pathA = os.path.join(dir_path, f"{name.upper()}_A.npz")
        pathB = os.path.join(dir_path, f"{name.upper()}_B.npz")

    if not (os.path.exists(pathA) and os.path.exists(pathB)):
        raise FileNotFoundError(f"❌ 未找到合成图 {name.upper()} 文件: {pathA}, {pathB}")

    G_A = load_npz_graph(pathA)
    G_B = load_npz_graph(pathB)
    G_A = make_masks(G_A)
    G_B = make_masks(G_B)
    print(f"✅ Loaded synthetic pair: {name.upper()} | A({G_A.num_nodes}), B({G_B.num_nodes})")
    return G_A, G_B

# ------------------------------------------------------------
# 主统一接口
# ------------------------------------------------------------
def load_dataset(name: str, mode: str = "single", root: str = DEFAULT_DATA_ROOT):
    """
    mode = 'single'   -> 返回单个图 (data,)
    mode = 'transfer' -> 返回一对图 (data_A, data_B)
    """
    name = name.lower()

    # 单图模式
    if mode == 'single':
        if name in ['cora', 'citeseer', 'pubmed']:
            data = load_real_dataset(name, root)
            return data,
        elif name in ['p1', 'p2', 'p3']:
            data_A, _ = load_synthetic_dataset(name, root)
            return data_A,
        else:
            raise ValueError(f"未知数据集名称: {name}")

    # 迁移模式
    elif mode == 'transfer':
        if ',' not in name:
            raise ValueError("迁移模式需指定格式: 'src,tgt'")
        src, tgt = [s.strip() for s in name.split(',')]

        if src in ['cora', 'citeseer', 'pubmed']:
            data_A = load_real_dataset(src, root)
        else:
            data_A, _ = load_synthetic_dataset(src, root)

        if tgt in ['cora', 'citeseer', 'pubmed']:
            data_B = load_real_dataset(tgt, root)
        else:
            _, data_B = load_synthetic_dataset(tgt, root)

        return data_A, data_B

    else:
        raise ValueError("mode 必须是 'single' 或 'transfer'")
