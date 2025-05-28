"""
model.py，已經修改成rope position encoding了
"""

import math
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# -----------------------------------------------------------------------------
def build_rope(seq_len, dim, device="gpu"):
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    angle = pos * freq  # shape: (seq_len, dim/2)
    return torch.cos(angle), torch.sin(angle)

def apply_rope(x, cos, sin):
    # x: (batch, seq_len, dim)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# -----------------------------------------------------------------------------
# RoPEEncoderLayer：自定義 Encoder Layer with RoPE Attention
# -----------------------------------------------------------------------------
class RoPEEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch, seq_len, d_model)
        B, S, D = src.size()
        cos, sin = build_rope(S, D, src.device)
        cos = cos.unsqueeze(0)  # shape: (1, seq_len, dim/2)
        sin = sin.unsqueeze(0)
        src_qk = apply_rope(src, cos, sin)

        attn_output, _ = self.self_attn(src_qk, src_qk, src)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src

# -----------------------------------------------------------------------------
# TransformerClassifier：使用 RoPE 編碼的分類模型
# -----------------------------------------------------------------------------
class TransformerClassifier(nn.Module):
    """
    Transformer Encoder classifier with:
      - RoPE rotary position embedding
      - avg-pooling + dropout + final linear layer
    Args:
      feature_dim:  原始特徵維度
      seq_len:      序列長度 (group_size)
      d_model:      Transformer 隱藏維度
      nhead:        multi-head attention 頭數
      num_layers:   Transformer encoder 層數
      dropout:      dropout 比例
      num_classes:  分類類別數
    Input x shape: (batch, seq_len, feature_dim)
    Output logits: (batch, num_classes)
    """
    def __init__(
        self,
        feature_dim: int,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()

        # 1. Embedding：把原始 feature 映射到 d_model 維
        self.embedding = nn.Linear(feature_dim, d_model)

        # 2. RoPE-based Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList([
            RoPEEncoderLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])

        # 3. 池化 + Dropout + 全連接輸出
        self.pool    = nn.AdaptiveAvgPool1d(1)  # over seq_len dimension
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim)
        returns: logits (batch, num_classes)
        """
        # (1) Embedding → (batch, seq_len, d_model)
        h = self.embedding(x)

        # (2) RoPE Transformer Encoder
        for layer in self.encoder_layers:
            h = layer(h)

        # (3) Avg-pooling: 先轉 (batch, d_model, seq_len)
        h = h.transpose(1, 2)
        h = self.pool(h).squeeze(-1)  # → (batch, d_model)

        # (4) Dropout + FC → logits
        h = self.dropout(h)
        logits = self.fc(h)           # → (batch, num_classes)
        return logits

"""
GenderCV_raw.py → 改為 LevelCV_raw.py：預測 level（4類）分類任務，使用 micro ROC AUC
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback

# ------------------ 模型引入 ------------------
#from model import TransformerClassifier

# ------------------ 1. 載入 txt 感測資料 ------------------
ALL_COLUMNS = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
selected_columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
max_len = 1651

def load_raw_dataset(txt_dir: str, info_csv_path: str, selected_columns, max_len: int, label_name="level"):
    txt_dir = Path(txt_dir)
    info = pd.read_csv(info_csv_path)

    # --- 特別處理：level 分類標籤重新編碼（從 2,3,4,5 → 0,1,2,3） ---
    if label_name == "level":
        info = info[info[label_name].isin([2,3,4,5])].copy()
        info[label_name] = info[label_name].map({2: 0, 3: 1, 4: 2, 5: 3})

    uid_to_label = dict(zip(info["unique_id"], info[label_name]))
    label_map = {v: v for v in sorted(set(uid_to_label.values()))}  # identity map

    col_idx = [ALL_COLUMNS.index(c) for c in selected_columns]
    X_list, y_list = [], []
    for file in sorted(txt_dir.glob("*.txt"), key=lambda f: int(f.stem)):
        uid = int(file.stem)
        if uid not in uid_to_label:
            continue
        try:
            arr = np.loadtxt(file)
        except:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = arr[:, col_idx]
        if arr.shape[0] < max_len:
            pad_len = max_len - arr.shape[0]
            arr = np.pad(arr, ((0, pad_len), (0, 0)), mode='constant')
        else:
            arr = arr[:max_len]
        X_list.append(torch.tensor(arr, dtype=torch.float32))
        y_list.append(torch.tensor(uid_to_label[uid], dtype=torch.long))

    X = torch.stack(X_list)
    y = torch.stack(y_list)
    return X.numpy(), y.numpy(), label_map

# 載入資料（label 改為 level）
X, y, label_map = load_raw_dataset(
    txt_dir='39_Training_Dataset/train_data',
    info_csv_path='39_Training_Dataset/train_info.csv',
    selected_columns=selected_columns,
    max_len=max_len,
    label_name='level'
)

# ------------------ 2. 模型設定 ------------------
FEATURE_D = X.shape[2]
SEQ_LEN   = X.shape[1]
NUM_CLASSES = len(label_map)

# ------------------ 3. 初始化 callback（He 初始化） ------------------
def kaiming_init(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class KaimingInitializer(Callback):
    def on_train_begin(self, net, **kwargs):
        net.module_.apply(kaiming_init)

# ------------------ 4. Skorch 模型包裝 ------------------
net = NeuralNetClassifier(
    module=TransformerClassifier,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    max_epochs=20,
    lr=1e-3,
    batch_size=32,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    callbacks=[KaimingInitializer()],
    iterator_train__shuffle=True,

    # 傳給 module 的參數
    module__feature_dim = FEATURE_D,
    module__seq_len     = SEQ_LEN,
    module__d_model     = 64,
    module__nhead       = 4,
    module__num_layers  = 2,
    module__dropout     = 0.1,
    module__num_classes = NUM_CLASSES
)

# ------------------ 5. GridSearchCV 超參數搜尋 ------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    'batch_size':        [128],
    'module__d_model':   [128, 256],
    'module__nhead':     [1, 2],
    'module__num_layers':[1, 2],
    'lr':                [1e-3],
    'module__dropout':   [0.0, 0.1, 0.3]
}

gs = GridSearchCV(
    net,
    param_grid=param_grid,
    scoring='roc_auc_ovr',  # 適用於 multi-class（one-vs-rest）
    cv=cv,
    verbose=1
)

print("[INFO] Starting GridSearchCV for LEVEL classification...")
gs.fit(X, y)

# ------------------ 6. 儲存結果 ------------------
Path("CVRecord/Level").mkdir(parents=True, exist_ok=True)
pd.DataFrame(gs.cv_results_).to_csv('CVRecord/Level/TEncoder+RawTxt.csv', index=True)

print("=== Level CV Done ===")
print("Best params:", gs.best_params_)
print("Best AUC:",    gs.best_score_)
