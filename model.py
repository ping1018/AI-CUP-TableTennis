import math
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# PositionalEncoding：可訓練的位置編碼
# 初始化使用標準 sinusoidal 公式，再註冊為 nn.Parameter，讓模型可隨梯度更新
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Learnable sinusoidal positional encoding.
    Args:
      d_model:  編碼維度 (一致於 Transformer 隱藏維度)
      max_len:  可支援的最大序列長度
    Input x shape: (batch, seq_len, d_model)
    Output    shape: (batch, seq_len, d_model)
    """
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        # 1. 先用 sinusoidal 公式初始化
        pe = torch.zeros(max_len, d_model)                              # (max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # -> (1, max_len, d_model)

        # 2. 註冊為可訓練參數
        self.pe = nn.Parameter(pe, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # 直接相加位置編碼
        return x + self.pe[:, :x.size(1), :]


# -----------------------------------------------------------------------------
# TransformerClassifier：Encoder 分類器
# -----------------------------------------------------------------------------
class TransformerClassifier(nn.Module):
    """
    Transformer Encoder classifier with:
      - learnable positional encoding
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

        # 2. Learnable positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,           # 使用 (batch, seq_len, d_model) 輸入
            dim_feedforward=d_model * 4,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4. 池化 + Dropout + 全連接輸出
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

        # (2) Add positional encoding
        h = self.pos_encoder(h)

        # (3) Transformer Encoder
        h = self.transformer_encoder(h)

        # (4) Avg-pooling: 先轉 (batch, d_model, seq_len)
        h = h.transpose(1, 2)
        h = self.pool(h).squeeze(-1)  # → (batch, d_model)

        # (5) Dropout + FC → logits
        h = self.dropout(h)
        logits = self.fc(h)           # → (batch, num_classes)
        return logits
