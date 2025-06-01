# model2.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwingCNN(nn.Module):
    """
    動態深度 1D‐CNN，並可自訂 Dropout 比例：
      - 輸入： (batch, 27, feature_dim)
      - num_layers 決定堆疊幾層 Conv1d+BN+ReLU+MaxPool1d
      - 每層的 out_channels = in_channels * 2
      - 卷積堆疊完後做 AdaptiveMaxPool1d → 再兩層全連接 → logits
      - dropout_rate 可傳入控制 fc1 後的 Dropout 比例
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout_rate: float = 0.5
    ):
        """
        Args:
          feature_dim: 輸入通道數 D（也就是原本的特徵維度）
          num_classes: 分類類別數 C
          num_layers:  要疊幾層 Conv+Pool（預設 2）
          dropout_rate: FC1 後面那層的 Dropout 比例（預設 0.5）
        """
        super(SwingCNN, self).__init__()
        self.num_layers = num_layers

        # 1. 動態建立 Conv1d→BN→ReLU→MaxPool1d 堆疊區段
        self.conv_blocks = nn.ModuleList()
        in_ch = feature_dim
        for i in range(num_layers):
            out_ch = in_ch * 2
            self.conv_blocks.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    padding=1  # same padding
                )
            )
            self.conv_blocks.append(nn.BatchNorm1d(out_ch))
            self.conv_blocks.append(nn.ReLU(inplace=True))
            self.conv_blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_ch = out_ch  # 下一層的輸入通道

        # 2. 全域最大池化：把時間維從 T_reduced 壓到 1
        self.global_pool = nn.AdaptiveMaxPool1d(output_size=1)

        # 3. 全連接層（第一段）：in_features = in_ch = feature_dim * (2^num_layers)
        self.fc1 = nn.Linear(in_ch, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu_fc = nn.ReLU(inplace=True)

        # 4. 最後輸出層： (128 → num_classes)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 27, feature_dim)
        1. permute → (batch, feature_dim, 27)
        2. 依序通過 num_layers 個 Conv1d+BN+ReLU+MaxPool1d
        3. global_pool → (batch, in_ch, 1)
        4. squeeze → (batch, in_ch)
        5. fc1 → dropout → ReLU → (batch, 128)
        6. fc_out → logits → (batch, num_classes)
        """
        # 1. 先把原本 (batch, 27, D) → (batch, D, 27)
        x = x.permute(0, 2, 1)

        # 2. 動態堆疊的卷積區段
        for layer in self.conv_blocks:
            x = layer(x)

        # 3. 全域最大池化: (batch, in_ch, T_reduced) → (batch, in_ch, 1)
        x = self.global_pool(x)

        # 4. 去掉最後的時間維度: (batch, in_ch, 1) → (batch, in_ch)
        x = x.view(x.size(0), -1)

        # 5. 全連接段
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu_fc(x)

        # 6. 輸出 logits
        logits = self.fc_out(x)
        return logits


if __name__ == '__main__':
    # 範例測試
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 例如：feature_dim=34, num_classes=5, 要疊 3 層卷積, dropout=0.3
    model = SwingCNN(feature_dim=34, num_classes=5, num_layers=3, dropout_rate=0.3).to(device)

    dummy_input = torch.randn(2, 27, 34).to(device)   # (batch=2, time=27, feat=34)
    dummy_output = model(dummy_input)

    print("Dummy input shape :", dummy_input.shape)    # 預期 (2, 27, 34)
    print("Dummy output shape:", dummy_output.shape)   # 預期 (2, 5)
    print("Logits example:\n", dummy_output)
