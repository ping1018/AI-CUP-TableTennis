# train_cnn_hold.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import Accuracy

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 直接從你已有的 model2.py 裡匯入 SwingCNN
# -----------------------------------------------------------------------------
from model2 import SwingCNN

# 如果 model2.py 裡還沒定義 init_weights_kaiming，就在這裡補上
def init_weights_kaiming(m):
    """
    對 Conv1d 和 Linear 層使用 Kaiming 正態初始化 (He normal initialization)：
      - weight: kaiming_normal_
      - bias: zeros_
    可用：model.apply(init_weights_kaiming)
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# -----------------------------------------------------------------------------
# 載入 data_prepare.py 中的 load_and_preprocess 函式
# -----------------------------------------------------------------------------
from data_prepare import load_and_preprocess

# =========================
# 1. 參數設定 (configurations)
# =========================

DATAPATH    = './tabular_data_train'                  # 特徵 CSV 資料夾
INFO_PATH   = '39_Training_Dataset/train_info.csv'     # train_info.csv 路徑
GROUP_SIZE  = 27                                       # 每個樣本 (揮拍段) 固定長度 27
TEST_SIZE   = 0.2                                      # 測試集比例
RANDOM_SEED = 42                                       # 隨機種子
BATCH_SIZE  = 128                                        # Lightning DataLoader 的 batch size
LR          = 1e-3                                     # 學習率 (learning rate)
MAX_EPOCHS  = 20                                       # 最大訓練 Epoch
NUM_CLASSES = 2                                        # hold racket handed 二元分類 => 2 類

# =========================
# 2. 資料前處理：呼叫 load_and_preprocess()
# =========================

data_dict = load_and_preprocess(
    datapath     = DATAPATH,
    info_path    = INFO_PATH,
    group_size   = GROUP_SIZE,
    test_size    = TEST_SIZE,
    random_state = RANDOM_SEED
)

# -----------------------------------------------------------------------------
# 取出 X / y，這裡以 'hold racket handed' 作為 target (二元分類)
# -----------------------------------------------------------------------------

X_train = data_dict['X_train']       # shape = (N_train_segments, 27, feature_dim)
X_test  = data_dict['X_test']        # shape = (N_test_segments, 27, feature_dim)

y_train_hold = data_dict['y_hold']       # shape = (N_train_segments,)
y_test_hold  = data_dict['y_hold_te']    # shape = (N_test_segments,)

FEATURE_DIM = X_train.shape[2]       # 由 data_prepare.py 自動推算 (例如 34)

# 如果要改其他目標 (舉例):
#   y_train_target = data_dict['y_level']      # 多類別 (level) 分類
#   y_test_target  = data_dict['y_level_te']
#   NUM_CLASSES = len(data_dict['le_level'].classes_)  # 改成對應類別數

# =========================
# 3. LightningModule 定義 (改用 CNN)
# =========================

class LitCNNClassifier(pl.LightningModule):
    """
    PyTorch Lightning Module for CNN 二元分類 (hold racket handed)。
    內部使用 SwingCNN，並且在初始化時套用 Kaiming 初始化。
    """

    def __init__(self, feature_dim: int, num_classes: int, lr: float = 1e-3):
        super(LitCNNClassifier, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        # 使用 SwingCNN 作為主體
        self.model = SwingCNN(feature_dim=feature_dim, num_classes=num_classes)
        # 套用 Kaiming 初始化
        self.model.apply(init_weights_kaiming)

        # 損失函數 (CrossEntropyLoss) + 分類精度 (Accuracy)
        self.criterion = nn.CrossEntropyLoss()
        # task="multiclass" 雖然是 binary，但是也可用 multiclass
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 27, feature_dim)
        回傳 logits: (batch_size, num_classes)
        """
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)                        # (batch, num_classes)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc",  self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc",  self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# =========================
# 4. 建立 Dataset / DataLoader
# =========================

train_dataset = TensorDataset(X_train,       y_train_hold)
val_dataset   = TensorDataset(X_test,        y_test_hold)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

# =========================
# 5. Callback：畫訓練/驗證 Accuracy 曲線
# =========================

class PlotMetricsCallback(pl.Callback):
    """
    Lightning Callback：在每個 epoch 結束時計錄 train_acc / val_acc，
    並於訓練結束後繪製 Accuracy 曲線。
    """

    def __init__(self):
        super().__init__()
        self.train_accs = []
        self.val_accs   = []

    def on_train_epoch_end(self, trainer, pl_module):
        acc = trainer.callback_metrics.get("train_acc")
        if acc is not None:
            self.train_accs.append(acc.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        acc = trainer.callback_metrics.get("val_acc")
        if acc is not None:
            self.val_accs.append(acc.item())

    def on_train_end(self, trainer, pl_module):
        num_epochs = min(len(self.train_accs), len(self.val_accs))
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.train_accs[:num_epochs], label="Train Acc", color="green")
        plt.plot(epochs, self.val_accs[:num_epochs],   label="Val Acc",   color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


# =========================
# 6. 設定 Checkpoint
# =========================

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",                                 # 監控的指標 (validation accuracy)
    dirpath="cnn_hold_checkpoints",                    # 儲存路徑 (改成 hold 相關)
    filename="cnn-hold-{epoch:02d}-{val_acc:.2f}",     # 檔名格式
    save_top_k=1,                                      # 只保留最好的 1 個
    mode="max"                                         # val_acc 越大越好
)

# =========================
# 7. 執行訓練
# =========================

lit_model = LitCNNClassifier(
    feature_dim=FEATURE_DIM,
    num_classes=NUM_CLASSES,
    lr=LR
)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",      # 自動選 GPU / CPU
    callbacks=[PlotMetricsCallback(), checkpoint_callback],
    logger=False             # 不需要 TensorBoard logger 時可關閉
)

trainer.fit(lit_model, train_loader, val_loader)

# 訓練完成後，存下最佳模型權重 (純 PyTorch state_dict)
best_ckpt_path = checkpoint_callback.best_model_path
print(f"Best checkpoint saved at: {best_ckpt_path}")

best_model = SwingCNN(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES)
state = torch.load(best_ckpt_path)["state_dict"]
# Lightning checkpoint 的 key 會是 "model.<layer_name>..."，用 strict=False 載入即可
best_model.load_state_dict(state, strict=False)
torch.save(best_model.state_dict(), "cnn_hold_model.pth")
print("Saved PyTorch weights to cnn_hold_model.pth")

# =========================
# 8. 繪製 ROC Curve (可選)
# =========================

from sklearn.metrics import roc_curve, auc

def plot_roc(model: nn.Module, dataloader: DataLoader, device="cpu"):
    """
    繪製 ROC Curve:
      - model 輸出 raw logits (batch, 2)
      - dataloader: 驗證資料載入器
    """
    model.eval()
    model.to(device)

    all_probs   = []  # 取 class=1 的機率
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)                        # (batch, 2)
            probs  = F.softmax(logits, dim=-1)[:, 1]  # class=1 的機率
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--", color="gray")
    plt.xlabel("False Positive Rate");   plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right");   plt.grid(alpha=0.3);   plt.show()

    return roc_auc

# 如果想畫 ROC:
device = "cuda" if torch.cuda.is_available() else "cpu"
best_model = SwingCNN(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES).to(device)
best_model.load_state_dict(torch.load("cnn_hold_model.pth"))

roc_auc_value = plot_roc(best_model, DataLoader(val_dataset, batch_size=BATCH_SIZE), device)
print("ROC AUC :", roc_auc_value)
