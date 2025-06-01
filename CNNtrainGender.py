# CNNtrainGender.py

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
    可以透過 model.apply(init_weights_kaiming) 套用。
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

DATAPATH    = './tabular_data_train'                   # 特徵 CSV 資料夾
INFO_PATH   = '39_Training_Dataset/train_info.csv'      # train_info.csv 路徑
GROUP_SIZE  = 27                                        # 每個樣本(揮拍段)固定長度 27
TEST_SIZE   = 0.2                                       # 測試集比例
RANDOM_SEED = 42                                        # 隨機種子
BATCH_SIZE  = 128                                      # Lightning DataLoader 的 batch size
LR          = 1e-3                                      # 學習率 (learning rate)
MAX_EPOCHS  = 20                                        # 最大訓練 Epoch

# =========================
# 2. LightningModule 定義 (針對 gender 二元分類)
# =========================

class LitCNNGenderClassifier(pl.LightningModule):
    """
    用 SwingCNN 做二分類 (gender)。
    """

    def __init__(self, feature_dim: int, num_classes: int, lr: float = 1e-3):
        super(LitCNNGenderClassifier, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = SwingCNN(feature_dim=feature_dim, num_classes=num_classes)
        self.model.apply(init_weights_kaiming)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
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
# 3. Callback：蒐集並繪製訓練/驗證 Accuracy 和 Loss 曲線
# =========================

class PlotMetricsCallback(pl.Callback):
    """
    Lightning Callback：在每個 epoch 結束時計錄 train_acc / val_acc 和 train_loss / val_loss，
    並於訓練結束後繪製 Accuracy (準確率) 與 Loss (損失) 曲線。
    注意：因為 Lightning 在「sanity check」階段會先執行一次 validation，
    導致 val_accs 或 val_losses 可能多出一筆，因此在繪圖時要取 min(train_len, val_len)。
    """

    def __init__(self):
        super().__init__()
        self.train_accs    = []
        self.val_accs      = []
        self.train_losses  = []
        self.val_losses    = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_acc = trainer.callback_metrics.get("train_acc")
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_acc is not None:
            self.train_accs.append(train_acc.item())
        if train_loss is not None:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        val_acc = trainer.callback_metrics.get("val_acc")
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_acc is not None:
            self.val_accs.append(val_acc.item())
        if val_loss is not None:
            self.val_losses.append(val_loss.item())

    def on_train_end(self, trainer, pl_module):
        # 取最小長度，避免 sanity check 造成長度不一致
        n = min(len(self.train_accs), len(self.val_accs))
        epochs = range(1, n + 1)

        # 繪製 Accuracy 曲線
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_accs[:n], label="Train Acc", color="green")
        plt.plot(epochs, self.val_accs[:n],   label="Val Acc",   color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()
        plt.grid(alpha=0.3)

        # 繪製 Loss 曲線
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_losses[:n], label="Train Loss", color="blue")
        plt.plot(epochs, self.val_losses[:n],   label="Val Loss",   color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

# =========================
# 4. evaluate_model：顯示混淆矩陣、分類報告，並計算平均驗證 Loss
# =========================

from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model: nn.Module, dataloader: DataLoader, data_dict, device="cpu"):
    """
    載入純權重後做一次驗證集推論，並列印混淆矩陣 (Confusion Matrix) 與分類報告 (Classification Report)。
    同時計算並顯示「平均驗證 Loss」(Average Validation Loss)。
    因為 classes_ 可能是 int，classification_report 的 target_names 只能是 str，所以先把它轉成字串。
    """
    model.eval()
    model.to(device)

    all_preds   = []
    all_targets = []
    total_loss  = 0.0
    n_samples   = 0
    criterion   = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in dataloader:
            x   = x.to(device)
            y   = y.to(device)
            logits = model(x)                     # (batch, num_classes)
            loss   = criterion(logits, y)
            preds  = torch.argmax(logits, dim=-1)

            bs = x.size(0)
            total_loss += loss.item() * bs
            n_samples  += bs

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # 平均驗證 Loss
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0

    # 混淆矩陣 (Confusion Matrix)
    cm = confusion_matrix(all_targets, all_preds)

    # 把 LabelEncoder.classes_（原本是 int）都轉成字串
    labels_str = [str(c) for c in data_dict['le_gender'].classes_]

    # 呼叫 classification_report，並傳入字串列表作為 target_names
    report = classification_report(
        all_targets,
        all_preds,
        target_names=labels_str,
        zero_division=0   # 若某些類別沒有任何預測，避免出現 warning
    )

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    print(f"\nAverage Validation Loss: {avg_loss:.4f}")

    return avg_loss

# =========================
# 5. 保留「執行訓練、儲存模型、評估」流程在 if __name__ == '__main__': 中
# =========================

if __name__ == '__main__':
    # 5.1 資料前處理：呼叫 load_and_preprocess()
    data_dict = load_and_preprocess(
        datapath     = DATAPATH,
        info_path    = INFO_PATH,
        group_size   = GROUP_SIZE,
        test_size    = TEST_SIZE,
        random_state = RANDOM_SEED
    )

    # 5.2 取 X / y（gender 二元分類）
    X_train = data_dict['X_train']        # (N_train_segments, 27, feature_dim)
    X_test  = data_dict['X_test']         # (N_test_segments, 27, feature_dim)
    y_train = data_dict['y_gender']       # (N_train_segments,)
    y_test  = data_dict['y_gender_te']    # (N_test_segments,)

    FEATURE_DIM = X_train.shape[2]
    NUM_CLASSES = len(data_dict['le_gender'].classes_)  # 2

    # 5.3 建 Dataset / DataLoader
    #    （注意：Windows 下如果要用 persistent_workers，num_workers 必須 > 0；若遇錯誤可以改 num_workers=0）
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset   = TensorDataset(X_test,  y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True  # 只有在 num_workers>0 時才能用
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    # 5.4 建立 LightningModule
    lit_model = LitCNNGenderClassifier(
        feature_dim=FEATURE_DIM,
        num_classes=NUM_CLASSES,
        lr=LR
    )

    # 5.5 設定 Checkpoint (只保留 val_acc 最好的權重)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="cnn_gender_checkpoints",
        filename="cnn-gender-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max"
    )

    # 5.6 開始訓練
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",              # 自動選 GPU / CPU
        callbacks=[PlotMetricsCallback(), checkpoint_callback],
        logger=False                     # 不需要 TensorBoard logger 時可關閉
    )
    trainer.fit(lit_model, train_loader, val_loader)

    # 5.7 訓練完後，存下最佳模型權重 (純 PyTorch state_dict)
    best_ckpt_path = checkpoint_callback.best_model_path
    print(f"\nBest checkpoint saved at: {best_ckpt_path}")

    best_model = SwingCNN(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES).to(torch.device("cpu"))
    state = torch.load(best_ckpt_path)["state_dict"]
    best_model.load_state_dict(state, strict=False)
    torch.save(best_model.state_dict(), "cnn_gender_model.pth")
    print("Saved PyTorch weights to cnn_gender_model.pth")

    # 5.8 在驗證集上做一次完整的評估：印出混淆矩陣 + 分類報告 + 平均 Loss
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model.to(device)
    best_model.load_state_dict(torch.load("cnn_gender_model.pth"))

    evaluate_model(
        best_model,
        DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True),
        data_dict,
        device
    )
