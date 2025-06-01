import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score

from model3 import CNN_LSTM
from data_prepare_padded import load_and_preprocess_wholefile_with_padding

# ==== Configuration ====
DATAPATH = '39_Training_Dataset/train_data'
INFO_PATH = '39_Training_Dataset/train_info.csv'
BATCH_SIZE = 128
LR = 1e-3
MAX_EPOCHS = 20
NUM_CLASSES = 2  # 根據目前等級是否為二分類

# ==== Load Data ====
data_dict = load_and_preprocess_wholefile_with_padding(DATAPATH, INFO_PATH)
X_train, y_train = data_dict['X_train'], data_dict['y_level']
X_test, y_test = data_dict['X_test'], data_dict['y_level_te']
lengths_train, lengths_test = data_dict['lengths_train'], data_dict['lengths_test']
label_encoder = data_dict['le_level']
class_names = label_encoder.classes_

# ==== Dataset + Dataloader ====
class PadDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = X
        self.y = y
        self.lengths = lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]

def collate_fn(batch):
    x, y, lengths = zip(*batch)
    return torch.stack(x), torch.tensor(y), torch.tensor(lengths)

train_loader = DataLoader(PadDataset(X_train, y_train, lengths_train), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(PadDataset(X_test, y_test, lengths_test), batch_size=BATCH_SIZE, collate_fn=collate_fn)

# ==== Lightning Module ====
class LitCNNLSTM(pl.LightningModule):
    def __init__(self, input_channels=6, num_classes=NUM_CLASSES, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = CNN_LSTM(input_channels=input_channels, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x, lengths):
        return self.model(x, lengths)

    def _shared_step(self, batch):
        x, y, lengths = batch
        logits = self(x, lengths)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ==== Callback: Accuracy Plot ====
class PlotMetricsCallback(Callback):
    def __init__(self):
        self.train_accs = []
        self.val_accs = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_accs.append(trainer.callback_metrics["train_acc"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_accs.append(trainer.callback_metrics["val_acc"].item())

    def on_train_end(self, trainer, pl_module):
        plt.plot(self.train_accs, label='Train Acc', color='blue')
        plt.plot(self.val_accs, label='Val Acc', color='orange')
        plt.title("Accuracy Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

# ==== Train ====
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    dirpath="cnn_lstm_checkpoints",
    filename="cnn-lstm-level-{epoch:02d}-{val_acc:.2f}",
    save_top_k=1,
    mode="max"
)

model = LitCNNLSTM(input_channels=6, num_classes=NUM_CLASSES, lr=LR)
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    callbacks=[PlotMetricsCallback(), checkpoint_callback],
    logger=False
)
trainer.fit(model, train_loader, val_loader)

# ==== Save best model ====
best_model = model.model.cpu()
torch.save(best_model.state_dict(), "rawsignallevel.pth")
print("Best model saved to rawsignallevel.pth")

# ==== Evaluate ====
def evaluate_model(model, dataloader):
    model.eval()
    model.to("cpu")
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y, lengths in dataloader:
            logits = model(x, lengths)
            probs = F.softmax(logits, dim=1)[:, 1]  # 只取 class=1 機率
            all_probs.extend(probs.numpy())
            all_labels.extend(y.numpy())

    auc_score = roc_auc_score(all_labels, all_probs)
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    cm = confusion_matrix(all_labels, preds)

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    print(f"Test ROC AUC: {roc_auc:.4f}")

evaluate_model(best_model, val_loader)
