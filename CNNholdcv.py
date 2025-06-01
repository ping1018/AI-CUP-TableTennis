# =============================================================================
# File: cv_cnn_hold_with_kaiming.py
# 說明：在 Skorch + GridSearchCV 中，針對 “hold racket handed” (持拍手) 進行交叉驗證，
#       並且對 SwingCNN 加入 Kaiming 初始化。使用動態層數和 Dropout 參數進行搜尋。
# =============================================================================

import pandas as pd
import torch
import numpy as np

from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import make_scorer, roc_auc_score
import torch.optim as optim
import torch.nn as nn

from data_prepare import load_and_preprocess
from model2 import SwingCNN  # 載入更新後的可動態層數 SwingCNN

# -------------------------------------------------------------------------
# 定義 Kaiming 初始化函式（He 正態）
# -------------------------------------------------------------------------
def init_weights_kaiming(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# -------------------------------------------------------------------------
# 自訂一個 Skorch Callback，在 module 建立好之後呼叫 init_weights_kaiming
# -------------------------------------------------------------------------
class InitWeightsCallback(Callback):
    def initialize(self, net):
        # 當 net.module_ 建立完畢後，就對其所有 Conv1d / Linear 層套用 Kaiming 初始化
        net.module_.apply(init_weights_kaiming)

# =============================================================================
# 1. 載入資料（只取 train 部分，用來做 CV）
# =============================================================================
data = load_and_preprocess(
    datapath     = './tabular_data_train',
    info_path    = '39_Training_Dataset/train_info.csv',
    group_size   = 27,
    test_size    = 0.2,       # 這裡的 test 只是切出最後獨立測試集
    random_state = 42
)

# X_all: (N_segments, 27, feature_dim)
X_all   = data['X_train'].numpy()
# y_all: (N_segments,) → “持拍手” (0=一邊手, 1=另一邊手)
y_all   = data['y_hold'].numpy()
# groups: (N_segments,) → 依 player_id 進行分群
groups  = data['group_player_ids'].numpy()

FEATURE_DIM = X_all.shape[2]           # 例如 34
NUM_CLASSES = len(np.unique(y_all))    # 持拍手二分類，所以應該是 2

# =============================================================================
# 2. 建立 Skorch NeuralNetClassifier，並加入 InitWeightsCallback
# =============================================================================
net = NeuralNetClassifier(
    module         = SwingCNN,
    criterion      = nn.CrossEntropyLoss,
    optimizer      = optim.Adam,
    max_epochs     = 20,
    lr             = 1e-3,
    batch_size     = 32,
    device         = 'cuda',
    iterator_train__shuffle = True,

    # 傳給 SwingCNN 的參數：
    # feature_dim: 輸入通道數 (即原始特徵維度)
    # num_classes: 分類數 (二分類)
    module__feature_dim  = FEATURE_DIM,
    module__num_classes  = NUM_CLASSES,

    # 之後可以透過 module__num_layers、module__dropout_rate 參數來調整深度與 Dropout
    # 這邊先不固定，留到 GridSearch 的 param_grid 裡搜尋

    # 加入自訂的初始化 Callback
    callbacks      = [InitWeightsCallback()]
)

# =============================================================================
# 3. 設定 GroupKFold：依 player_id 作為 group
# =============================================================================
cv = GroupKFold(n_splits=5)

# =============================================================================
# 4. 定義要做 GridSearchCV 的參數網格
#    - batch_size、lr 固定為多選
#    - module__num_layers: 動態調整卷積層數 (1,2,3)
#    - module__dropout_rate: 調整 Dropout 比例 (0.3, 0.5)
# =============================================================================
param_grid = {
    'batch_size':          [128],
    'lr':                  [1e-3],
    'module__num_layers':  [1, 2, 3,4],
    'module__dropout_rate':[0.0,0.1,0.2],
}

# 評估指標：使用 ROC-AUC（二分類最佳化指標）
roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# =============================================================================
# 5. 執行 GridSearchCV（確保同一位 player_id 不會拆到 train & val）
# =============================================================================
gs = GridSearchCV(
    estimator  = net,
    param_grid = param_grid,
    scoring    = roc_auc_scorer,
    cv         = cv.split(X_all, y_all, groups=groups),
    verbose    = 2
)

gs.fit(X_all, y_all)

# =============================================================================
# 6. 輸出結果
# =============================================================================
pd.DataFrame(gs.cv_results_).to_csv(
    'CVRecord/Hold/SwingCNN_GroupKFold_Hold_Kaiming.csv',
    index=True
)

print("=== Hold-hand CV with Kaiming 初始化 結果 ===")
print("Best params:", gs.best_params_)
print("Best ROC-AUC:", gs.best_score_)
