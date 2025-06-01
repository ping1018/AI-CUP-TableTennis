# =============================================================================
# File: cv_cnn_level_with_kaiming.py
# 說明：在 Skorch + GridSearchCV 中，針對 “level” (等級) 進行交叉驗證，
#       並且對 SwingCNN 加入 Kaiming 初始化。使用動態層數和 Dropout 參數進行搜尋。
# =============================================================================

import pandas as pd
import torch
import numpy as np

from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import make_scorer, accuracy_score
import torch.optim as optim
import torch.nn as nn

from data_prepare import load_and_preprocess
from model2 import SwingCNN  # 已存在於 model2.py

# -------------------------------------------------------------------------
# 定義 Kaiming 初始化函式（He 正態）
# -------------------------------------------------------------------------
def init_weights_kaiming(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# -------------------------------------------------------------------------
# 自訂一個 Skorch Callback，在 module 建立後呼叫 init_weights_kaiming
# -------------------------------------------------------------------------
class InitWeightsCallback(Callback):
    def initialize(self, net):
        # 當 net.module_ 建立好之後，就套用 Kaiming 初始化
        net.module_.apply(init_weights_kaiming)

# =============================================================================
# 1. 載入資料（只取 train 部分，用來做 CV）
# =============================================================================
data = load_and_preprocess(
    datapath     = './tabular_data_train',
    info_path    = '39_Training_Dataset/train_info.csv',
    group_size   = 27,
    test_size    = 0.2,       # 這裡的 test 只是切出最終測試集
    random_state = 42
)

X_all   = data['X_train'].numpy()            # (N_segments, 27, feature_dim)
y_all   = data['y_level'].numpy()            # (N_segments,) → 等級編碼
groups  = data['group_player_ids'].numpy()   # (N_segments,) → player_id 分群

FEATURE_DIM = X_all.shape[2]                 # 例如 34
NUM_CLASSES = len(data['le_level'].classes_) # 等級總共有幾個類別

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

    # 傳給 SwingCNN 的參數
    module__feature_dim  = FEATURE_DIM,
    module__num_classes  = NUM_CLASSES,
    # module__num_layers 與 module__dropout_rate 後續由 param_grid 控制

    # 加入自訂的 Kaiming 初始化 Callback
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
    'batch_size':          [ 128],
    'lr':                  [1e-3],
    'module__num_layers':  [1, 2, 3,4],
    'module__dropout_rate':[0.0,0.1,0.2],
}

# 評估指標：使用「Accuracy」作為多類別分類指標
acc_scorer = make_scorer(accuracy_score)

# =============================================================================
# 5. 執行 GridSearchCV（確保同一位 player_id 不會拆到 train & val）
# =============================================================================
gs = GridSearchCV(
    estimator  = net,
    param_grid = param_grid,
    scoring    = acc_scorer,
    cv         = cv.split(X_all, y_all, groups=groups),
    verbose    = 2
)

gs.fit(X_all, y_all)

# =============================================================================
# 6. 輸出結果
# =============================================================================
pd.DataFrame(gs.cv_results_).to_csv(
    'CVRecord/Level/SwingCNN_GroupKFold_Level_Kaiming.csv',
    index=True
)

print("=== Level CV with Kaiming 初始化 結果 ===")
print("Best params:", gs.best_params_)
print("Best Accuracy:", gs.best_score_)
