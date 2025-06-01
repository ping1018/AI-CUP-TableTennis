# =============================================================================
# File: cv_cnn_gender_with_kaiming.py
# 說明：在 Skorch + GridSearchCV 中，對 SwingCNN 加入 Kaiming 初始化，
#      並把 num_layers、dropout_rate、lr、batch_size 納入超參搜尋，
#      使用 GroupKFold 確保同一位 player_id 不會拆到 train & val。
# =============================================================================

import pandas as pd
import torch
import numpy as np

from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from sklearn.model_selection import GridSearchCV, GroupKFold
import torch.optim as optim
import torch.nn as nn

from data_prepare import load_and_preprocess
from model2 import SwingCNN  # 已更新為支援 num_layers、dropout_rate 的版本

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
        # net.module_ 在建立好之後就套用 Kaiming 初始化
        net.module_.apply(init_weights_kaiming)

# =============================================================================
# 1. 載入資料（只取 train 部分，用來做 CV）
# =============================================================================
data = load_and_preprocess(
    datapath     = './tabular_data_train',
    info_path    = '39_Training_Dataset/train_info.csv',
    group_size   = 27,
    test_size    = 0.2,       # 這裡的 test 只是最後留作獨立測試
    random_state = 42
)

X_all   = data['X_train'].numpy()             # (N_segments, 27, feature_dim)
y_all   = data['y_gender'].numpy()            # (N_segments,)
groups  = data['group_player_ids'].numpy()    # (N_segments,)

FEATURE_DIM = X_all.shape[2]                  # 例如 34
NUM_CLASSES = len(np.unique(y_all))           # gender = 2

# =============================================================================
# 2. 建立 Skorch NeuralNetClassifier，並預設一組參數
# =============================================================================
net = NeuralNetClassifier(
    module         = SwingCNN,
    criterion      = nn.CrossEntropyLoss,
    optimizer      = optim.Adam,
    max_epochs     = 20,
    lr             = 1e-3,
    batch_size     = 128,
    device         = 'cuda',               # 若無 GPU 可改 'cpu'
    iterator_train__shuffle = True,

    # 傳給 SwingCNN 的固定參數（其餘在 param_grid 裡搜索）
    module__feature_dim   = FEATURE_DIM,
    module__num_classes   = NUM_CLASSES,
    module__num_layers    = 2,             # 搜索網格裡會被覆蓋
    module__dropout_rate  = 0.5,           # 搜索網格裡會被覆蓋

    # 加入剛剛自訂的初始化 Callback
    callbacks      = [InitWeightsCallback()],
)

# =============================================================================
# 3. 設定 GroupKFold：依 player_id 作為 group
# =============================================================================
cv = GroupKFold(n_splits=5)

# =============================================================================
# 4. 定義要做 GridSearchCV 的參數網格
#    - module__num_layers：測試 1, 2, 3 層 Conv
#    - module__dropout_rate：測試不同 Dropout 比例
#    - lr / batch_size 也納入一起搜尋
# =============================================================================
param_grid = {
    'module__num_layers':    [1, 2, 3, 4],
    'module__dropout_rate':  [0.0,0.1,0.2],
    'lr':                    [1e-3],
    'batch_size':            [128],
}

# =============================================================================
# 5. 執行 GridSearchCV（確保同一位 player_id 不會拆到 train & val）
# =============================================================================
gs = GridSearchCV(
    estimator = net,
    param_grid= param_grid,
    scoring   = 'roc_auc', 
    cv        = cv.split(X_all, y_all, groups=groups),          # Windows 下建議設為 1，避免多執行緒問題
    verbose   = 2
)

gs.fit(X_all, y_all)

# =============================================================================
# 6. 輸出結果
# =============================================================================
pd.DataFrame(gs.cv_results_) \
  .to_csv('CVRecord/Gender/SwingCNN_GroupKFold_Kaiming.csv', index=True)

print("=== Gender CV with Kaiming 初始化 結果 ===")
print("Best params:", gs.best_params_)
print("Best ROC‐AUC:", gs.best_score_)
