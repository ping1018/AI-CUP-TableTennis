# CNNLSTM_years_scv.py
# 針對 years (年資分類) 使用 CNN_LSTM 模型進行交叉驗證 (GroupKFold)

import pandas as pd
import torch
import numpy as np
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import make_scorer, roc_auc_score
import torch.nn as nn
import torch.optim as optim

from data_prepare_padded import load_and_preprocess_with_padding
from model3 import CNN_LSTM

# -------------------------------------------------------------------------
# 初始化權重：Kaiming Normal
# -------------------------------------------------------------------------
def init_weights_kaiming(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class InitWeightsCallback(Callback):
    def on_model_initialized(self, net, model, **kwargs):
        model.apply(init_weights_kaiming)

# -------------------------------------------------------------------------
# 1. 載入資料
# -------------------------------------------------------------------------
print("\n分類目標：years")
data = load_and_preprocess_with_padding(
    datapath     = '39_Training_Dataset/train_data',
    info_path    = '39_Training_Dataset/train_info.csv',
    test_size    = 0.2,
    random_state = 42
)

X_all     = data['X_train'].numpy()
y_all     = data['y_years'].numpy()
groups    = data['group_player_ids'].numpy()
lengths   = data['lengths_train'].numpy()

print("Train 分布：", dict(pd.Series(y_all[groups < max(groups) * 0.8]).value_counts()))
print("Test  分布：", dict(pd.Series(y_all[groups >= max(groups) * 0.8]).value_counts()))

NUM_CLASSES = len(np.unique(y_all))

# -------------------------------------------------------------------------
# 2. 定義 Skorch 模型
# -------------------------------------------------------------------------
class SkorchCNNLSTM(CNN_LSTM):
    def forward(self, x):
        lengths_tensor = torch.full((x.size(0),), x.size(1), dtype=torch.long)
        return super().forward(x, lengths_tensor)

net = NeuralNetClassifier(
    module         = SkorchCNNLSTM,
    criterion      = nn.CrossEntropyLoss,
    optimizer      = optim.Adam,
    max_epochs     = 20,
    lr             = 1e-3,
    batch_size     = 128,
    device         = 'cuda' if torch.cuda.is_available() else 'cpu',
    iterator_train__shuffle = True,

    module__input_channels    = X_all.shape[2],
    module__num_classes       = NUM_CLASSES,

    # Grid Search 會搜尋這些：
    # module__conv_out_channels
    # module__lstm_hidden_size

    callbacks      = [InitWeightsCallback()]
)

# -------------------------------------------------------------------------
# 3. 設定 GridSearchCV
# -------------------------------------------------------------------------
cv = GroupKFold(n_splits=5)
param_grid = {
    'module__conv_out_channels': [32, 64],
    'module__lstm_hidden_size': [128, 256]
}
roc_auc_scorer = make_scorer(roc_auc_score,  multi_class='ovr')

# error_score='raise' 可協助找到失敗原因
gs = GridSearchCV(
    estimator  = net,
    param_grid = param_grid,
    scoring    = roc_auc_scorer,
    cv         = cv.split(X_all, y_all, groups=groups),
    verbose    = 2,
    error_score='raise'
)

gs.fit(X_all, y_all)

# -------------------------------------------------------------------------
# 4. 儲存結果
# -------------------------------------------------------------------------
pd.DataFrame(gs.cv_results_).to_csv(
    'CVRecord/Years/CNNLSTM_GroupKFold_Years.csv', index=True
)

print("=== Years CV with CNN-LSTM 結果 ===")
print("Best params:", gs.best_params_)
print("Best ROC-AUC:", gs.best_score_)
