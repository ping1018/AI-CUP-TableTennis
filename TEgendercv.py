import pandas as pd
import torch
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
import torch.optim as optim
import torch.nn as nn

from data_prepare import load_and_preprocess
from model import TransformerClassifier



# --- 1. 載入資料 ---
data = load_and_preprocess(
    datapath='./tabular_data_train',
    info_path='39_Training_Dataset/train_info.csv',
    test_size    = 0.2, 
    group_size=27,
    random_state = 42
    
)
X = data['X_train'].cpu().numpy()
y = data['y_gender'].cpu().numpy()
groups  = data['group_player_ids'].numpy()

# --- 2. 基本超參 ---
FEATURE_D = X.shape[2]
SEQ_LEN   = X.shape[1]

# --- 3. 建立 Skorch net（指定 GPU） ---
net = NeuralNetClassifier(
    module=TransformerClassifier,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    max_epochs=20,
    lr=1e-3,
    batch_size=32,
    device='cuda',                      # force GPU
    iterator_train__shuffle=True, 

    # 傳給 module 的參數
    module__feature_dim = FEATURE_D,
    module__seq_len     = SEQ_LEN,
    module__d_model     = 64,
    module__nhead       = 4,
    module__num_layers  = 2,
    module__dropout     = 0.1,
    module__num_classes = 2
)

# --- 4. CV 與參數網格 ---
cv = StratifiedGroupKFold(n_splits=5)
param_grid = {
    'batch_size':        [128],
    'module__d_model':   [32,64],
    'module__nhead':     [1,2,4],
    'module__num_layers':[1,2,3],
    'lr':                [1e-3],
    'module__dropout': [0.0]
}

# --- 5. GridSearchCV（單進程 n_jobs=1）---
gs = GridSearchCV(
    net,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv.split(X, y, groups=groups),
    verbose=2              
)
gs.fit(X, y)

# --- 6. 存檔並印出結果 ---
pd.DataFrame(gs.cv_results_).to_csv('CVRecord/Gender/TEncoder.csv', index=True)
print("=== Gender CV ===")
print("Best params:", gs.best_params_)
print("Best AUC:",    gs.best_score_)
