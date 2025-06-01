import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch

def load_and_preprocess(
    datapath: str,
    info_path: str,
    group_size: int = 27,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    讀取並前處理 AI Cup 資料。
    參數：
      datapath     - 特徵 CSV 檔所在資料夾
      info_path    - train_info.csv 路徑
      group_size   - 每個樣本組大小（比賽固定為 27 次揮拍）
      test_size    - 測試集比例
      random_state - 隨機亂數種子
    回傳：
      包含訓練 / 測試資料與對應標籤，以及四個 LabelEncoder 物件
    """
    # 1. 讀取官方 train_info.csv，並劃分玩家到訓練 / 測試
    info = pd.read_csv(info_path)
    unique_players = info['player_id'].unique()
    train_players, test_players = train_test_split(
        unique_players,
        test_size=test_size,
        random_state=random_state
    )

    # 2. 準備合併所有玩家的特徵與對應標籤
    datalist    = list(Path(datapath).glob('**/*.csv'))
    target_cols = ['gender', 'hold racket handed', 'play years', 'level']

    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_cols)
    x_test  = pd.DataFrame()
    y_test  = pd.DataFrame(columns=target_cols)

    for file in datalist:
        uid = int(Path(file).stem)
        row = info[info['unique_id'] == uid]
        if row.empty:
            # 若此 unique_id 不在 train_info，跳過
            continue

        # 讀取該次揮拍的所有時間序列特徵
        df = pd.read_csv(file)

        # 將該筆 metadata 重複到與 df 相同長度
        target_rep = pd.concat(
            [row[target_cols]] * len(df),
            ignore_index=True
        )

        pid = row['player_id'].iloc[0]
        if pid in train_players:
            # 加入訓練集
            x_train = pd.concat([x_train, df], ignore_index=True)
            y_train = pd.concat([y_train, target_rep], ignore_index=True)
        else:
            # 加入測試集
            x_test  = pd.concat([x_test, df], ignore_index=True)
            y_test  = pd.concat([y_test, target_rep], ignore_index=True)

    # 3. 特徵標準化到 [0,1]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled  = scaler.transform(x_test)

    # 4. 為了符合 group_size，裁掉尾巴不足一組的資料
    n_train = (len(X_train_scaled) // group_size) * group_size
    n_test  = (len(X_test_scaled)  // group_size) * group_size

    # 轉成 PyTorch tensor
    Xt_train = torch.tensor(
        X_train_scaled[:n_train],
        dtype=torch.float32
    )
    Xt_test  = torch.tensor(
        X_test_scaled[:n_test],
        dtype=torch.float32
    )

    # 重塑為 (num_groups, group_size, feature_dim)
    Xg_train = Xt_train.view(
        -1, group_size, Xt_train.shape[1]
    )
    Xg_test  = Xt_test.view(
        -1, group_size, Xt_test.shape[1]
    )

    # 5. 用 LabelEncoder 分別編碼四個目標
    le_years  = LabelEncoder()
    le_level  = LabelEncoder()
    le_gender = LabelEncoder()
    le_hold   = LabelEncoder()

    # 5a. play years
    years_train = le_years.fit_transform(
        y_train['play years'].values
    )[:n_train]
    years_test  = le_years.transform(
        y_test['play years'].values
    )[:n_test]

    # 5b. level
    level_train = le_level.fit_transform(
        y_train['level'].values
    )[:n_train]
    level_test  = le_level.transform(
        y_test['level'].values
    )[:n_test]

    # 5c. gender（二元分類）
    gender_train = le_gender.fit_transform(
        y_train['gender'].values
    )[:n_train]
    gender_test  = le_gender.transform(
        y_test['gender'].values
    )[:n_test]

    # 5d. hold racket handed（二元分類）
    hold_train = le_hold.fit_transform(
        y_train['hold racket handed'].values
    )[:n_train]
    hold_test  = le_hold.transform(
        y_test['hold racket handed'].values
    )[:n_test]

    # 6. 假設同一組內標籤都相同，取每組的第一筆作為該組標籤
    yg_years      = torch.tensor(years_train,  dtype=torch.long).view(-1, group_size)[:, 0]
    yg_level      = torch.tensor(level_train,  dtype=torch.long).view(-1, group_size)[:, 0]
    yg_gender     = torch.tensor(gender_train, dtype=torch.long).view(-1, group_size)[:, 0]
    yg_hold       = torch.tensor(hold_train,   dtype=torch.long).view(-1, group_size)[:, 0]

    yg_years_te   = torch.tensor(years_test,   dtype=torch.long).view(-1, group_size)[:, 0]
    yg_level_te   = torch.tensor(level_test,   dtype=torch.long).view(-1, group_size)[:, 0]
    yg_gender_te  = torch.tensor(gender_test,  dtype=torch.long).view(-1, group_size)[:, 0]
    yg_hold_te    = torch.tensor(hold_test,    dtype=torch.long).view(-1, group_size)[:, 0]

    # 7. 回傳前處理結果
    return {
        'X_train':     Xg_train,      # 訓練集特徵
        'y_years':     yg_years,      # 訓練集球齡標籤
        'y_level':     yg_level,      # 訓練集等級標籤
        'y_gender':    yg_gender,     # 訓練集性別標籤
        'y_hold':      yg_hold,       # 訓練集持拍手標籤

        'X_test':      Xg_test,       # 測試集特徵
        'y_years_te':  yg_years_te,   # 測試集球齡標籤
        'y_level_te':  yg_level_te,   # 測試集等級標籤
        'y_gender_te': yg_gender_te,  # 測試集性別標籤
        'y_hold_te':   yg_hold_te,    # 測試集持拍手標籤

        'le_years':    le_years,      # play years 編碼器（反編碼用）
        'le_level':    le_level,      # level 編碼器（反編碼用）
        'le_gender':   le_gender,     # gender 編碼器（反編碼用）
        'le_hold':     le_hold        # hold 編碼器（反編碼用）
    }


if __name__ == '__main__':
    data = load_and_preprocess(
        datapath='./tabular_data_train',
        info_path='39_Training_Dataset/train_info.csv',
        group_size=27
    )

    # 原有的 shape 列印
    print("X_train shape:    ", data['X_train'].shape)
    print("y_years shape:    ", data['y_years'].shape)
    print("y_level shape:    ", data['y_level'].shape)
    print("y_gender shape:   ", data['y_gender'].shape)
    print("y_hold shape:     ", data['y_hold'].shape)
    print("X_test shape:     ", data['X_test'].shape)
    print("y_years_te shape: ", data['y_years_te'].shape)
    print("y_level_te shape: ", data['y_level_te'].shape)
    print("y_gender_te shape:", data['y_gender_te'].shape)
    print("y_hold_te shape:  ", data['y_hold_te'].shape)
    print("\n" + "="*30 + "\n")

    # helper：列印一個任務的類別分佈
    from collections import Counter
    def print_dist(y_tensor, le, title):
        cnt = Counter(y_tensor.tolist())
        print(f"{title} 分佈：")
        for cls_int, c in sorted(cnt.items()):
            cls_name = le.inverse_transform([cls_int])[0]
            print(f"  {cls_name:<10} ({cls_int}): {c}")
        print()

    # --- 訓練集分佈 ---
    print(">>> Train 集分佈 <<<")
    print(f"總樣本數：{data['X_train'].shape[0]}\n")
    print_dist(data['y_years'],  data['le_years'],  "球齡 (years)")
    print_dist(data['y_level'],  data['le_level'],  "等級 (level)")
    print_dist(data['y_gender'], data['le_gender'], "性別 (gender)")
    print_dist(data['y_hold'],   data['le_hold'],   "持拍手 (hold)")

    # --- 驗證集分佈（目前的 test 當作 val） ---
    print(">>> Validation 集分佈 <<<")
    print(f"總樣本數：{data['X_test'].shape[0]}\n")
    print_dist(data['y_years_te'],  data['le_years'],  "球齡 (years)")
    print_dist(data['y_level_te'],  data['le_level'],  "等級 (level)")
    print_dist(data['y_gender_te'], data['le_gender'], "性別 (gender)")
    print_dist(data['y_hold_te'],   data['le_hold'],   "持拍手 (hold)")

