import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import torch
from collections import Counter

def load_and_preprocess_with_padding(datapath, info_path, test_size=0.2, random_state=42):
    info = pd.read_csv(info_path)
    files = sorted(Path(datapath).glob('*.txt'), key=lambda x: int(x.stem))

    X = []
    y_gender, y_hold, y_years, y_level = [], [], [], []
    player_ids = []

    for file in files:
        uid = int(file.stem)
        row = info[info['unique_id'] == uid]
        if row.empty:
            continue
        arr = np.loadtxt(file)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        X.append(torch.tensor(arr, dtype=torch.float32))
        y_gender.append(row['gender'].values[0])
        y_hold.append(row['hold racket handed'].values[0])
        y_years.append(row['play years'].values[0])
        y_level.append(row['level'].values[0])
        player_ids.append(row['player_id'].values[0])

    X = np.array(X, dtype=object)
    y_gender = np.array(y_gender)
    y_hold = np.array(y_hold)
    y_years = np.array(y_years)
    y_level = np.array(y_level)
    player_ids = np.array(player_ids)

    # 分群切分，確保 player 不重複
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y_level, groups=player_ids))

    def show_distribution(label, name):
        print(f"\n分類目標：{name}")
        print("Train 分布：", Counter(label[train_idx]))
        print("Test  分布：", Counter(label[test_idx]))

    show_distribution(y_gender, "gender")
    show_distribution(y_hold, "hold")
    show_distribution(y_years, "years")
    show_distribution(y_level, "level")

    le_gender = LabelEncoder()
    le_hold = LabelEncoder()
    le_years = LabelEncoder()
    le_level = LabelEncoder()

    y_train_gender = torch.tensor(le_gender.fit_transform(y_gender[train_idx]), dtype=torch.long)
    y_test_gender  = torch.tensor(le_gender.transform(y_gender[test_idx]), dtype=torch.long)
    y_train_hold   = torch.tensor(le_hold.fit_transform(y_hold[train_idx]), dtype=torch.long)
    y_test_hold    = torch.tensor(le_hold.transform(y_hold[test_idx]), dtype=torch.long)
    y_train_years  = torch.tensor(le_years.fit_transform(y_years[train_idx]), dtype=torch.long)
    y_test_years   = torch.tensor(le_years.transform(y_years[test_idx]), dtype=torch.long)
    y_train_level  = torch.tensor(le_level.fit_transform(y_level[train_idx]), dtype=torch.long)
    y_test_level   = torch.tensor(le_level.transform(y_level[test_idx]), dtype=torch.long)

    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    lengths_train = torch.tensor([x.shape[0] for x in X_train], dtype=torch.long)
    lengths_test = torch.tensor([x.shape[0] for x in X_test], dtype=torch.long)
    X_train_pad = pad_sequence(X_train, batch_first=True)
    X_test_pad = pad_sequence(X_test, batch_first=True)

    return {
        'X_train': X_train_pad,
        'X_test': X_test_pad,
        'y_gender': y_train_gender,
        'y_gender_te': y_test_gender,
        'y_hold': y_train_hold,
        'y_hold_te': y_test_hold,
        'y_years': y_train_years,
        'y_years_te': y_test_years,
        'y_level': y_train_level,
        'y_level_te': y_test_level,
        'le_gender': le_gender,
        'le_hold': le_hold,
        'le_years': le_years,
        'le_level': le_level,
        'group_player_ids': torch.tensor(player_ids[train_idx]),
        'group_player_ids_te': torch.tensor(player_ids[test_idx]),
        'lengths_train': lengths_train,
        'lengths_test': lengths_test
    }
    
if __name__ == "__main__":
    data = load_and_preprocess_with_padding(
        datapath="39_Training_Dataset/train_data",
        info_path="39_Training_Dataset/train_info.csv"
    )
    print("X_train shape:", data['X_train'].shape)
    print("y_level distribution:", torch.bincount(data['y_level']))


