import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from wrapper.time_features import time_features


class Dataset_ETT(Dataset):
    def __init__(
        self,
        root_path: str,
        data_filename: str = "ETTh1.csv",
        flag: str = "train",
        size: List[int] = [96, 24],
        features: str = "M",
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 1,
        freq: str = "h",
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_filename
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = {
            "ETTh1": [
                0,
                12 * 30 * 24 - self.seq_len,
                12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
            ],
            "ETTh2": [
                0,
                12 * 30 * 24 - self.seq_len,
                12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
            ],
            "ETTm1": 12 * 30 * 24 * 4,
            "ETTm2": 12 * 30 * 24 * 4,
        }
        border2s = {
            "ETTh1": [
                12 * 30 * 24,
                12 * 30 * 24 + 4 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24,
            ],
            "ETTh2": [
                12 * 30 * 24,
                12 * 30 * 24 + 4 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24,
            ],
            "ETTm1": 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            "ETTm2": 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
        }

        data_name = self.data_path.split(".")[0]
        border1 = border1s[data_name][self.set_type]
        border2 = border2s[data_name][self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unknown features type: {self.features}")

        if self.scale:
            train_data = df_data[border1s[data_name][0] : border2s[data_name][0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            if "ETTh" in data_name:
                df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(["date"], 1).values
            elif "ETTm" in data_name:
                df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        total_len_for_split = self.end_idx - self.start_idx
        num_samples = total_len_for_split - self.seq_len - self.pred_len + 1
        return max(0, num_samples)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
