import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


def time_features(dates, freq="h"):
    features = []
    features.append(dates.month.to_numpy())
    features.append(dates.day.to_numpy())
    features.append(dates.weekday.to_numpy())
    features.append(dates.hour.to_numpy())
    if freq == "t":
        features.append(dates.minute.to_numpy() // 15)

    features[0] = features[0] / 12.0 - 0.5
    features[1] = features[1] / 31.0 - 0.5
    features[2] = features[2] / 6.0 - 0.5
    features[3] = features[3] / 23.0 - 0.5
    if freq == "t":
        features[4] = features[4] / 3.0 - 0.5

    return np.stack(features, axis=1)


class Dataset_ETT(Dataset):
    def __init__(
        self,
        root_path: str,
        data_filename: str = "ETTh1.csv",
        flag: str = "train",
        size: List[int] = [96, 48, 24], 
        features: str = "M", 
        target: str = "OT",  
        scale: bool = True, 
        timeenc: int = 1,  
        freq: str = "h", 
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

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

        """
        Standard ETT splits:
        ETTh1/ETTh2: Train 12 months, Val 4 months, Test 4 months
        ETTm1/ETTm2: Train 12 months, Val 4 months, Test 4 months
        Note: Border indices might slightly vary based on exact start/end dates or implementations.
              Using common split points based on index count.
        """
        border1s = {
            "ETTh1": 12 * 30 * 24,
            "ETTh2": 12 * 30 * 24,
            "ETTm1": 12 * 30 * 24 * 4,
            "ETTm2": 12 * 30 * 24 * 4,
        }
        border2s = {
            "ETTh1": 12 * 30 * 24 + 4 * 30 * 24,
            "ETTh2": 12 * 30 * 24 + 4 * 30 * 24,
            "ETTm1": 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            "ETTm2": 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
        }

        data_name = self.data_path.split(".")[0]
        border1 = border1s.get(data_name, 0)
        border2 = border2s.get(data_name, len(df_raw) * 0.7)

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unknown features type: {self.features}")

        df_data_values = df_data.values

        if self.scale:
            train_data = df_data_values[:border1]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data_values)
            print(f"Data scaled using StandardScaler (fit on train split: 0-{border1})")
        else:
            data = df_data_values

        df_stamp = df_raw[["date"]][:]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 1:
            data_stamp = time_features(df_stamp["date"], freq=self.freq)
        else:
            data_stamp = np.zeros((len(df_stamp), 1))

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp

        border1 = (
            border1 - self.seq_len
        )  # Ensure enough history for the first validation sample
        border2 = (
            border2 - self.seq_len
        )  # Ensure enough history for the first test sample

        if self.flag == "train":
            self.start_idx = 0
            self.end_idx = border1
        elif self.flag == "val":
            self.start_idx = border1
            self.end_idx = border2
        else:  # test
            self.start_idx = border2
            self.end_idx = len(self.data_x)  # Go to the end

        print(
            f"Dataset '{data_name}' ({self.flag}): Using data indices from {self.start_idx} to {self.end_idx-1}"
        )

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        true_index = self.start_idx + index

        s_begin = true_index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        if r_end > self.end_idx + self.pred_len:
            pass

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.features == "MS":
            target_idx = list(self.scaler.feature_names_in_).index(self.target)
            seq_y = seq_y[:, target_idx : target_idx + 1]

        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        total_len_for_split = self.end_idx - self.start_idx
        num_samples = total_len_for_split - self.seq_len - self.pred_len + 1
        return max(0, num_samples)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Applies inverse scaling to the data.
        Assumes data corresponds to the features the scaler was fit on,
        or is a single column corresponding to the target if features='MS'.
        """
        if not self.scale:
            return data

        if self.features == "MS":
            num_features = len(self.scaler.mean_)
            target_idx = list(self.scaler.feature_names_in_).index(self.target)

            original_shape = data.shape
            if len(original_shape) == 3:
                batch_size, seq_len, _ = original_shape
                data_padded = np.zeros((batch_size * seq_len, num_features))
                data_reshaped = data.reshape(batch_size * seq_len, 1)
                data_padded[:, target_idx] = data_reshaped[:, 0]
                inversed = self.scaler.inverse_transform(data_padded)[:, target_idx]
                return inversed.reshape(batch_size, seq_len, 1)
            elif len(original_shape) == 2:
                seq_len, _ = original_shape
                data_padded = np.zeros((seq_len, num_features))
                data_padded[:, target_idx] = data[:, 0]
                inversed = self.scaler.inverse_transform(data_padded)[:, target_idx]
                return inversed.reshape(seq_len, 1)
            else:
                data_padded = np.zeros((1, num_features))
                data_padded[0, target_idx] = data
                inversed = self.scaler.inverse_transform(data_padded)[:, target_idx]
                return inversed[0]

        elif self.features == "S":
            return self.scaler.inverse_transform(data)
        else:
            return self.scaler.inverse_transform(data)
