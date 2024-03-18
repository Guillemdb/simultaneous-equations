from pathlib import Path

import pandas as pd
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader


class SemData(LightningDataModule):
    DATA_FOLDER_ = Path("datos")

    def __init__(
        self, dataset_name: str, test_pct: float = 0.15, val_pct: float = 0.1, batch_size: int = 32
    ):
        super().__init__()
        self.num_y_, self.num_x_, self.num_samples_ = [int(s) for s in dataset_name.split(" ")[1:]]
        self.batch_size_ = batch_size
        self.dataset_name_ = dataset_name
        self.data_paths_ = self.get_model_paths(self.dataset_name_)
        self.test_pct_ = test_pct
        self.val_pct_ = val_pct
        self.val_ix_ = None
        self.test_ix_ = None
        self.x_ = None
        self.y_ = None
        self.mask_x_ = None
        self.mask_y_ = None
        self.model_params_ = None
        self.data_ = None
        self._load_data()

    @staticmethod
    def get_model_paths(name, data_folder=DATA_FOLDER_):
        model_path = Path(data_folder) / name
        data = {
            "path": model_path,
            "model": model_path / "modelo.txt",
            "x": model_path / "X.txt",
            "y": model_path / "Y.txt",
            "data": model_path / "datos.xlsx",
        }
        return data

    @staticmethod
    def read_text_file(path: Path):
        df = pd.read_csv(path, delim_whitespace=True, decimal=",", header=None)
        return df.values

    def _load_data(self):
        self.x_ = self._load_x()
        self.y_ = self._load_y()
        self.model_params_ = self._load_model_params()
        self.mask_y_ = self.model_params_[:, : self.num_y_]
        self.mask_x_ = self.model_params_[:, self.num_y_ :]
        self.data_ = self._load_model_data()
        self._test_ix_ = int(self.num_samples_ * self.test_pct_)
        self._val_ix_ = int(self.num_samples_ * self.val_pct_)
        self.x_train_ = self.x_[: (-self._test_ix_ - self._val_ix_)]
        self.x_val_ = self.x_[-(self._test_ix_ + self._val_ix_) : -self._test_ix_]
        self.x_test_ = self.x_[-self._test_ix_ :]
        self.y_train_ = self.y_[: (-self._test_ix_ - self._val_ix_)]
        self.y_val_ = self.y_[-(self._test_ix_ + self._val_ix_) : -self._test_ix_]
        self.y_test_ = self.y_[-self._test_ix_ :]

    def _load_x(self):
        return self.read_text_file(self.data_paths_["x"])

    def _load_y(self):
        return self.read_text_file(self.data_paths_["y"])

    def _load_model_params(self):
        return self.read_text_file(self.data_paths_["model"])

    def _load_model_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        x = torch.from_numpy(self.x_train_).float()
        y = torch.from_numpy(self.y_train_).float()
        return DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        x = torch.from_numpy(self.x_val_).float()
        y = torch.from_numpy(self.y_val_).float()
        return DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        x = torch.from_numpy(self.x_test_).float()
        y = torch.from_numpy(self.y_test_).float()
        return DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=self.batch_size)
