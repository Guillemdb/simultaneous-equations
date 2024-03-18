import os
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import product
import pickle



class OldSemData(LightningDataModule):
    DATA_FOLDER_ = Path("new_datos")

    def __init__(
        self,
        dataset_name: str,
        n_train: int,
        n_test: int,
        n_val: int,
        batch_size: int = 32,
        use_splits: bool = True,
    ):
        super().__init__()
        self.num_y_, self.num_x_, self.num_samples_ = [
            int(s) for s in dataset_name.split(" ")[-3:]
        ]
        self.batch_size = batch_size
        self.dataset_name_ = dataset_name
        self.data_paths_ = self.get_model_paths(self.dataset_name_)
        self.n_train = n_train
        self.n_test = n_test
        self.n_val = n_val
        self.x_ = None
        self.y_ = None
        self.mask_x_ = None
        self.mask_y_ = None
        self.model_params_ = None
        self.data_ = None
        self.use_splits_ = use_splits
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
    def read_text_file(path: Path, decimal=","):
        df = pd.read_csv(path, delim_whitespace=True, decimal=decimal, header=None)
        return df.values

    def _load_data(self):
        
        self.x_ = self._load_x().astype(np.float32)
        self.y_ = self._load_y().astype(np.float32)
        self.data_ = self._load_model_data().astype(np.float32)
        self.model_params_ = self.data_.astype(bool)  # self._load_model_params()
        self.mask_y_ = self.model_params_[:, : self.num_y_]
        self.mask_x_ = self.model_params_[:, self.num_y_ :]

        if self.data_ is not None:
            self.solution_y_ = self.data_[:, : self.num_y_]
            self.solution_x_ = self.data_[:, self.num_y_ :]
            # self.solution_x_ = self.data_[:, : self.num_x_ + 1]
            # self.solution_y_ = self.data_[:, self.num_x + 1 :]

        if self.use_splits_:
            all_val_samples = self.n_test + self.n_val
            self.x_train_ = self.x_[-(self.n_train + all_val_samples): -all_val_samples]
            self.x_val_ = self.x_[-all_val_samples : -self.n_test]
            self.x_test_ = self.x_[-all_val_samples :]
            self.y_train_ = self.y_[-(self.n_train + all_val_samples): -all_val_samples]
            self.y_val_ = self.y_[-all_val_samples : -self.n_test]
            self.y_test_ = self.y_[-all_val_samples :]

        else:
            self.x_train_ = self.x_  # [: (-self._test_ix_ - self._val_ix_)]
            self.x_val_ = self.x_  # [-(self._test_ix_ + self._val_ix_) : -self._test_ix_]
            self.x_test_ = self.x_  # [-self._test_ix_ :]
            self.y_train_ = self.y_  # [: (-self._test_ix_ - self._val_ix_)]
            self.y_val_ = self.y_  # [-(self._test_ix_ + self._val_ix_) : -self._test_ix_]
            self.y_test_ = self.y_  # [-self._test_ix_ :]

    def _load_x(self):
        return self.read_text_file(self.data_paths_["x"])

    def _load_y(self):
        return self.read_text_file(self.data_paths_["y"])

    def _load_model_params(self):
        return self.read_text_file(self.data_paths_["model"])

    def _load_model_data(self):
        return self.read_text_file(self.data_paths_["model"])

    def train_dataloader(self) -> DataLoader:
        x = torch.from_numpy(self.x_train_).double()
        y = torch.from_numpy(self.y_train_).double()
        return DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        x = torch.from_numpy(self.x_val_).double()
        y = torch.from_numpy(self.y_val_).double()
        return DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self) -> DataLoader:
        x = torch.from_numpy(self.x_test_).double()
        y = torch.from_numpy(self.y_test_).double()
        return DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=self.batch_size, shuffle=True
        )


class SemData(OldSemData):
    @staticmethod
    def get_model_paths(name, data_folder=OldSemData.DATA_FOLDER_):
        model_path = Path(data_folder) / name
        data = {
            "path": model_path,
            "model": model_path / "modelo.txt",
            "x": model_path / "X.txt",
            "y": model_path / "Y.txt",
            "solution_original": model_path / "solucion.txt",
            "solution_mc2e": model_path / "MC2E_sol.txt",
            "error": model_path / "MC2E_errores.txt",
        }
        return data

    def _load_data(self):
        super()._load_data()
        self.error_ref_ = self._load_error()
        self.data_mc2e_ = self._load_solution_mc2e()
        if self.data_mc2e_ is not None:
            self.solution_mc2e_y_ = self.data_mc2e_[:, : self.num_y_]
            self.solution_mc2e_x_ = self.data_mc2e_[:, self.num_y_ :]

    def _load_model_data(self):
        return self.read_text_file(self.data_paths_["solution_original"], decimal=".")

    def _load_solution_mc2e(self):
        try:
            return self.read_text_file(self.data_paths_["solution_mc2e"], decimal=".")
        except FileNotFoundError:
            return None

    def _load_error(self):
        try:
            return self.read_text_file(self.data_paths_["error"], decimal=".")
        except FileNotFoundError:
            return None


class ExcelData(OldSemData):
    @staticmethod
    def get_model_paths(name, data_folder=OldSemData.DATA_FOLDER_):
        model_path = Path(data_folder) / name
        data = {
            "path": model_path,
            "model": model_path / "modelo.txt",
            "x": model_path / "X.txt",
            "y": model_path / "Y.txt",
            "old_data": model_path / "datos.xlsx",
            "data": model_path / "datos_new.xlsx",
        }
        return data

    def _load_excel_values(self, sheet_name):
        return pd.read_excel(self.data_paths_["data"], sheet_name=sheet_name, header=None).values

    def _load_x(self):
        return self._load_excel_values("X")

    def _load_y(self):
        return self._load_excel_values("Y")

    def _load_model_params(self):
        return self._load_excel_values("modelo")

    def _load_model_data(self):
        return self._load_excel_values("solucion modelo")


class EcuacionSimultanea(LightningModule):
    def __init__(self, mask_exogena, mask_endogena, bias=None, init_model=True, lr=0.001):
        super().__init__()
        if isinstance(mask_exogena, np.ndarray):
            mask_exogena = torch.from_numpy(mask_exogena).double()
        if isinstance(mask_endogena, np.ndarray):
            mask_endogena = torch.from_numpy(mask_endogena).double()

        if bias is None:
            bias = einops.asnumpy(mask_exogena[-1])
            mask_exogena = mask_exogena[:-1].clone()
        mask_endogena = mask_endogena - torch.diag_embed(torch.diagonal(mask_endogena))
        self.exog_params = nn.Parameter(mask_exogena.double())
        self.exog_mask = nn.Parameter(mask_exogena.bool().double().detach(), requires_grad=False)
        self.endog_params = nn.Parameter(mask_endogena.double())
        self.eye = torch.eye(mask_endogena.size(0), device=mask_endogena.device)
        self.endog_mask = nn.Parameter(mask_endogena.bool().double().detach(), requires_grad=False)
        self.bias = nn.Parameter(torch.from_numpy(bias).double()).double()
        self.mask_exogena = mask_exogena
        self.mask_endogena = mask_endogena
        self.n_zeros_y = self.mask_endogena.sum()
        self.error = 0
        self.lr = lr
        if init_model:
            self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.endog_params.data)
        nn.init.xavier_uniform_(self.exog_params.data)
        self.bias.data.fill_(0.0)

    def get_error(self, x, y):
        y_hat = self.forward(x, y)
        return y_hat - y

    def forward(self, x, y):
        x = x[:, :-1]
        endog = self.endog_params * self.endog_mask  # - self.eye
        exog = self.exog_params * self.exog_mask
        y_hat = x @ exog + y @ endog + self.bias + self.error
        return y_hat

    def get_mats(self):
        endog = self.endog_params * self.endog_mask  # - self.eye
        exog = self.exog_params * self.exog_mask
        exog_sq = exog @ exog.T
        endog_sq = endog @ endog.T
        return exog_sq, endog_sq

    def loss(self, y, y_hat):
        # endog = self.endog_params * self.endog_mask
        # exog = self.exog_params * self.exog_mask
        # exog_sq = exog @ exog.T
        # endog_sq = endog @ endog.T

        # dist_x = MultivariateNormal(torch.zeros(exog_sq.shape[0]), torch.abs(exog_sq) + 1e-3)
        # dist_y = MultivariateNormal(torch.zeros(endog_sq.shape[0]), torch.abs(endog_sq) + 1e-3)
        # entropy_x = dist_x.entropy()
        # entropy_y = dist_y.entropy()

        return (
            torch.mean((y_hat - y) ** 2)  # - 1e-6 * entropy_x - 1e-6 * entropy_y
            # - 1e-3 * torch.abs(self.endog_params * self.endog_mask).sum()
            # - 1e-4 * torch.norm(self.endog_params * self.endog_mask)
            # - 1e-3 * torch.abs(self.exog_params * self.exog_mask).sum()
            # - 1e-4 * torch.norm(self.exog_params * self.exog_mask)
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, y)
        loss = self.loss(y=y, y_hat=y_hat)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, y)
        loss = self.loss(y=y, y_hat=y_hat)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, y)
        loss = self.loss(y=y, y_hat=y_hat)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


from pytorch_lightning.callbacks import ModelCheckpoint


def train_model(dataset_name, init_model, n_train, n_test, n_val, lr, max_epochs, seed):
    model_init = 'random' if init_model else 'mc2e'
    lr_name = "highlr" if lr == 0.01 else "lowlr"
    checkpoint_name = f"{dataset_name}_{model_init}_{n_test + n_val}_{lr_name}_{seed}"
    checkpoint_path = f"./modelos/{checkpoint_name}"
    data_path = Path(__file__).parent / f"exp_results/{checkpoint_name}.pkl"
    if data_path.exists():
        print(f"Skipping {checkpoint_name}")
        with open(data_path, "rb") as f:
            return pickle.load(f)
    bs = min(32, n_train, n_test, n_val)
    ds = SemData(dataset_name=dataset_name,
                  use_splits=True, n_train=n_train, n_val=n_val, n_test=n_test, batch_size=bs)
    mask_x, mask_y = ds.solution_x_.T, ds.solution_y_.T  # ds.mask_x_.T, ds.mask_y_.T
    model = EcuacionSimultanea(
        mask_exogena=mask_x, init_model=init_model, mask_endogena=mask_y, bias=None
    )
    pl.seed_everything(seed)
    # Set up model checkpoint to save best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Log hyperparameters and optimizer type, etc.

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=max_epochs,
        check_val_every_n_epoch=50,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback],
    )

    # Add MLFlow logging callback
    # trainer.callbacks.append(MLFlowCallback(log_models=False))
    trainer.fit(model, ds)
    chkpoint_path = Path(checkpoint_path) / os.listdir(checkpoint_path)[0]
    best = EcuacionSimultanea(mask_exogena=mask_x, mask_endogena=mask_y, bias=None)
    best.load_state_dict(torch.load(chkpoint_path)["state_dict"])
    test_result = trainer.test(best, ds.test_dataloader())
    #test_result = trainer.test(model, ds.test_dataloader())
    run_data = {"dataset": dataset_name,
                "model": model_init,
                 "n_train": n_train,
                 "n_test": n_test+ n_val,
                 "lr": lr, 
                 "max_epochs": max_epochs,
                "test_loss": test_result[0]["test_loss_epoch"]
                }

    with open(data_path, "wb") as f:
        pickle.dump(run_data, f)
    return run_data
    # Log final validation loss and checkpoint path
    # mlflow.log_metric("val_loss", trainer.callback_metrics.get("val_loss"))
    # mlflow.log_artifact(checkpoint_callback.best_model_path)

def main():
    init_model_vals = [False, True]#[False]
    lr_epoch_values = [(0.01, 2500), (0.00001, 4000)] #[(0.01, 5000)]#
    sample_values = [(3399, 300, 301)]#[(700, 100, 200), (70, 15, 15)]
    dataset_names = [#"modelo sigma 01menos varianza 2 4 1000",
                    "modelo sigma 01menos varianza 10 20 4000",
                    "modelo sigma 01menos varianza 20 40 8000",
                    #"modelo 2 4 1000",
                    "modelo 10 20 4000",
                    "modelo 20 40 8000",
                    ]
    
    all_results = []
    for i in range(1, 11):
        for init_model, (lr, max_epochs), (n_train, n_test, n_val), dataset_name in product(
            init_model_vals, lr_epoch_values, sample_values, dataset_names
        ):
            print(f"Training {dataset_name} with init_model={init_model}, lr={lr}, max_epochs={max_epochs} n_train={n_train} seed={i*1000}")
            result = train_model(dataset_name, init_model, n_train, n_test, n_val, lr, max_epochs, i*1000)
            all_results.append(result)
    with open(Path(__file__).parent / "exp_results/all_results_todos_datos_8k.pkl", "wb") as f:
        pickle.dump(all_results, f)

if __name__ == "__main__":
    main()
