import numpy as np
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn


class EcuacionSimultanea(LightningModule):
    def __init__(self, mask_exogena, mask_endogena, bias=None):
        super().__init__()
        if isinstance(mask_exogena, np.ndarray):
            mask_exogena = torch.from_numpy(mask_exogena).float()
        if isinstance(mask_endogena, np.ndarray):
            mask_endogena = torch.from_numpy(mask_endogena).float()
        self.exog_params = nn.Parameter(torch.randn_like(mask_exogena.float()))
        self.exog_mask = nn.Parameter(mask_exogena.bool().float(), requires_grad=False)
        mask_endogena = (
            mask_endogena  # - torch.eye(mask_endogena.shape(0), device=mask_endogena.device)
        )
        self.endog_params = nn.Parameter(torch.randn_like(mask_endogena.float()))
        self.endog_mask = nn.Parameter(mask_endogena.bool().float(), requires_grad=False)
        self.bias = 0 if bias is None else nn.Parameter(torch.from_numpy(bias))
        self.error = 0

    def get_error(self, x, y):
        y_hat = self.forward(x, y)
        return y_hat - y

    def forward(self, x, y):
        endog = self.endog_params * self.endog_mask
        exog = self.exog_params * self.exog_mask
        y_hat = x @ exog + y @ endog + self.bias + self.error
        return y_hat

    def loss(self, y, y_hat):
        return torch.mean((y_hat - y) ** 2)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, y)
        loss = self.loss(y=y, y_hat=y_hat)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
