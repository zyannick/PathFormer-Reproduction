import pytorch_lightning as pl
import torch.nn as nn
import ml_collections
from torchmetrics import (
    MeanSquaredError,
    R2Score,
    ExplainedVariance,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    SymmetricMeanAbsolutePercentageError,
)
from torchmetrics import MetricCollection
from torchmetrics import Metric
import torch
from typing import List, Tuple


def set_metrics():
    metrics = MetricCollection(
        {
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "mspe": SymmetricMeanAbsolutePercentageError(),
            "mape": MeanAbsolutePercentageError(),
            "corr": ExplainedVariance(),
        }
    )
    return metrics

import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon_squared = epsilon ** 2

    def forward(self, prediction, target):
        diff = prediction - target
        loss = torch.sqrt(torch.addcmul(torch.tensor(self.epsilon_squared, device=diff.device), diff, diff))
        return loss.mean()
    
class CharbonnierLossWeighted(nn.Module):
    def __init__(self, epsilon=1e-3, forecast_steps=12):
        super(CharbonnierLossWeighted, self).__init__()
        self.epsilon_squared = epsilon ** 2
        self.forecast_steps = forecast_steps

    def forward(self, prediction, target):
        diff = prediction - target
        weights = torch.linspace(1.0, 0.1, steps=self.forecast_steps).to(prediction.device)
        loss = torch.sqrt(((diff) ** 2 + self.epsilon_squared)) * weights
        return loss.mean()


class Forecaster(pl.LightningModule):

    def __init__(self, model: nn.Module, config: ml_collections.ConfigDict):
        super(Forecaster, self).__init__()
        self.model = model
        self.config = config
        self.train_metrics = set_metrics()
        self.val_metrics = set_metrics()
        self.test_metrics = set_metrics()
        if self.config.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.config.loss_type == "charbonnier":
            self.loss_fn = CharbonnierLoss()
        elif self.config.loss_type == "charbonnier_weighted":
            self.loss_fn = CharbonnierLossWeighted(
                forecast_steps=self.config.pred_len
            )
        self.save_hyperparameters(config.to_dict())

    def forward(self, signal):
        return self.model(signal)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=None,
        **kwargs,
    ):
        return super().load_from_checkpoint(
            checkpoint_path, map_location, hparams_file, strict, **kwargs
        )

    def configure_optimizers(self):
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        if self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.train_epochs, eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                steps_per_epoch=self.config.train_steps,
                pct_start=self.config.pct_start,
                epochs=self.config.train_epochs,
                max_lr=self.config.learning_rate,
            )

        elif self.config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def model_prediction(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        
        batch_x = batch_x.to(torch.float32)
        batch_y = batch_y.to(torch.float32)
        
        output = self(batch_x)
        y_pred = output.get("prediction")
        balance_loss = output.get("balance_loss")

        f_dim = -1 if self.config.features == "MS" else 0

        y_pred = y_pred[:, -self.config.pred_len :, f_dim:]
        y_truth = batch_y[:, -self.config.pred_len :, f_dim:].to(self.device)

        loss = self.loss_fn(y_pred, y_truth)

        if balance_loss is not None:
            loss += balance_loss
            

        return y_pred, y_truth, loss

    def training_step(self, batch, batch_idx):
        y_pred, y_truth, loss = self.model_prediction(batch)
        batch_size = y_pred.shape[0]
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, metric in self.train_metrics.items():
            y_pred = y_pred.contiguous().view(batch_size, -1)
            y_truth = y_truth.contiguous().view(batch_size, -1)
            metric(y_pred, y_truth)
            self.log(f"train_{name}", metric, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y_truth, loss = self.model_prediction(batch)
        batch_size = y_pred.shape[0]
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, metric in self.val_metrics.items():
            y_pred = y_pred.contiguous().view(batch_size, -1)
            y_truth = y_truth.contiguous().view(batch_size, -1)
            metric(y_pred, y_truth)
            self.log(f"val_{name}", metric, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_pred, y_truth, loss = self.model_prediction(batch)
        batch_size = y_pred.shape[0]
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, metric in self.test_metrics.items():
            y_pred = y_pred.contiguous().view(batch_size, -1)
            y_truth = y_truth.contiguous().view(batch_size, -1)
            metric(y_pred, y_truth)
            self.log(f"test_{name}", metric, on_step=False, on_epoch=True, prog_bar=True)
        return loss
