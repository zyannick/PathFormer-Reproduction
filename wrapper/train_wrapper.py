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
            "rse": R2Score(),
            "corr": ExplainedVariance(),
        }
    )
    return metrics


class Forecaster(pl.LightningModule):

    def __init__(self, model: nn.Module, config: ml_collections.ConfigDict):
        super(Forecaster, self).__init__()
        self.model = model
        self.config = config
        self.train_metrics = set_metrics()
        self.val_metrics = set_metrics()
        self.test_metrics = set_metrics()
        self.loss_fn = nn.MSELoss()
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
                optimizer, T_max=self.config.epochs, eta_min=0
            )
        elif self.config.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                total_steps=self.config.epochs,
                epochs=self.config.epochs,
                steps_per_epoch=self.config.steps_per_epoch,
            )

        elif self.config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
        }

    def model_prediction(self, batch):
        batch_x, batch_y = batch
        output = self(batch_x)
        prediction = output.get("prediction")
        balance_loss = output.get("balance_loss")

        f_dim = -1 if self.args.features == "MS" else 0

        prediction = prediction[:, -self.args.pred_len :, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
        y_pred = prediction.detach().cpu()
        y_truth = batch_y.detach().cpu()

        loss = self.loss_fn(y_pred, y_truth)

        if balance_loss is not None:
            loss += balance_loss

        return y_pred, y_truth, loss

    def training_step(self, batch, batch_idx):
        y_pred, y_truth, loss = self.model_prediction(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics(y_pred, y_truth), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y_truth, loss = self.model_prediction(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics(y_pred, y_truth), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_pred, y_truth, loss = self.model_prediction(batch)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics(y_pred, y_truth), prog_bar=True)
        return loss
