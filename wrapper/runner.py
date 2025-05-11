import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
import warnings

import torch
import argparse
import datetime
import hashlib
import json
import pickle
import random
import traceback
from glob import glob
import setproctitle

import ml_collections
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from modules.PathFormer import PathFormer
from pytorch_lightning.strategies import DDPStrategy
import multiprocessing as mp
import pytorch_lightning as pl

from wrapper.data_wrapper import DataModule
from wrapper.train_wrapper import Forecaster
from wrapper.utils_parser import merge_argument_and_configs
from wrapper.experiment_naming import generate_name


def runner():
    
    setproctitle.setproctitle("PathFormer")
    config = merge_argument_and_configs()

    pl.seed_everything(config.seed)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.model_id = f"{config.model_id}_{time_str}"
    config.run_name = generate_name()
    config.checkpoints = os.path.join(config.checkpoints, config.run_name)

    data_module = DataModule(config)
    model = PathFormer(config)
    
    config.train_steps = data_module.train_steps

    forecaster = Forecaster(model, config)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config.checkpoints,
        filename="PathFormer-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    logger = MLFlowLogger(
        experiment_name="PathFormer_" + config.data + "_" + str(config.pred_len),
        tracking_uri="sqlite:///mlflow/mlruns.db",
        artifact_location="mlflow/artifacts",
        run_name=config.run_name,
    )

    trainer = pl.Trainer(
        max_epochs=config.train_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        # log_every_n_steps=10,
        precision=16 if config.use_amp else 32,
    )

    trainer.fit(forecaster, data_module)
    trainer.test(forecaster, data_module, ckpt_path="best")
    
#train.py --pred_len 96 --config_file /home/yzoetgna/projects/PathFormer-Reproduction/params/ETTh1.toml 
