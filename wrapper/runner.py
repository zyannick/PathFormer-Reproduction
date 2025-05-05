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
from wrapper.utils_parser import parse_args


def runner():
    args = parse_args()
    config = ml_collections.ConfigDict()
    config.update(vars(args))

    pl.seed_everything(config.seed)

    data_module = DataModule(config)
    model = PathFormer(config)

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
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    logger = MLFlowLogger(
        experiment_name="PathFormer",
        tracking_uri="file:./mlruns",
        run_name=f"{config.model_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )

    trainer = pl.Trainer(
        max_epochs=config.train_epochs,
        accelerator="gpu" if config.use_gpu else "cpu",
        devices=config.devices if config.use_multi_gpu else 1,
        strategy=(
            DDPStrategy(find_unused_parameters=False) if config.use_multi_gpu else None
        ),
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        log_every_n_steps=10,
    )

    trainer.fit(forecaster, data_module)
    if config.do_test:
        trainer.test(forecaster, data_module, ckpt_path="best")



