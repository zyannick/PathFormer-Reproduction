#!/bin/bash


for pred_len in 96 192 336 720
do
    python train.py --pred_len $pred_len --loss_type charbonnier --config_file params/ETTh1.toml
done