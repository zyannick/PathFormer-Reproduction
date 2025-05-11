


for pred_len in 96 192 336 720
do
    python train.py --pred_len $pred_len --config_file /home/yzoetgna/projects/PathFormer-Reproduction/params/ETTh1.toml
done