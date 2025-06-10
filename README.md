# Pathformer Reproduction and Extension

This project involves reproducing the results of the ICLR 2024 paper: "[Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting](https://arxiv.org/abs/2402.05956v5)" and exploring potential extensions and improvements.

## Usage

To use and run experiments:

```
conda create -n path_former python=3.12 -y
conda activate path_former
pip install -r requirements.txt
```

Depending on you gpu you can update the version of pytorch/pytorch_ligthning.

To launch the training:

```
python train.py --pred_len 96 --config_file params/ETTh1.toml 
```

You can also launch training of experiment through scripts like:

```
./scripts/ETTh1.sh
```

## Analysis and Reproduction Goals

The primary goal is to thoroughly understand and validate the Pathformer model. Key analysis steps include:

* **Reproducing Core Results:** Replicate the results of Table 1, 2 and 3
* **Hyperparameter Optimization:** Finetuning performance using `Optuna`
* **Experiment Tracking:** Employ `Mlflow` to log parameters, metrics (MAE, MSE), training times, and visualizations for reproducibility and rigorous comparison.
* **Sensitivity Analysis:** Investigate the model's sensitivity to parameters like the number of selected pathways (K, Table 4), the pool of available patch sizes, and input sequence length (H).
* **Visualization & Error Analysis:** Replicate pathway weight visualizations (Figure 4) and prediction plots (Figure 6) to understand model behavior and identify areas for improvement.
* **Large Dataset Validation:** Test performance on larger datasets (PEMS07, Wind Power, Table 10) mentioned in the paper's appendix.

## Results


## Futur work
