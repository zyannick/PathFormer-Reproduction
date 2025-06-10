import argparse
import torch
import sys
import tomllib
from ml_collections import ConfigDict
import numpy as np


class SmartArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        args, argv = super().parse_known_args(args, namespace)
        args._specified_args = set()

        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.startswith("--"):
                if "=" in arg:
                    key = arg.split("=")[0]
                    args._specified_args.add(key)
                    i += 1
                else:
                    key = arg
                    args._specified_args.add(key)
                    i += 2
            else:
                i += 1
        return args, argv


def full_parse_args():
    parser = SmartArgumentParser(description="Multivariate Time Series Forecasting")

    # basic config
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument(
        "--model",
        type=str,
        default="PathFormer",
        help="model name, options: [PathFormer]",
    )
    parser.add_argument("--model_id", type=str, default="ETT.sh")
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1024, help="random seed")

    # data loader
    parser.add_argument("--data", type=str, default="custom", help="dataset type")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./dataset/weather",
        help="root path of the data file",
    )
    parser.add_argument(
        "--data_path", type=str, default="weather.csv", help="data file"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S]; M:multivariate predict multivariate, S:univariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="optimizer, options: [adam, adamw]",
    )
    
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        help="loss type, options: [mse, charbonnier]",
    )
    
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        help="scheduler, options: [cosine, onecycle, plateau]",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument(
        "--low_frequency", type=int, default=1, help="low frequency for seasonality"
    )
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        default=False,
        help="DLinear: a linear layer for each variate(channel) individually",
    )
    parser.add_argument(
        "--list_kernel_size_trend",
        type=list,
        default=[4, 8, 12],
        help="kernel size for trend",
    )
    # model
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--num_nodes", type=int, default=21)
    parser.add_argument("--layer_nums", type=int, default=3)
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="choose the Top K patch size at the every layer ",
    )

    parser.add_argument(
        "--noisy_gating",
        type=bool,
        default=True,
        help="use noisy gating"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.00005,
        help="weight decay for Adam optimizer",
    )

    parser.add_argument(
        "--k_seasonality",
        type=int,
        default=3,
        help="choose the Top K patch size at the every layer ",
    )
    parser.add_argument("--num_experts_list", type=list, default=[4, 4, 4])
    parser.add_argument(
        "--patch_size_list",
        nargs="+",
        type=int,
        default=[16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2],
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="whether to predict unseen future data",
    )
    parser.add_argument("--revin", type=int, default=1, help="whether to apply RevIN")
    parser.add_argument("--drop", type=float, default=0.1, help="dropout ratio")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--residual_connection", type=int, default=0)
    parser.add_argument("--metric", type=str, default="mae")
    parser.add_argument("--batch_norm", type=int, default=0)

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=4, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=50, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="optimizer learning rate"
    )
    parser.add_argument("--lradj", type=str, default="TST", help="adjust learning rate")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )
    parser.add_argument("--pct_start", type=float, default=0.4, help="pct_start")
    parser.add_argument(
        "--factorized", type=bool, default=True, help="use factorized routing"
    )
    parser.add_argument(
        "--dynamic", type=bool, default=False, help="use dynamic routing"
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="2", help="device ids of multile gpus"
    )
    parser.add_argument(
        "--test_flop",
        action="store_true",
        default=False,
        help="See utils/tools for usage",
    )

    return parser


def merge_argument_and_configs() -> ConfigDict:
    parser = full_parse_args()
    args, _ = parser.parse_known_args()
    
    

    final_params = {}

    if args.config_file is not None:
        with open(args.config_file, "rb") as f:
            config = tomllib.load(f)
        common_settings = {k: v for k, v in config.items() if k != "runs"}
        all_runs_configs = config.get("runs", [])

        if not all_runs_configs:
            print(
                f"Error: No '[[runs]]' sections found in '{args.config_file}'",
                file=sys.stderr,
            )
            sys.exit(1)

        target_pred_len = args.pred_len

        target_run_config = None
        for run_config in all_runs_configs:
            if run_config.get("pred_len") == target_pred_len:
                target_run_config = run_config
                break

        if target_run_config is None:
            print(
                f"Error: No run configuration found for pred_len={target_pred_len} in '{args.config_file}'",
                file=sys.stderr,
            )
            sys.exit(1)

        final_params = common_settings.copy()
        final_params.update(target_run_config)

    config = ConfigDict(vars(args))
    for k, v in final_params.items():
        if f"--{k}" not in args._specified_args:
            config[k] = v
            
    config.patch_size_list = np.array(config.patch_size_list).reshape(config.layer_nums, -1).tolist()

    return config
