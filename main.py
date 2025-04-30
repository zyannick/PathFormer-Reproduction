import argparse


argument = argparse.ArgumentParser("CoMRes Reproduction")
argument.add_argument("--mode", type=str, default="train", choices=['train', 'test'], help="Launch mode")
argument.add_argument("--model_name", type=str, default="CoMRes", choices=["PathFormer", "ComRes"], help="Model name")
argument.add_argument("--data_root", type=str, help="Path to the data root")
argument.add_argument("--dataset_name", type=str, default="weather",  help="Name of the dataset to use for the training and the testing")
argument.add_argument("--features", type=str, default="M", choices=["M", "S"], help="M:Multivariate or S:Univariate")
argument.add_argument("--data_root", type=str, help="Path to the data root")