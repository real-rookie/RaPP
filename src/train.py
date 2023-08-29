import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from rapp.data import DataModule
from rapp.models import (
    AutoEncoder,
    AdversarialAutoEncoder,
    VariationalAutoEncoder,
    RaPP,
)


def main(
    model: str,
    dataset_normal: str,
    dataset_novel: str,
    normal_label: int,
    data_dir: str,
    hidden_size: int,
    n_layers: int,
    max_epochs: int,
    experiment_name: str,
    tracking_uri: str,
    n_trial: int,
    setting: str,
    loss_reduction: str,
    rapp_start_index: int,
    rapp_end_index: int,
):

    data_module = DataModule(
        dataset_normal=dataset_normal,
        dataset_novel=dataset_novel,
        data_dir=data_dir,
        normal_label=normal_label,
        setting=setting,
    )
    print("--------after define datamodule-----------")
    if model == "ae":
        auto_encoder = AutoEncoder(
            input_size=data_module.image_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            loss_reduction=loss_reduction,
        )
    elif model == "vae":
        auto_encoder = VariationalAutoEncoder(
            input_size=data_module.image_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            loss_reduction=loss_reduction,
        )
    elif model == "aae":
        auto_encoder = AdversarialAutoEncoder(
            input_size=data_module.image_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            d_layers=n_layers,
            loss_reduction=loss_reduction,
        )
    else:
        raise ValueError(f"Not valid model name {model}")
    print("--------after define autoencoder-----------")
    logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri)
    print("--------after define logger-----------")
    logger.log_hyperparams(
        {
            "model": model,
            "dataset_normal": dataset_normal,
            "dataset_novel": dataset_novel,
            "setting": setting,
            "normal_label": normal_label,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "max_epochs": max_epochs,
            "n_trial": n_trial,
            "loss_reduction": loss_reduction,
            "rapp_start_index": rapp_start_index,
            "rapp_end_index": rapp_end_index,
        }
    )
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(logger=logger, max_epochs=max_epochs, gpus=gpus)
    print("-------before fit--------")
    trainer.fit(auto_encoder, data_module)
    print("-------after fit--------")
    rapp = RaPP(
        model=auto_encoder,
        rapp_start_index=rapp_start_index,
        rapp_end_index=rapp_end_index,
        loss_reduction=loss_reduction,
    )
    print("-------before rapp fit--------")
    rapp.fit(data_module.train_dataloader())
    print("-------after rapp fit--------")
    print("-------before rapp test--------")
    result = rapp.test(data_module.test_dataloader())
    print("-------after rapp test--------")
    logger.log_metrics(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ae", choices=["ae", "aae", "vae"])
    parser.add_argument("--dataset_normal", type=str, default="MNIST",
                        choices=["MNIST", "FashionMNIST", "CIFAR10"])
    parser.add_argument("--dataset_novel", type=str, default="MNIST",
                        choices=["MNIST", "FashionMNIST", "CIFAR10"],
                        help="useful only when setting is set_to_set")
    parser.add_argument("--normal_label", type=int, default=0,
                        help="useful only when setting is SIMO")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--hidden_size", type=int, default=20)
    # number of neurons of the layer between the encoder and the decoder
    parser.add_argument("--n_layers", type=int, default=10)
    # number of layers on either side, total = n_layers * 2
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--tracking_uri", type=str, default="file:./mlruns")
    parser.add_argument("--n_trial", type=int, default=0)
    parser.add_argument("--setting", type=str, default="SIMO",
                        choices=["SIMO", "inter_set", "set_to_set"])
    parser.add_argument("--rapp_start_index", type=int, default=0)
    parser.add_argument("--rapp_end_index", type=int, default=-1)
    parser.add_argument("--loss_reduction", type=str, default="sum",
                        choices=["sum", "mean"])
    args = parser.parse_args()
    assert args.dataset_normal != args.dataset_novel
    experiment_name = None
    if args.setting == "SIMO":
        experiment_name = f"{args.setting}_{args.dataset_normal}_{args.model}_{args.normal_label}"
    elif args.setting == "inter_set":
        experiment_name = f"{args.setting}_{args.dataset_normal}_{args.model}"
    elif args.setting == "set_to_set":
        experiment_name = f"{args.setting}_{args.dataset_normal}_{args.model}_{args.dataset_novel}"
    assert experiment_name is not None
    main(
        model=args.model,
        dataset_normal=args.dataset_normal,
        dataset_novel=args.dataset_novel,
        normal_label=args.normal_label,
        data_dir=args.data_dir,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        max_epochs=args.max_epochs,
        experiment_name=experiment_name,
        tracking_uri=args.tracking_uri,
        n_trial=args.n_trial,
        setting=args.setting,
        loss_reduction=args.loss_reduction,
        rapp_start_index=args.rapp_start_index,
        rapp_end_index=args.rapp_end_index,
    )
