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
    dataset: str,
    target_label: int,
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
        dataset=dataset,
        data_dir=data_dir,
        normal_label=target_label,
        normalize=True,
        setting=setting,
    )
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
    logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri)
    logger.log_hyperparams(
        {
            "model": model,
            "dataset": dataset,
            "setting": setting,
            "target_label": target_label,
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
    trainer.fit(auto_encoder, data_module)
    rapp = RaPP(
        model=auto_encoder,
        rapp_start_index=rapp_start_index,
        rapp_end_index=rapp_end_index,
        loss_reduction=loss_reduction,
    )
    rapp.fit(data_module.train_dataloader())
    result = rapp.test(data_module.test_dataloader())
    logger.log_metrics(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ae", choices=["ae", "aae", "vae"])
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FashionMNIST", "CIFAR10"])
    parser.add_argument("--target_label", type=int, default=0, help="useful only when setting=SIMO")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--hidden_size", type=int, default=20)
    # number of neurons of the layer between the encoder and the decoder
    parser.add_argument("--n_layers", type=int, default=10)
    # number of layers on either side, total = n_layers * 2
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--tracking_uri", type=str, default="file:./mlruns")
    parser.add_argument("--n_trial", type=int, default=0)
    parser.add_argument("--setting", type=str, default="SIMO", choices=["SIMO", "inter_set", "set_to_set"])
    parser.add_argument("--rapp_start_index", type=int, default=0)
    parser.add_argument("--rapp_end_index", type=int, default=-1)
    parser.add_argument(
        "--loss_reduction", type=str, default="sum", choices=["sum", "mean"]
    )
    args = parser.parse_args()

    main(
        model=args.model,
        dataset=args.dataset,
        target_label=args.target_label,
        data_dir=args.data_dir,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        max_epochs=args.max_epochs,
        experiment_name=f"{args.setting}_{args.dataset}_{args.model}_{args.target_label}",
        tracking_uri=args.tracking_uri,
        n_trial=args.n_trial,
        setting=args.setting,
        loss_reduction=args.loss_reduction,
        rapp_start_index=args.rapp_start_index,
        rapp_end_index=args.rapp_end_index,
    )
