"""
This module contains functions and utilities for training and validating
generative Variational Autoencoder (VAE) models.

Functions:
    train: Trains the model for one epoch.
    validate: Validates the model.
    save_checkpoint: Saves the model checkpoint.
    train_vae: Trains the VAE model over multiple epochs, validates it,
                and saves checkpoints.
"""

import torch
import torch.nn as nn
from livelossplot import PlotLosses
from tqdm import tqdm


def train(
    model,
    device,
    data_loader,
    optimizer,
    tepoch,
    curr_epoch,
    n_epochs,
    logs: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        device (str): Device to perform computations on.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the
                                                        training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model
                                                parameters.
        tepoch (tqdm.tqdm): tqdm progress bar object.
        curr_epoch (int): Current epoch number.
        n_epochs (int): Total number of epochs.
        logs (bool, optional): Whether to log the progress. Defaults to True.

    Returns:
        tuple: Average training loss, mean squared error, and KL divergence
        over the dataset.
    """
    model.train()
    train_loss, mse_loss, kl_loss = 0, 0, 0
    for batch_idx, X in enumerate(data_loader, start=1):
        X = X.float().to(device)
        optimizer.zero_grad()
        X_recon, kl_div = model(X)

        mse = (
            (X - X_recon) ** 2
        ).sum()  # TODO: put this in a separate function def calculate MSE().
        loss = mse + kl_div
        loss.backward()

        train_loss += loss.item()
        mse_loss += mse.item()
        kl_loss += kl_div.item()

        optimizer.step()

        if logs:
            tepoch.set_description(
                f"Epoch: {curr_epoch}/{n_epochs} | Batch: {batch_idx}/{len(data_loader)}"
            )
            tepoch.set_postfix(loss=loss.item() / X.size(0), refresh=False)
            tepoch.update(1)

    return (
        train_loss / len(data_loader.dataset),
        mse_loss / len(data_loader.dataset),
        kl_loss / len(data_loader.dataset),
    )


def validate(model, data_loader, device):
    """
    Validate the model.

    Args:
        model (torch.nn.Module): The model to validate.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the
                                                    validation data.
        device (str): Device to perform computations on.

    Returns:
        tuple: Average validation loss, mean squared error, and KL divergence
                over the dataset.
    """
    model.eval()
    val_loss, mse_loss, kl_loss = 0, 0, 0
    with torch.no_grad():
        for batch_idx, X in enumerate(data_loader):
            X = X.float().to(device)
            X_recon, kl_div = model(X)

            mse = (
                (X - X_recon) ** 2
            ).sum()  # TODO: same as train(), abstract into func.
            loss = mse + kl_div

            val_loss += loss.item()
            mse_loss += mse.item()
            kl_loss += kl_div.item()

    return (
        val_loss / len(data_loader.dataset),
        mse_loss / len(data_loader.dataset),
        kl_loss / len(data_loader.dataset),
    )


def save_checkpoint(
    model_weights, optimizer_info, model_save_path, epoch, train_loss, val_loss
):
    """
    Save the model checkpoint.

    Args:
        model_weights (dict): Model's state dictionary.
        optimizer_info (dict): Optimizer's state dictionary.
        model_save_path (str): Path to save the model checkpoint.
        epoch (int): Current epoch number.
        train_loss (float): Training loss at the current epoch.
        val_loss (float): Validation loss at the current epoch.
    """
    torch.save(
        obj={
            "model_state_dict": model_weights,
            "optimiser_state_dict": optimizer_info,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        f=model_save_path,
    )


def train_vae(
    n_epochs,
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    model_save_path,
    use_liveloss,
    device,
):
    """
    Train the VAE model.

    Args:
        n_epochs (int): Number of epochs to train.
        model (torch.nn.Module): The VAE model to train.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model
                                                parameters.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate
                                                        scheduler.
        train_loader (torch.utils.data.DataLoader): DataLoader providing the
                                                    training data.
        val_loader (torch.utils.data.DataLoader, optional): DataLoader
                                            providing the validation data.
        model_save_path (str, optional): Path to save the model checkpoints.
        use_liveloss (bool, optional): Whether to use livelossplot for logging.
                                        Defaults to False.
        device (str): Device to perform computations on.

    Returns:
        tuple: Trained model, training losses, and validation losses.
    """
    liveloss = PlotLosses()

    train_losses, train_mse_losses, train_kldiv_losses = [], [], []
    val_losses, val_mse_losses, val_kldiv_losses = [], [], []
    with tqdm(
        total=len(train_loader) * n_epochs, desc="Training", unit="batch"
    ) as tepoch:
        for i in range(n_epochs):
            logs = {}
            _train_losses = train(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                tepoch=tepoch,
                curr_epoch=i,
                n_epochs=n_epochs,
                device=device,
                logs=True,
            )

            train_losses.append(_train_losses[0])
            train_mse_losses.append(_train_losses[1])
            train_kldiv_losses.append(_train_losses[2])
            logs["train loss"] = _train_losses[0]  # for liveloss.

            if val_loader is not None:
                _val_losses = validate(
                    model=model,
                    data_loader=val_loader,
                    device=device,
                )
                val_losses.append(_val_losses[0])
                val_mse_losses.append(_val_losses[1])
                val_kldiv_losses.append(_val_losses[2])
                logs["val_train loss"] = _val_losses[0]  # for liveloss.

            # save a checkpoint.
            if model_save_path is not None:
                if _train_losses[0] <= min(train_losses):
                    save_checkpoint(
                        model_weights=model.state_dict(),
                        optimizer_info=optimizer.state_dict(),
                        model_save_path=model_save_path,
                        epoch=i,
                        train_loss=_train_losses[0],
                        val_loss=_val_losses[0],
                    )
            if use_liveloss:
                liveloss.update(logs)
                liveloss.send()

            if scheduler is not None:
                scheduler.step(_train_losses[0])

    return (
        model,
        (train_losses, train_mse_losses, train_kldiv_losses),
        (val_losses, val_mse_losses, val_kldiv_losses),
    )
