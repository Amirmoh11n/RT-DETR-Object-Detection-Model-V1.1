import os
import torch


def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    val_loss,
    path
):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict()
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Checkpoint saved: {path}"
    )


def load_checkpoint(
    model,
    optimizer,
    scaler,
    path,
    device
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    if (
        scaler is not None
        and
        "scaler_state_dict" in checkpoint
    ):

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    print(
        f"Checkpoint loaded: {path}"
    )

    return {
        "epoch": checkpoint["epoch"],
        "val_loss": checkpoint["val_loss"]
    }


def load_model_weights(
    model,
    path,
    device
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Model weights loaded: {path}"
    )

    return model