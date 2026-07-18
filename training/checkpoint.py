import os
import torch

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    path
):

    # Ensure directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        path
    )


def load_checkpoint(
    model,
    optimizer,
    path,
    device
):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

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

    return checkpoint["epoch"]

def load_model_weights(
    model,
    path,
    device
):
    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )