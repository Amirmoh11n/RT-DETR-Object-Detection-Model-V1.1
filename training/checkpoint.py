import os
import torch



def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    val_loss,
    path,
    scheduler=None,
    metrics=None
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

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict()
    }


    if scaler is not None:

        checkpoint[
            "scaler_state_dict"
        ] = scaler.state_dict()


    if scheduler is not None:

        checkpoint[
            "scheduler_state_dict"
        ] = scheduler.state_dict()


    if metrics is not None:

        checkpoint[
            "metrics"
        ] = metrics



    torch.save(
        checkpoint,
        path
    )


    print(
        f"Checkpoint saved: {path}"
    )



def load_checkpoint(
    model,
    optimizer=None,
    scaler=None,
    scheduler=None,
    path=None,
    device="cpu"
):

    if path is None:

        raise ValueError(
            "Checkpoint path is required"
        )


    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )


    checkpoint = torch.load(
        path,
        map_location=device
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )



    if (
        optimizer is not None
        and
        "optimizer_state_dict" in checkpoint
    ):

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )



    if (
        scaler is not None
        and
        "scaler_state_dict" in checkpoint
    ):

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )



    if (
        scheduler is not None
        and
        "scheduler_state_dict" in checkpoint
    ):

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )



    print(
        f"Checkpoint loaded: {path}"
    )


    return {

        "epoch":
            checkpoint.get(
                "epoch",
                0
            ),

        "val_loss":
            checkpoint.get(
                "val_loss",
                None
            ),

        "metrics":
            checkpoint.get(
                "metrics",
                {}
            )
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
        checkpoint[
            "model_state_dict"
        ]
    )


    print(
        f"Model weights loaded: {path}"
    )


    return model