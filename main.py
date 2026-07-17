import torch

from torch.cuda.amp import GradScaler

from training.checkpoint import load_checkpoint

from configs.config import (
    EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    MIN_DELTA
)

from training.metrics import (
    evaluate_map
)

from data.dataset import (
    get_datasets,
    get_dataloaders
)

from models.rtdetr_model import (
    load_model
)

from training.train import (
    train_one_epoch
)

from training.validate import (
    validate
)

from training.test import (
    test_model
)

from training.early_stopping import (
    EarlyStopping
)

from training.checkpoint import (
    save_checkpoint
)

train_history = []
val_history = []

def main():

    processor,\
    model,\
    device = load_model()

    train_dataset,\
    val_dataset,\
    test_dataset = get_datasets(
        processor
    )

    train_loader,\
    val_loader,\
    test_loader = get_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    scaler = GradScaler()

    early_stopping = EarlyStopping(
        patience=PATIENCE,
        min_delta=MIN_DELTA
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler
        )

        val_loss = validate(
            model,
            val_loader,
            device
        )

        train_history.append(
            train_loss
        )

        val_history.append(
            val_loss
        )
        scheduler.step(
            val_loss
        )
        
        val_metrics = evaluate_map(
            model,
            val_loader,
            processor,
            device
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                "checkpoints/best_model.pth"
            )

            print(
                "Best model saved."
            )

        if early_stopping(
            val_metrics.map50
        ):

            print(
                "Early stopping triggered."
            )

            break
    
    load_checkpoint(
    model,
    optimizer,
    "checkpoints/best_model.pth",
    device
    )
    test_loss = test_model(
        model,
        test_loader,
        device
    )

    print(
        f"\nTest Loss: {test_loss:.4f}"
    )


if __name__ == "__main__":
    main()