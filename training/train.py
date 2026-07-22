import torch

from tqdm import tqdm


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    scaler
):

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for batch in progress_bar:

        pixel_values = batch[
            "pixel_values"
        ].to(device)

        labels = [
            {
                k: v.to(device)
                for k, v in target.items()
            }
            for target in batch["labels"]
        ]

        optimizer.zero_grad()

        with torch.amp.autocast(
                "cuda",
                enabled=device.type == "cuda"
        ):

            outputs = model(
                pixel_values=pixel_values,
                labels=labels
            )

            loss = outputs.loss

        if not torch.isfinite(loss):

            print(
                f"Invalid loss detected: {loss}"
            )

            continue

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss
        /
        len(train_loader)
    )

    return epoch_loss