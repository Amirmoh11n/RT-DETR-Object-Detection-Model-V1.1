import torch

from tqdm import tqdm


def train_one_epoch(model,train_loader,optimizer,device,scaler):

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
                for k, v in t.items()
            }
            for t in batch["labels"]
        ]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            outputs = model(
                pixel_values=pixel_values,
                labels=labels
            )

            loss = outputs.loss

        scaler.scale(loss).backward()

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