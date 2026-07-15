import torch

from tqdm import tqdm


@torch.no_grad()
def validate(model,val_loader,device):

    model.eval()

    running_loss = 0.0

    progress_bar = tqdm(
        val_loader,
        desc="Validation",
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

        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss

        running_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    val_loss = (
        running_loss
        /
        len(val_loader)
    )

    return val_loss