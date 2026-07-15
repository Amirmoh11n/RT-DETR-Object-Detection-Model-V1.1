import torch

from tqdm import tqdm


@torch.no_grad()
def test_model(
    model,
    test_loader,
    device
):

    model.eval()

    running_loss = 0.0

    progress_bar = tqdm(
        test_loader,
        desc="Testing",
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

    test_loss = (
        running_loss
        /
        len(test_loader)
    )

    return test_loss