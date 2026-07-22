import torch

from training.metrics import evaluate_map


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    processor,
    device
):

    model.eval()

    metrics = evaluate_map(
        model=model,
        dataloader=dataloader,
        processor=processor,
        device=device
    )

    return metrics