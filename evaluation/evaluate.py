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
    print(metrics.keys())
    return {
        "map": metrics["map"],
        "map50": metrics["map_50"],
        "map75": metrics["map_75"],
        "mar300": metrics["mar_300"]
    }