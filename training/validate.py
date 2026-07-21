from .metrics import evaluate_map


def evaluate(
    model,
    dataloader,
    processor,
    device
):
    """
    Run full evaluation pipeline.

    Returns:
        dict:
            map
            map50
            map75
            mar100
    """

    metrics = evaluate_map(
        model=model,
        dataloader=dataloader,
        processor=processor,
        device=device
    )

    return metrics