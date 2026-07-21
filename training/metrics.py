import torch

from torchmetrics.detection.mean_ap import (
    MeanAveragePrecision
)


@torch.no_grad()
def evaluate_map(
    model,
    dataloader,
    processor,
    device
):

    model.eval()


    metric = MeanAveragePrecision(
        max_detection_thresholds=[1, 10, 300]
    )


    for batch in dataloader:


        pixel_values = batch[
            "pixel_values"
        ].to(device)


        labels = batch["labels"]


        outputs = model(
            pixel_values=pixel_values
        )


        target_sizes = torch.stack(
            [
                t["orig_size"]
                for t in labels
            ]
        ).to(device)


        predictions = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.0
        )


        preds = []
        targets = []


        for pred, target in zip(
            predictions,
            labels
        ):


            preds.append(
                {
                    "boxes":
                        pred["boxes"].cpu(),

                    "scores":
                        pred["scores"].cpu(),

                    "labels":
                        pred["labels"].cpu()
                }
            )


            targets.append(
                {
                    "boxes":
                        target["boxes"].cpu(),

                    "labels":
                        target["class_labels"].cpu()
                }
            )


        metric.update(
            preds,
            targets
        )


    results = metric.compute()


    return {
        "map":
            results["map"].item(),

        "map50":
            results["map_50"].item(),

        "map75":
            results["map_75"].item(),

        "mar100":
            results["mar_100"].item()
    }