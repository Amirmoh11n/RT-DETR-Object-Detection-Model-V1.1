import torch

from torchmetrics.detection.mean_ap import (
    MeanAveragePrecision
)

from torchvision.ops import box_convert


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

        predictions = (
            processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=0.01
            )
        )

        preds = []
        targets = []

        for pred, target in zip(
            predictions,
            labels
        ):

            # -----------------------------
            # Predictions
            # -----------------------------

            pred_boxes = (
                pred["boxes"]
                .detach()
                .cpu()
            )

            pred_scores = (
                pred["scores"]
                .detach()
                .cpu()
            )

            pred_labels = (
                pred["labels"]
                .detach()
                .cpu()
            )

            preds.append(
                {
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels
                }
            )

            # -----------------------------
            # Ground Truth
            # -----------------------------

            gt_boxes = (
                target["boxes"]
                .clone()
                .cpu()
            )

            gt_boxes = box_convert(
                gt_boxes,
                in_fmt="cxcywh",
                out_fmt="xyxy"
            )

            image_height, image_width = (
                target["orig_size"]
                .cpu()
                .tolist()
            )

            gt_boxes[:, [0, 2]] *= image_width
            gt_boxes[:, [1, 3]] *= image_height

            targets.append(
                {
                    "boxes": gt_boxes,
                    "labels": (
                        target["class_labels"]
                        .cpu()
                    )
                }
            )

        # بسیار مهم
        metric.update(
            preds,
            targets
        )

    results = metric.compute()

    print(
        "Metric keys:",
        results.keys()
    )

    return {
        "mAP": results["map"].item(),
        "mAP50": results["map_50"].item(),
        "mAP75": results["map_75"].item(),
        "mAR300": (
            results["mar_300"].item()
            if "mar_300" in results
            else results["mar_100"].item()
        )
    }