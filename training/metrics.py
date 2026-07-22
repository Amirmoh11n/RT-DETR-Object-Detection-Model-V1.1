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


        # Inference only
        outputs = model(
            pixel_values=pixel_values
        )


        # Original image sizes
        target_sizes = torch.stack(
            [
                t["orig_size"]
                for t in labels
            ]
        ).to(device)


        # Convert model outputs to xyxy pixel coordinates
        predictions = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.01
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

            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            pred_labels = pred["labels"].cpu()


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

            gt_boxes = target["boxes"].clone()


            # RT-DETR boxes:
            # cx,cy,w,h  (normalized)
            #
            # TorchMetrics needs:
            # xmin,ymin,xmax,ymax (pixels)

            gt_boxes = box_convert(
                gt_boxes,
                in_fmt="cxcywh",
                out_fmt="xyxy"
            )


            # Convert normalized coordinates to pixels

            image_height, image_width = (
                target["orig_size"]
                .cpu()
                .tolist()
            )


            gt_boxes[:, [0, 2]] *= image_width

            gt_boxes[:, [1, 3]] *= image_height


            targets.append(
                {
                    "boxes": gt_boxes.cpu(),
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

        "mar300":
            results["mar_300"].item()

    }