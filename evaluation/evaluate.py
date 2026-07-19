import torch

from tqdm import tqdm

from models.rtdetr_model import load_model

from evaluation.metrics import (
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    calculate_iou
)

from data.dataset import (
    get_datasets,
    get_dataloaders
)

IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5


@torch.no_grad()
def evaluate():

    processor, model, device = load_model()

    train_dataset, val_dataset, test_dataset = get_datasets(
        processor
    )

    _, _, test_loader = get_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset
    )

    model.eval()

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    progress_bar = tqdm(
        test_loader,
        desc="Evaluating"
    )

    for batch in progress_bar:

        pixel_values = batch[
            "pixel_values"
        ].to(device)

        labels = batch[
            "labels"
        ]

        outputs = model(
            pixel_values=pixel_values
        )

        target_sizes = torch.tensor(
            [
                (
                    pixel_values.shape[2],
                    pixel_values.shape[3]
                )
                for _ in range(
                    pixel_values.shape[0]
                )
            ]
        ).to(device)

        results = processor.post_process_object_detection(
            outputs,
            threshold=CONFIDENCE_THRESHOLD,
            target_sizes=target_sizes
        )

        for prediction, ground_truth in zip(
            results,
            labels
        ):

            pred_boxes = prediction[
                "boxes"
            ].cpu()

            pred_labels = prediction[
                "labels"
            ].cpu()

            gt_boxes = ground_truth[
                "boxes"
            ].cpu()

            gt_labels = ground_truth[
                "class_labels"
            ].cpu()

            matched_gt = set()

            for pred_box, pred_label in zip(
                pred_boxes,
                pred_labels
            ):

                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, (
                    gt_box,
                    gt_label
                ) in enumerate(
                    zip(
                        gt_boxes,
                        gt_labels
                    )
                ):

                    if gt_idx in matched_gt:
                        continue

                    if int(pred_label) != int(gt_label):
                        continue

                    iou = calculate_iou(
                        pred_box,
                        gt_box
                    )

                    if iou > best_iou:

                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= IOU_THRESHOLD:

                    true_positives += 1

                    matched_gt.add(
                        best_gt_idx
                    )

                else:

                    false_positives += 1

            false_negatives += (
                len(gt_boxes)
                - len(matched_gt)
            )

    precision = calculate_precision(
        true_positives,
        false_positives
    )

    recall = calculate_recall(
        true_positives,
        false_negatives
    )

    f1_score = calculate_f1_score(
        precision,
        recall
    )

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

    print("\nEvaluation Results")
    print("-" * 40)

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1 Score: {f1_score:.4f}"
    )

    print(
        f"TP: {true_positives}"
    )

    print(
        f"FP: {false_positives}"
    )

    print(
        f"FN: {false_negatives}"
    )

    return metrics


if __name__ == "__main__":

    evaluate()