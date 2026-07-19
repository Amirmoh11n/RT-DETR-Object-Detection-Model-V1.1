from typing import Tuple

import torch


def calculate_precision(
    true_positives: int,
    false_positives: int
) -> float:

    denominator = true_positives + false_positives

    if denominator == 0:
        return 0.0

    return true_positives / denominator


def calculate_recall(
    true_positives: int,
    false_negatives: int
) -> float:

    denominator = true_positives + false_negatives

    if denominator == 0:
        return 0.0

    return true_positives / denominator


def calculate_f1_score(
    precision: float,
    recall: float
) -> float:

    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return 2 * (precision * recall) / denominator


def calculate_iou(
    box_a: torch.Tensor,
    box_b: torch.Tensor
) -> float:

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])

    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(
        0,
        x2 - x1
    ) * max(
        0,
        y2 - y1
    )

    area_a = max(
        0,
        box_a[2] - box_a[0]
    ) * max(
        0,
        box_a[3] - box_a[1]
    )

    area_b = max(
        0,
        box_b[2] - box_b[0]
    ) * max(
        0,
        box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)