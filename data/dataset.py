import os
import torch
import numpy as np

from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset
)

from torchvision.datasets import (
    VOCDetection
)

from data.collate import collate_fn

from data.classes import (
    CLASS_TO_ID,
    ID_TO_CLASS
)

from configs.config import (
    BATCH_SIZE,
    NUM_WORKERS
)


VOC_ROOT = "./datasets/VOCtrainval_11-May-2012"


class VOCRTDETRDataset(Dataset):

    def __init__(
        self,
        dataset,
        processor
    ):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(
        self,
        idx
    ):

        image, target = self.dataset[idx]

        annotation = {
            "image_id": idx,
            "annotations": []
        }

        objects = target[
            "annotation"
        ].get(
            "object",
            []
        )

        if not isinstance(
            objects,
            list
        ):
            objects = [objects]

        for obj in objects:

            class_name = obj["name"]

            if class_name not in CLASS_TO_ID:
                continue

            category_id = CLASS_TO_ID[
                class_name
            ]

            bbox = obj["bndbox"]

            xmin = float(
                bbox["xmin"]
            )

            ymin = float(
                bbox["ymin"]
            )

            xmax = float(
                bbox["xmax"]
            )

            ymax = float(
                bbox["ymax"]
            )

            width = xmax - xmin
            height = ymax - ymin

            if width <= 0 or height <= 0:
                continue

            annotation[
                "annotations"
            ].append(
                {
                    "bbox": [
                        xmin,
                        ymin,
                        width,
                        height
                    ],
                    "category_id": category_id,
                    "area": (
                        width * height
                    ),
                    "iscrowd": 0
                }
            )

        encoding = self.processor(
            images=image,
            annotations=annotation,
            return_tensors="pt"
        )

        pixel_values = encoding[
            "pixel_values"
        ].squeeze(0)

        labels = encoding[
            "labels"
        ][0]

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


def create_splits(
    dataset,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42
):

    np.random.seed(seed)

    indices = np.random.permutation(
        len(dataset)
    )

    train_size = int(
        train_ratio * len(dataset)
    )

    val_size = int(
        val_ratio * len(dataset)
    )

    train_indices = indices[
        :train_size
    ]

    val_indices = indices[
        train_size:
        train_size + val_size
    ]

    test_indices = indices[
        train_size + val_size:
    ]

    train_subset = Subset(
        dataset,
        train_indices
    )

    val_subset = Subset(
        dataset,
        val_indices
    )

    test_subset = Subset(
        dataset,
        test_indices
    )

    return (
        train_subset,
        val_subset,
        test_subset
    )


def load_voc_dataset():

    os.makedirs(
        VOC_ROOT,
        exist_ok=True
    )

    voc_dir = os.path.join(
        VOC_ROOT,
        "VOCdevkit"
    )

    download_dataset = (
        not os.path.exists(voc_dir)
    )

    if download_dataset:

        print(
            "VOC dataset not found. Downloading..."
        )

    else:

        print(
            "Using cached VOC dataset."
        )

    dataset = VOCDetection(
        root=VOC_ROOT,
        year="2012",
        image_set="train",
        download=download_dataset
    )

    return dataset


def get_datasets(
    processor
):

    dataset = load_voc_dataset()

    (
        train_subset,
        val_subset,
        test_subset
    ) = create_splits(
        dataset
    )

    train_dataset = VOCRTDETRDataset(
        train_subset,
        processor
    )

    val_dataset = VOCRTDETRDataset(
        val_subset,
        processor
    )

    test_dataset = VOCRTDETRDataset(
        test_subset,
        processor
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


def get_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            NUM_WORKERS > 0
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            NUM_WORKERS > 0
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            NUM_WORKERS > 0
        )
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )