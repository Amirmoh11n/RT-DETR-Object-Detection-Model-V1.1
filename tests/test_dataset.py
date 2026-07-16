import torch

from transformers import (
    RTDetrImageProcessor
)

from torchvision.datasets import VOCDetection

from data.dataset import (
    VOCRTDETRDataset
)

from configs.config import (
    DATASET_ROOT,
    MODEL_NAME
)


def create_test_dataset():

    processor = RTDetrImageProcessor.from_pretrained(
        MODEL_NAME
    )

    dataset = VOCDetection(
        root=DATASET_ROOT,
        year="2012",
        image_set="train",
        download=False
    )


    wrapped_dataset = VOCRTDETRDataset(
        dataset,
        processor
    )

    return wrapped_dataset



def test_dataset_length():

    dataset = create_test_dataset()

    assert len(dataset) > 0



def test_dataset_sample():

    dataset = create_test_dataset()

    sample = dataset[0]


    assert isinstance(
        sample,
        dict
    )


    assert (
        "pixel_values"
        in sample
    )


    assert (
        "labels"
        in sample
    )



def test_image_tensor_shape():

    dataset = create_test_dataset()

    sample = dataset[0]


    image = sample[
        "pixel_values"
    ]


    assert isinstance(
        image,
        torch.Tensor
    )


    assert image.ndim == 3


    assert image.shape[0] == 3



def test_labels_structure():

    dataset = create_test_dataset()

    sample = dataset[0]


    labels = sample[
        "labels"
    ]


    assert isinstance(
        labels,
        dict
    )


    assert (
        "class_labels"
        in labels
    )


    assert (
        "boxes"
        in labels
    )



def test_bbox_validity():

    dataset = create_test_dataset()

    sample = dataset[0]

    labels = sample[
        "labels"
    ]


    boxes = labels[
        "boxes"
    ]


    assert boxes.ndim == 2


    assert boxes.shape[1] == 4


    assert torch.all(
        boxes >= 0
    )


    assert torch.all(
        boxes <= 1
    )



def test_class_ids_range():

    dataset = create_test_dataset()

    sample = dataset[0]


    labels = sample[
        "labels"
    ]


    class_ids = labels[
        "class_labels"
    ]


    num_classes = 20


    assert torch.all(
        class_ids >= 0
    )


    assert torch.all(
        class_ids < num_classes
    )