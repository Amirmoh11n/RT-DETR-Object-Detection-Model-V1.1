import torch

from models.rtdetr_model import load_model
from configs.config import MODEL_NAME

from data.classes import (
    ID_TO_CLASS,
    CLASS_TO_ID
)


def test_model_loading(model_bundle):

    processor, model, device = model_bundle

    assert model is not None
    assert processor is not None

    assert isinstance(
        device,
        torch.device
    )


def test_number_of_classes():

    _, model, _ = load_model()

    assert model.config.num_labels == len(
        CLASS_TO_ID
    )

    assert model.config.num_labels == len(
        ID_TO_CLASS
    )


def test_model_device():

    _, model, device = load_model()

    model_device = next(
        model.parameters()
    ).device

    assert model_device.type == device.type


def test_training_forward_backward(model_bundle):

    _, model, device = model_bundle


    model.eval()

    dummy_image = torch.rand(
        3,
        640,
        640
    )

    inputs = {
        "pixel_values":
            dummy_image.unsqueeze(0).to(device)
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )


    assert outputs is not None

    assert hasattr(
        outputs,
        "logits"
    )

    assert hasattr(
        outputs,
        "pred_boxes"
    )


def test_model_output_shape():

    processor, model, device = load_model()

    model.eval()

    dummy_input = torch.rand(
        1,
        3,
        640,
        640
    ).to(device)


    with torch.no_grad():

        outputs = model(
            pixel_values=dummy_input
        )


    batch_size = (
        outputs.logits.shape[0]
    )


    num_classes = (
        outputs.logits.shape[-1]
    )


    assert batch_size == 1

    assert num_classes == len(
        CLASS_TO_ID
    )