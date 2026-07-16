import pytest
import torch

from models.rtdetr_model import load_model


@pytest.fixture(scope="session")
def model_bundle():

    processor, model, device = load_model()

    model.eval()

    return (
        processor,
        model,
        device
    )