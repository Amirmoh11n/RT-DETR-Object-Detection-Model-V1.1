import os

import torch

from PIL import Image

from inference.predict_image import (
    Predictor
)

from configs.config import (
    CHECKPOINT_PATH
)


TEST_IMAGE = (
    "assets/test_image.jpg"
)



def test_image_exists():

    assert os.path.exists(
        TEST_IMAGE
    )



def test_predictor_loading():

    predictor = Predictor()

    assert predictor.model is not None

    assert predictor.processor is not None



def test_prediction_output():

    predictor = Predictor()


    results = predictor.predict(
        TEST_IMAGE,
        threshold=0.5
    )


    assert results is not None


    assert "boxes" in results

    assert "scores" in results

    assert "labels" in results



def test_prediction_types():

    predictor = Predictor()


    results = predictor.predict(
        TEST_IMAGE
    )


    assert isinstance(
        results["boxes"],
        torch.Tensor
    )


    assert isinstance(
        results["scores"],
        torch.Tensor
    )


    assert isinstance(
        results["labels"],
        torch.Tensor
    )



def test_bbox_shape():

    predictor = Predictor()


    results = predictor.predict(
        TEST_IMAGE
    )


    boxes = results["boxes"]


    if len(boxes) > 0:

        assert boxes.ndim == 2

        assert boxes.shape[1] == 4



def test_scores_range():

    predictor = Predictor()


    results = predictor.predict(
        TEST_IMAGE
    )


    scores = results["scores"]


    if len(scores) > 0:

        assert torch.all(
            scores >= 0
        )

        assert torch.all(
            scores <= 1
        )