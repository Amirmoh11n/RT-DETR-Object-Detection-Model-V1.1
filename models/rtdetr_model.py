import torch

from transformers import (
    RTDetrImageProcessor,
    RTDetrForObjectDetection
)

from data.classes import (
    ID_TO_CLASS,
    CLASS_TO_ID
)

from configs.config import (
    MODEL_NAME,
    CHECKPOINT_PATH
)


def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_model_weights(
    model,
    checkpoint_path,
    device
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


def load_model():

    processor = RTDetrImageProcessor.from_pretrained(
        MODEL_NAME
    )


    model = RTDetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_TO_ID),
        id2label=ID_TO_CLASS,
        label2id=CLASS_TO_ID,
        ignore_mismatched_sizes=True
    )


    device = get_device()


    model.to(device)


    # Load fine-tuned weights
    load_model_weights(
        model,
        CHECKPOINT_PATH,
        device
    )


    model.eval()


    return (
        processor,
        model,
        device
    )