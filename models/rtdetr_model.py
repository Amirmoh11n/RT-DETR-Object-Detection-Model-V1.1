import torch

from transformers import (
    RTDetrImageProcessor,
    RTDetrForObjectDetection
)

from data.classes import (
    ID_TO_CLASS,
    CLASS_TO_ID
)

from configs.config import MODEL_NAME

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    return (
        processor,
        model,
        device
    )
