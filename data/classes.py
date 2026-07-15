VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

CLASS_TO_ID = {
    name: idx
    for idx, name in enumerate(VOC_CLASSES)
}

ID_TO_CLASS = {
    idx: name
    for idx, name in enumerate(VOC_CLASSES)
}