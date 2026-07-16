from models.rtdetr_model import load_model
from data.dataset import get_datasets

processor, model, device = load_model()

train_dataset, val_dataset, test_dataset = get_datasets(
    processor
)

sample = test_dataset[0]

print(type(sample))

print(sample.keys())

print(sample["labels"].keys())

print(sample["labels"]["boxes"])

print(sample["labels"]["class_labels"])