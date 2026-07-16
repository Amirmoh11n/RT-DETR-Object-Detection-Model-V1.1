import torch

from torch.utils.data import DataLoader

from models.rtdetr_model import load_model


class DummyDetectionDataset(torch.utils.data.Dataset):

    def __len__(self):
        return 2


    def __getitem__(self, idx):

        return {
            "pixel_values":
                torch.rand(
                    3,
                    640,
                    640
                ),

            "labels":
                {
                    "class_labels":
                        torch.tensor([1]),

                    "boxes":
                        torch.tensor(
                            [
                                [
                                    0.2,
                                    0.2,
                                    0.5,
                                    0.5
                                ]
                            ]
                        )
                }
        }



def dummy_collate(batch):

    return {
        "pixel_values":
            torch.stack(
                [
                    x["pixel_values"]
                    for x in batch
                ]
            ),

        "labels":
            [
                x["labels"]
                for x in batch
            ]
    }



def test_training_forward_backward():

    _, model, device = load_model()

    model.train()


    loader = DataLoader(
        DummyDetectionDataset(),
        batch_size=2,
        collate_fn=dummy_collate
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5
    )


    batch = next(
        iter(loader)
    )


    outputs = model(
        pixel_values=batch["pixel_values"].to(device),
        labels=[
            {
                k:v.to(device)
                for k,v in label.items()
            }
            for label in batch["labels"]
        ]
    )


    loss = outputs.loss


    assert loss is not None

    assert loss.item() > 0


    loss.backward()

    optimizer.step()



def test_optimizer_updates_weights():

    _, model, device = load_model()

    model.train()


    old_weight = (
        next(model.parameters())
        .clone()
        .detach()
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5
    )


    image = torch.rand(
        1,
        3,
        640,
        640
    ).to(device)


    label = [
        {
            "class_labels":
                torch.tensor([1]).to(device),

            "boxes":
                torch.tensor(
                    [
                        [
                            0.2,
                            0.2,
                            0.5,
                            0.5
                        ]
                    ]
                ).to(device)
        }
    ]


    output = model(
        pixel_values=image,
        labels=label
    )


    loss = output.loss


    loss.backward()

    optimizer.step()


    new_weight = (
        next(model.parameters())
        .clone()
        .detach()
    )


    assert not torch.equal(
        old_weight,
        new_weight
    )