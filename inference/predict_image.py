import os

import cv2
import torch

from PIL import Image

from configs.config import (
    CHECKPOINT_PATH,
    OUTPUT_DIR
)

from models.rtdetr_model import (
    load_model
)

from training.checkpoint import (
    load_model_weights
)

from data.classes import (
    ID_TO_CLASS
)

class Predictor:

    def __init__(self):

        self.processor,\
        self.model,\
        self.device = load_model()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-5
        )

        load_model_weights(
            self.model,
            CHECKPOINT_PATH,
            self.device
        )

        self.model.eval()
        
    @torch.no_grad()
    def predict(
        self,
        image_path,
        threshold=0.5
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model(
            **inputs
        )

        target_sizes = torch.tensor(
            [image.size[::-1]]
        ).to(self.device)

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes
        )[0]

        return results

    def visualize(
        self,
        image_path,
        threshold=0.5
    ):

        results = self.predict(
            image_path,
            threshold
        )

        image = cv2.imread(
            image_path
        )

        for score,\
            label,\
            box in zip(

            results["scores"],
            results["labels"],
            results["boxes"]

        ):

            score = score.item()

            label = label.item()

            xmin,\
            ymin,\
            xmax,\
            ymax = box.tolist()

            cv2.rectangle(
                image,
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                (0,255,0),
                2
            )

            text = (
                f"{ID_TO_CLASS[label]} "
                f"{score:.2f}"
            )

            cv2.putText(
                image,
                text,
                (int(xmin), int(ymin)-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            "prediction.jpg"
        )

        cv2.imwrite(
            output_path,
            image
        )

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":

    predictor = Predictor()

    predictor.visualize(
        "sample.jpg",
        threshold=0.5
    )