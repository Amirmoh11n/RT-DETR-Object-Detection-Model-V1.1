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

        self.processor, \
        self.model, \
        self.device = load_model()

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

        if not os.path.exists(image_path):

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(
            **inputs
        )

        target_sizes = torch.tensor(
            [image.size[::-1]],
            device=self.device
        )

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
            image_path=image_path,
            threshold=threshold
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise ValueError(
                f"Failed to load image: {image_path}"
            )

        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):

            score = float(
                score.item()
            )

            label = int(
                label.item()
            )

            xmin, ymin, xmax, ymax = map(
                int,
                box.tolist()
            )

            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                (0, 255, 0),
                2
            )

            class_name = ID_TO_CLASS.get(
                label,
                f"class_{label}"
            )

            text = (
                f"{class_name} "
                f"{score:.2f}"
            )

            cv2.putText(
                image,
                text,
                (
                    xmin,
                    max(
                        ymin - 5,
                        15
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

        image_name = os.path.basename(
            image_path
        )

        file_name, _ = os.path.splitext(
            image_name
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{file_name}_prediction.jpg"
        )

        cv2.imwrite(
            output_path,
            image
        )

        print(
            f"Saved: {output_path}"
        )

        return output_path


if __name__ == "__main__":

    predictor = Predictor()

    predictor.visualize(
        "sample.jpg",
        threshold=0.5
    )