import os

import cv2
import torch
import numpy as np

from PIL import Image

from configs.config import (
    CHECKPOINT_PATH
)

from models.rtdetr_model import (
    load_model
)

from training.checkpoint import (
    load_model_weights
)


class PredictionVisualizer:

    def __init__(self):

        (
            self.processor,
            self.model,
            self.device
        ) = load_model()

        load_model_weights(
            self.model,
            CHECKPOINT_PATH,
            self.device
        )

        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        threshold: float = 0.5
    ):

        if not os.path.exists(
            image_path
        ):

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

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

        width, height = image.size

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=[
                (
                    height,
                    width
                )
            ]
        )[0]

        return image, results

    def draw_predictions(
        self,
        image_path: str,
        output_path: str = "results/prediction.jpg",
        threshold: float = 0.5
    ):

        image, results = self.predict(
            image_path=image_path,
            threshold=threshold
        )

        image = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )

        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"]
        ):

            x1, y1, x2, y2 = map(
                int,
                box.cpu().tolist()
            )

            score = float(
                score.item()
            )

            label = int(
                label.item()
            )

            class_name = (
                self.model.config.id2label.get(
                    label,
                    f"class_{label}"
                )
            )

            text = (
                f"{class_name}: "
                f"{score:.2f}"
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                text,
                (
                    x1,
                    max(
                        y1 - 10,
                        15
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        output_dir = os.path.dirname(
            output_path
        )

        if output_dir:

            os.makedirs(
                output_dir,
                exist_ok=True
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

    visualizer = PredictionVisualizer()

    visualizer.draw_predictions(
        image_path="assets/test_image.jpg",
        output_path="results/prediction.jpg",
        threshold=0.5
    )