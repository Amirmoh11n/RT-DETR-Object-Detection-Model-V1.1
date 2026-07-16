import cv2
import torch

from PIL import Image

from models.rtdetr_model import load_model

import numpy as np

class PredictionVisualizer:

    def __init__(self):

        (
            self.processor,
            self.model,
            self.device
        ) = load_model()


    def predict(
        self,
        image_path: str,
        threshold: float = 0.5
    ):

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

        with torch.no_grad():

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

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        for box, score, label in zip(
            boxes,
            scores,
            labels
        ):

            x1, y1, x2, y2 = (
                box.cpu()
                .numpy()
                .astype(int)
            )

            class_name = (
                self.model.config.id2label[
                    int(label)
                ]
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
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        cv2.imwrite(
            output_path,
            image
        )

        print(
            f"Saved: {output_path}"
        )

if __name__ == "__main__":

    visualizer = PredictionVisualizer()

    visualizer.draw_predictions(
        image_path="assets/test_image.jpg",
        output_path="results/prediction.jpg",
        threshold=0.5
    )