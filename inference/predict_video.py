import os

import cv2
import torch

from PIL import Image

from configs.config import (
    CHECKPOINT_PATH
)

from data.classes import (
    ID_TO_CLASS
)

from models.rtdetr_model import (
    load_model
)

from training.checkpoint import (
    load_model_weights
)


class VideoPredictor:

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
    def detect_frame(
        self,
        frame,
        threshold=0.5
    ):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

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

    def draw_boxes(
        self,
        frame,
        results
    ):

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
                frame,
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
                frame,
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

        return frame

    def process_video(
        self,
        input_video,
        output_video,
        threshold=0.5
    ):

        if not os.path.exists(
            input_video
        ):

            raise FileNotFoundError(
                f"Video not found: {input_video}"
            )

        output_dir = os.path.dirname(
            output_video
        )

        if output_dir:

            os.makedirs(
                output_dir,
                exist_ok=True
            )

        cap = cv2.VideoCapture(
            input_video
        )

        if not cap.isOpened():

            raise ValueError(
                f"Failed to open video: {input_video}"
            )

        fps = int(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        if fps <= 0:
            fps = 30

        writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (
                width,
                height
            )
        )

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = self.detect_frame(
                frame,
                threshold
            )

            frame = self.draw_boxes(
                frame,
                results
            )

            writer.write(
                frame
            )

            frame_count += 1

        cap.release()

        writer.release()

        print(
            f"Processed {frame_count} frames"
        )

        print(
            f"Saved: {output_video}"
        )

        return output_video


if __name__ == "__main__":

    predictor = VideoPredictor()

    predictor.process_video(
        input_video="traffic.mp4",
        output_video="outputs/videos/result.mp4",
        threshold=0.5
    )