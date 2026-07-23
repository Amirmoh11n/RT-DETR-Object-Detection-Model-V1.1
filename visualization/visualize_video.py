import os

import cv2
import torch

from configs.config import (
    CHECKPOINT_PATH
)

from models.rtdetr_model import (
    load_model
)

from training.checkpoint import (
    load_model_weights
)


class VideoVisualizer:

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

        fps = int(
            cap.get(
                cv2.CAP_PROP_FPS
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

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            inputs = self.processor(
                images=rgb,
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
                [
                    (
                        height,
                        width
                    )
                ],
                device=self.device
            )

            results = self.processor.post_process_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=target_sizes
            )[0]

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
                    f"{class_name} "
                    f"{score:.2f}"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
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

    visualizer = VideoVisualizer()

    visualizer.process_video(
        input_video="assets/video.mp4",
        output_video="results/output.mp4",
        threshold=0.5
    )