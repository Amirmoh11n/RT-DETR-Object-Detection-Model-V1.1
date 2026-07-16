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

        self.processor,\
        self.model,\
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

        image = Image.fromarray(rgb)

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


    def draw_boxes(
        self,
        frame,
        results
    ):

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
                frame,
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                (0,255,0),
                2
            )

            text = (
                f"{ID_TO_CLASS[label]}"
                f" {score:.2f}"
            )

            cv2.putText(
                frame,
                text,
                (int(xmin), int(ymin)-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

        return frame


    def process_video(
        self,
        input_video,
        output_video,
        threshold=0.5
    ):

        cap = cv2.VideoCapture(
            input_video
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

        writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

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

            writer.write(frame)

        cap.release()

        writer.release()

        print(
            f"Saved: {output_video}"
        )

if __name__ == "__main__":

    predictor = VideoPredictor()

    predictor.process_video(
        "traffic.mp4",
        "outputs/videos/result.mp4",
        threshold=0.5
    )