import cv2
import torch

from models.rtdetr_model import load_model


class VideoVisualizer:

    def __init__(self):

        (
            self.processor,
            self.model,
            self.device
        ) = load_model()

        self.model.eval()


    def process_video(
        self,
        input_video,
        output_video,
        threshold=0.5
    ):

        cap = cv2.VideoCapture(
            input_video
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

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

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
                k:v.to(self.device)
                for k,v in inputs.items()
            }

            with torch.no_grad():

                outputs = self.model(
                    **inputs
                )

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
                    f"{class_name} "
                    f"{score:.2f}"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    text,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )

            writer.write(
                frame
            )

        cap.release()

        writer.release()

        print(
            f"Saved: {output_video}"
        )


if __name__ == "__main__":

    visualizer = VideoVisualizer()

    visualizer.process_video(
        input_video="assets/video.mp4",
        output_video="results/output.mp4",
        threshold=0.5
    )