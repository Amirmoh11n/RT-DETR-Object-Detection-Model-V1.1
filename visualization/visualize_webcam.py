import cv2
import torch

from models.rtdetr_model import load_model


class WebcamVisualizer:

    def __init__(self):

        (
            self.processor,
            self.model,
            self.device
        ) = load_model()

        self.model.eval()


    def run(
        self,
        threshold=0.5
    ):

        cap = cv2.VideoCapture(
            0
        )

        while True:

            success, frame = cap.read()

            if not success:
                break

            height, width = (
                frame.shape[:2]
            )

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

            for box, score, label in zip(
                results["boxes"],
                results["scores"],
                results["labels"]
            ):

                x1,y1,x2,y2 = (
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
                    (x1,y1),
                    (x2,y2),
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

            cv2.imshow(
                "RT-DETR Detection",
                frame
            )

            key = cv2.waitKey(
                1
            )

            if key == 27:
                break

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    WebcamVisualizer().run()