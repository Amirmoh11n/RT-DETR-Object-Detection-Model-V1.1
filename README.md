# RTDETR-Model-v1.1

A complete Object Detection pipeline built with RT-DETR and Pascal VOC 2012 using PyTorch and Hugging Face Transformers.

---

## Features

* RT-DETR (Real-Time Detection Transformer)
* Pascal VOC 2012 Dataset Support
* Automatic Dataset Download & Caching
* Train / Validation / Test Split
* Mixed Precision Training (AMP)
* Gradient Clipping
* Learning Rate Scheduler
* Early Stopping
* Checkpoint Saving & Loading
* Resume Training Support
* mAP Evaluation
* Precision / Recall / F1 Metrics
* Image Inference
* Video Inference
* Prediction Visualization
* Unit Tests

---

## Dataset

This project uses Pascal VOC 2012.

Dataset structure:

```text
datasets/
└── VOC/
    └── VOCdevkit/
        └── VOC2012/
```

The dataset is downloaded automatically on the first execution.

After the initial download, cached files are reused and no additional downloads are required.

---

## Installation

```bash
git clone https://github.com/Amirmoh11n/RT-DETR-Object-Detection-Model-V1.1

cd RT-DETR-Object-Detection-Model-V1.1

pip install -r requirements.txt
```

---

## Training

Start training:

```bash
python main.py
```

Training includes:

* Training Loss
* Validation Loss
* mAP
* mAP50
* mAP75
* mAR300
* Learning Rate Scheduling
* Early Stopping
* Best Model Saving

---

## Checkpoints

Best checkpoint location:

```text
checkpoints/
└── best_model.pth
```

Stored information:

* Model Weights
* Optimizer State
* Scheduler State
* AMP Scaler State
* Epoch Number
* Validation Loss
* Validation Metrics

Checkpoints are compatible with:

* Linux
* Windows
* Google Colab

---

## Evaluation

Run evaluation:

```bash
python -m evaluation.evaluate
```

Reported metrics:

* mAP
* mAP50
* mAP75
* mAR300
* Precision
* Recall
* F1 Score
* IoU

---

## Image Inference

```bash
python inference/predict_image.py
```

Output:

```text
outputs/
└── prediction.jpg
```

---

## Video Inference

```bash
python inference/predict_video.py
```

Output:

```text
outputs/
└── videos/
    └── result.mp4
```

---
---

---

## Results

### 📸 Image Prediction

Example output from the fine-tuned RT-DETR model:

### 🎥 Video Prediction Demo

<video src="https://github.com/user-attachments/assets/2c5adc67-2a1f-444c-9a99-2ffaa4d60b96" width="800">
</video>

<p align="center">
  <img src="[assets/demo.gif](https://github.com/user-attachments/assets/2c5adc67-2a1f-444c-9a99-2ffaa4d60b96)" width="80%">
</p>

---

## Project Structure

```text
RTDETR-Model-v1.1/
│
├── configs/
├── data/
├── evaluation/
├── inference/
├── models/
├── training/
├── visualization/
├── tests/
│
├── datasets/
├── checkpoints/
├── outputs/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Testing

Run all tests:

```bash
pytest
```

Run specific test modules:

```bash
pytest tests/test_model.py

pytest tests/test_dataset.py

pytest tests/test_inference.py
```

---

## Environment

Tested with:

* Python 3.12
* PyTorch 2.x
* CUDA 12.x
* Hugging Face Transformers
* Google Colab
* Linux
* Windows

---

## Current Version

### v1.1

Implemented:

* Dataset Pipeline
* Training Pipeline
* Validation Pipeline
* Evaluation Pipeline
* Image Inference
* Video Inference
* Visualization Tools
* Checkpoint System
* Automatic Dataset Caching

---

## Future Roadmap

Planned improvements:

* TensorBoard Integration
* Weights & Biases Integration
* ONNX Export
* TensorRT Export
* Docker Support
* Multi-GPU Training
* Hyperparameter Search
* COCO Dataset Support

---

## License

MIT License
