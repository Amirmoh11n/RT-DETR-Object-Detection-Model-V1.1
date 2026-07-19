# RTDETR-Model-v1.1

A modular object detection framework based on **RT-DETR (Real-Time Detection Transformer)**, developed using **PyTorch** and **Hugging Face Transformers**.

This project aims to build a clean, maintainable, and extensible computer vision pipeline covering dataset preparation, model training, evaluation, inference, visualization, and testing. The repository is structured following software engineering best practices rather than being limited to a single training notebook.

---

# Project Overview

Object detection remains one of the most important tasks in computer vision. Traditional CNN-based detectors such as Faster R-CNN, SSD, and YOLO have dominated the field for years. More recently, transformer-based architectures have demonstrated the ability to model long-range dependencies and global context more effectively.

RT-DETR (Real-Time Detection Transformer) combines the advantages of transformer-based detection with real-time inference performance, making it a strong candidate for modern object detection systems.

This project explores RT-DETR as a practical engineering solution while focusing on:

- Modular software architecture
- Reproducible experimentation
- Dataset abstraction
- Training and evaluation workflows
- Deployment readiness
- Extensibility for future datasets and tasks

---

# Project Goals

The primary objectives of this repository are:

- Build a complete RT-DETR training pipeline
- Fine-tune RT-DETR on Pascal VOC
- Create reusable dataset and model abstractions
- Support image, video, and webcam inference
- Implement evaluation metrics for object detection
- Maintain a testable and production-oriented codebase
- Establish a foundation for future custom datasets

---

# RT-DETR Architecture

RT-DETR (Real-Time Detection Transformer) is a transformer-based object detector designed to provide both high accuracy and low inference latency.

Key characteristics include:

- Transformer-based object detection
- End-to-end detection pipeline
- NMS-free inference
- Real-time performance
- Global feature modeling
- Efficient deployment capabilities

The implementation in this repository uses the Hugging Face Transformers integration of RT-DETR.

---

# Dataset

## Pascal VOC 2012

The current version of the project uses the Pascal VOC 2012 dataset.

The dataset contains 20 object categories:

```text
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
```

The dataset pipeline includes:

- XML annotation parsing
- Annotation conversion
- RT-DETR-compatible formatting
- Dataset shuffling
- Train / Validation / Test splitting
- DataLoader generation

---

# Data Processing Pipeline

The data processing workflow follows the structure below:

```text
Pascal VOC Dataset
        │
        ▼
Annotation Parsing
        │
        ▼
RT-DETR Annotation Formatting
        │
        ▼
Dataset Shuffling
        │
        ▼
Train / Validation / Test Split
        │
        ▼
DataLoader Creation
```

This design ensures reproducibility and simplifies future dataset integrations.

---

# Project Structure

```text
RTDETR-Model-v1.1
│
├── configs/
│   └── config.py
│
├── data/
│   ├── classes.py
│   ├── collate.py
│   └── dataloader.py
│
├── models/
│   ├── rtdetr_model.py
│   └── checkpoint.py
│
├── training/
│   ├── train.py
│   ├── validate.py
│   └── early_stopping.py
│
├── inference/
│   ├── predictor.py
│   ├── predict_image.py
│   └── predict_video.py
│
├── visualization/
│   ├── visualize_predictions.py
│   ├── visualize_video.py
│   └── visualize_webcam.py
│
├── evaluation/
│   ├── metrics.py
│   └── evaluate.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_training.py
│   └── test_inference.py
│
├── assets/
├── checkpoints/
├── results/
│
├── requirements.txt
├── pytest.ini
└── README.md
```

---

# Technology Stack

| Category | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Detection Model | RT-DETR |
| Transformers | Hugging Face Transformers |
| Dataset | Pascal VOC |
| Computer Vision | OpenCV |
| Visualization | Matplotlib |
| Testing | PyTest |
| Serialization | SafeTensors |

---

# Training Pipeline

The training process follows a standard deep learning workflow:

```text
Dataset
    │
    ▼
Preprocessing
    │
    ▼
RT-DETR Processor
    │
    ▼
Forward Pass
    │
    ▼
Loss Computation
    │
    ▼
Backward Pass
    │
    ▼
Optimizer Update
    │
    ▼
Validation
    │
    ▼
Checkpoint Saving
```

The training system is designed to support future additions such as:

- Early stopping
- Learning rate scheduling
- Mixed precision training
- Distributed training

---

# Model Checkpoints

The repository supports saving and loading model checkpoints.

Checkpoint contents may include:

- Model weights
- Optimizer state
- Current epoch
- Training metadata

Saved checkpoints are stored inside:

```text
checkpoints/
```

Example:

```text
epoch_1.pth
best_model.pth
```

---

# Evaluation Pipeline

The evaluation module is responsible for measuring object detection performance.

Planned evaluation metrics include:

- Precision
- Recall
- F1 Score
- IoU (Intersection over Union)
- mAP@50
- mAP@50:95

The evaluation system is designed to work independently from the training loop to allow reproducible benchmarking.

---

# Inference

The project supports multiple inference modes.

## Image Inference

Single-image object detection with bounding box visualization.

Output example:

```text
results/prediction.jpg
```

---

## Video Inference

Object detection on recorded video files.

Output example:

```text
results/output.mp4
```

---

## Webcam Inference

Real-time object detection using a connected webcam.

This mode is intended for rapid testing and demonstrations.

---

# Testing

The repository includes automated tests for critical components.

Current testing areas include:

- Dataset loading
- Dataset processing
- Model initialization
- Forward pass validation
- Backward pass validation
- Optimizer updates
- Inference pipeline validation

Tests are implemented using PyTest.

Run all tests:

```bash
pytest -v
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/RTDETR-Model-v1.1.git

cd RTDETR-Model-v1.1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage

## Train

```bash
python train.py
```

---

## Evaluate

```bash
python evaluation/evaluate.py
```

---

## Image Inference

```bash
python inference/predict_image.py
```

---

## Video Inference

```bash
python inference/predict_video.py
```

---

## Webcam Inference

```bash
python visualization/visualize_webcam.py
```

---

# Future Work

This repository is intended to evolve beyond a basic fine-tuning project.

Future development directions include:

- Complete evaluation benchmark pipeline
- mAP computation
- TensorBoard integration
- ONNX export
- TorchScript export
- Docker support
- Streamlit interface
- Hugging Face Spaces deployment
- Custom dataset integration
- Multi-dataset experimentation
- Hyperparameter optimization

---

# Motivation

The purpose of this project is not only to train an object detection model, but also to build a maintainable machine learning codebase that follows software engineering principles.

By separating datasets, models, training logic, evaluation, inference, and testing into dedicated modules, the project becomes easier to extend, debug, test, and deploy.

---

# License

This project is released under the MIT License.

---

# Author

**Amir**

Computer Vision • Deep Learning • Object Detection • Vision Transformers
