# RT-DETR Pascal VOC Object Detection

A modular object detection project built with **PyTorch**, **Transformers**, and **RT-DETR (Real-Time Detection Transformer)**.

The project focuses on fine-tuning an open-source Vision Transformer-based object detector on the **Pascal VOC 2012** dataset and building a complete training and inference pipeline using modern deep learning practices.

---

## Features

- RT-DETR object detection model
- Fine-tuning on Pascal VOC 2012
- Multi-class object detection (20 VOC classes)
- Modular project structure
- Train / Validation / Test split
- Global dataset shuffling
- Mixed Precision Training (AMP)
- Early Stopping
- Model Checkpointing
- mAP Evaluation
- Image Inference Pipeline

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- OpenCV
- TorchMetrics
- Pascal VOC 2012

---

## Project Structure

```text
RTDETR-VOC-Object-Detection/
│
├── configs/
├── data/
├── models/
├── training/
├── inference/
├── checkpoints/
├── outputs/
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

---

## Model

**RT-DETR (Real-Time Detection Transformer)**

RT-DETR combines Transformer-based object detection with real-time inference capabilities, providing strong detection performance while maintaining efficient execution.

Base model:

```text
PekingU/rtdetr_r50vd
```

---

## Dataset

**Pascal VOC 2012**

The model is fine-tuned on the Pascal VOC dataset containing 20 object categories:

- Person
- Car
- Bus
- Bicycle
- Dog
- Cat
- Train
- Horse
- Boat
- And other VOC classes

---

## Training Pipeline

```text
Dataset
   ↓
Global Shuffle
   ↓
Train / Validation / Test Split
   ↓
RT-DETR Fine-Tuning
   ↓
Validation
   ↓
Early Stopping
   ↓
Best Checkpoint
   ↓
Test Evaluation
```

---

## Evaluation

The project evaluates performance using:

- Validation Loss
- Mean Average Precision (mAP)
- mAP@50
- mAP@50:95

---

## Inference

The trained model can perform object detection on custom images and generate visual outputs with bounding boxes and confidence scores.

Example workflow:

```text
Input Image
      ↓
RT-DETR
      ↓
Object Detection
      ↓
Output Image
```

---

## Future Improvements

- Video Object Detection
- Webcam Inference
- Traffic Camera Analysis
- ONNX Export
- TensorRT Optimization
- Streamlit Demo

---

## Results

Training metrics, checkpoints, prediction samples, and evaluation results will be stored in the `outputs/` directory.

---

## Author

AmirMohammad Nashalji

Object Detection and Computer Vision Project built with PyTorch and RT-DETR.
