# Retinal Fundus Image Classification

A PyTorch implementation of a binary classifier for retinal fundus images using transfer learning with EfficientNet-B0.

## Overview

This project implements a deep learning model that classifies retinal fundus images as either "Normal" or "Abnormal". The model uses transfer learning with a pre-trained EfficientNet-B0 architecture to achieve high performance even with a small dataset.

## Requirements

```
torch
torchvision
timm
torchmetrics
PIL
```

Install dependencies with:
```
pip install torch torchvision timm torchmetrics
```

## Dataset Structure

Place images in the following structure:
```
train/
  ├── Normal/
  │   └── [normal images]
  └── Abnormal/
      └── [abnormal images]
```

## Usage

### Training

```
python train.py
```

Training will:
- Split data into 80% training and 20% validation sets
- Train for 10 epochs
- Use MPS acceleration on Mac if available
- Save the best model as `best_retina_model.pth`

### Inference

```
python inference.py path/to/image.jpg
```

## Results

The model achieves:
- 88.89% validation accuracy
- 0.8000 F1 score

See `Report.docx` for detailed analysis and methodology. 