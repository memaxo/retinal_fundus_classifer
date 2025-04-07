# Retinal Image Classification Project Report

## Project Overview
This project implements a deep learning model for binary classification of retinal fundus images into "Normal" and "Abnormal" categories. The model uses transfer learning with a pre-trained EfficientNet-B0 architecture to achieve high performance even with a small dataset.

## Dataset
- **Data Source**: Retinal fundus images organized in two classes (Normal/Abnormal)
- **Size**: 46 total images (30 Abnormal, 16 Normal)
- **Split**: 80% training (36 images), 20% validation (9 images)

## Model Architecture
- **Base Model**: EfficientNet-B0 pre-trained on ImageNet
- **Adaptation**: Final classification layer modified for binary classification
- **Framework**: PyTorch with timm (PyTorch Image Models) library

## Training Methodology
- **Transfer Learning**: Leveraged a pre-trained model to overcome small dataset limitations
- **Image Preprocessing**:
  - Resizing to 224×224 pixels
  - Normalization using ImageNet mean/std values
  - Data augmentation (random horizontal flips) to improve generalization
- **Training Parameters**:
  - Optimizer: Adam with 1e-4 learning rate
  - Loss Function: Cross-Entropy Loss
  - Batch Size: 8
  - Epochs: 10
  - Device: Apple Metal Performance Shaders (MPS) for GPU acceleration
- **Model Selection**: Best model saved based on validation F1 score

## Training Results
```
Epoch 1/10, Training Loss: 1.6134, Validation Accuracy: 0.7778, F1 Score: 0.6786
Saved new best model with F1 score: 0.6786
Epoch 2/10, Training Loss: 0.8220, Validation Accuracy: 0.5556, F1 Score: 0.5000
Epoch 3/10, Training Loss: 0.2159, Validation Accuracy: 0.6667, F1 Score: 0.5846
Epoch 4/10, Training Loss: 0.0200, Validation Accuracy: 0.7778, F1 Score: 0.6786
Epoch 5/10, Training Loss: 0.6804, Validation Accuracy: 0.8889, F1 Score: 0.8000
Saved new best model with F1 score: 0.8000
Epoch 6/10, Training Loss: 0.3608, Validation Accuracy: 0.8889, F1 Score: 0.8000
Epoch 7/10, Training Loss: 0.1246, Validation Accuracy: 0.8889, F1 Score: 0.8000
Epoch 8/10, Training Loss: 0.0925, Validation Accuracy: 0.8889, F1 Score: 0.8000
Epoch 9/10, Training Loss: 0.1740, Validation Accuracy: 0.8889, F1 Score: 0.8000
Epoch 10/10, Training Loss: 0.0001, Validation Accuracy: 0.7778, F1 Score: 0.4375
```

- **Best Model Performance** (Epoch 5):
  - Validation Accuracy: 88.89%
  - F1 Score: 0.8000
  - Training Loss: 0.6804

## Observations
1. **Convergence**: The model achieved high performance by epoch 5 and maintained it until epoch 9
2. **Potential Overfitting**: The drop in F1 score in epoch 10 (0.8000 → 0.4375) suggests overfitting despite low training loss
3. **Performance Plateau**: Validation accuracy plateaued at 88.89% for epochs 5-9
4. **Model Size-Performance Balance**: EfficientNet-B0 provided a good balance of model size and performance for this task

## Implementation Details
- **Core Components**:
  1. `train.py`: Main training script with model definition and training loop
  2. `inference.py`: Script for running inference on new images
  
- **Key Libraries**:
  - PyTorch: Core deep learning framework
  - torchvision: For dataset handling and transformations
  - timm: For accessing pre-trained EfficientNet models
  - torchmetrics: For accurate computation of Accuracy and F1 metrics

- **Hardware Acceleration**: Apple Metal Performance Shaders (MPS) for GPU acceleration on Mac

## Limitations and Future Work
1. **Dataset Size**: The small dataset (46 images) limits the model's generalization ability
2. **Class Imbalance**: Uneven distribution between classes (30 Abnormal vs 16 Normal)
3. **Validation Strategy**: A more robust k-fold cross-validation approach would provide better performance estimates

## Potential Improvements
1. **Data Collection**: Gather more retinal images for training
2. **Augmentation**: Implement more diverse augmentation techniques (rotations, color shifts, etc.)
3. **Model Ensembling**: Create ensemble of different models for improved performance
4. **Explainability**: Add visualization techniques (e.g., Grad-CAM) to highlight areas influencing classification

## Conclusion
Despite the small dataset, the model achieved promising results with 88.89% validation accuracy and 0.8000 F1 score. The use of transfer learning with EfficientNet-B0 and Apple's MPS for hardware acceleration enabled effective training even with limited data. The model provides a solid foundation for retinal image classification that could be further improved with more data and advanced techniques. 