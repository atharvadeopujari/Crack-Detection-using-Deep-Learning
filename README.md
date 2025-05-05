# Crack Detection in Pipes Using Deep Learning

This project focuses on developing a deep learning-based solution for detecting cracks in pipeline images. Leveraging the U-Net architecture, the model is trained to segment crack regions, emphasizing crack tips, from pipeline images obtained via destructive testing.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Methodology](#methodology)
- [Dataset](#dataset)
  - [Data Augmentation Techniques](#data-augmentation-techniques)
- [Model Architecture](#model-architecture)
  - [Parameters](#parameters)
- [Mixed Precision Training](#mixed-precision-training)
- [Loss Functions](#loss-functions)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

## Introduction

Pipelines are crucial to industries like oil and gas, water supply, and chemical processing but are prone to cracks due to operational and environmental factors. Traditional inspection methods often lack efficiency and accuracy. This project applies deep learning techniques to automate crack detection and enhance reliability.

## Features
- **Semantic segmentation** of crack regions using U-Net architecture.
- Handles **complex crack patterns**, varying **lighting conditions**, and limited data availability.
- Experiments with **four loss functions**: Binary Cross-Entropy (BCE), Dice Loss, Focal Loss, and Tversky Loss.
- Implements advanced **data augmentation** techniques to improve model generalization.
- Optimized using **mixed precision training** and **adaptive learning rate scheduling**.

## Methodology
1. **Data Preparation**: 
   - Generated segmentation masks using MATLAB's Image Segmenter tool.
   - Applied data augmentation techniques, including flipping and resizing, to handle data limitations.
2. **Model Architecture**:
   - U-Net with a symmetric encoder-decoder structure.
   - Skip connections for preserving spatial information.
   - Flexible input dimensions and mixed precision training for computational efficiency.
3. **Training Scheme**:
   - Custom training loop with JIT compilation for improved execution speed.
   - Optimized using the Adam optimizer with a learning rate schedule.

## Dataset

The dataset contains images of pipelines from destructive testing. Each image includes a binary segmentation mask highlighting crack regions.

### Data Augmentation Techniques
- Horizontal and vertical flipping.
- Resizing to 512x640 pixels.
- Repeated images to enhance dataset diversity.

## Model Architecture

- **Encoder**: Downsampling with convolutional and max-pooling layers.
- **Bridge**: Intermediate layers connecting encoder and decoder.
- **Decoder**: Upsampling layers with skip connections for precise segmentation.

### Parameters
- Total trainable parameters: 1,735,361.
- Batch size: 16.

## Mixed Precision Training

To enhance computational efficiency and reduce memory usage, the model leverages **mixed precision training**. This involves using:
- **float16** precision for most computations, reducing memory overhead.
- **float32** precision for key operations to ensure numerical stability.

### Benefits of Mixed Precision Training
1. **Faster Training**: Reduces training time by utilizing hardware acceleration on GPUs like NVIDIA's Tensor Cores.
2. **Lower Memory Usage**: Enables the use of larger batch sizes without exceeding memory limits.
3. **Efficient Utilization of Resources**: Optimizes performance on modern hardware while maintaining model accuracy.

Mixed precision training was implemented using TensorFlow's `mixed_float16` policy and integrated seamlessly into the training pipeline.

## Loss Functions

The project experimented with four loss functions:
1. **Binary Cross-Entropy (BCE)**: Baseline approach, sensitive to class imbalance.
2. **Dice Loss**: Handles class imbalance effectively by focusing on overlap.
3. **Focal Loss**: Prioritizes harder-to-classify pixels.
4. **Tversky Loss**: Customizable to penalize false positives or negatives.

**Conclusion**: Dice Loss achieved the best performance with the highest mean Intersection over Union (mIoU).

## Evaluation Metrics

1. **Mean Intersection over Union (mIoU)**: Measures overlap between predicted and ground truth masks.
2. **Mean Pixel Accuracy**: Calculates the fraction of correctly classified pixels.

## Results

![Crack Detection Result](Images%20for%20Report/1401.png)
*Figure: Comparision of different models*

|Loss Function |Training mIoU |Validation mIoU |Train Loss |Validation Loss |Train Accuracy |Validation Accuracy |
| - | :- | :- | :- | :- | :- | :- |
|BCE Loss |0\.80943 |0\.81339 |0\.00071 |0\.00144 |0\.98991 |0\.98821 |
|Dice Loss |0\.99024 |0\.988796 |0\.00954 |0\.01109 |0\.99719 |0\.99689 |
|Focal Loss |0\.59103 |0\.59285 |0\.00015 |0\.00046 |0\.99883 |0\.99775 |
|Tversky Loss |0\.98945 |0\.98721 |0\.00787 |0\.00948 |0\.99544 |0\.99371 |

- Dice Loss achieved a training mIoU of **0.9902** and validation mIoU of **0.9887**.
- BCE, Focal Loss, and Tversky Loss were less effective in handling the class imbalance of crack vs. background pixels.

## Future Work

- **Incorporating Attention Mechanisms**: Use Attention U-Net to focus on critical regions like crack tips.
- **Combining Loss Functions**: Experiment with Gaussian Tip Loss and Euclidean Distance Loss.
- **Exploring Advanced Architectures**: Investigate Transformer-based models for enhanced performance.
- **Expanding the Dataset**: Increase dataset diversity through synthetic data generation and augmentation.

## References

For a detailed list of references, please refer to the project report.

---

This project was developed by **Atharva Deopujari** as part of **SECTD-BARC**, January 2025.
