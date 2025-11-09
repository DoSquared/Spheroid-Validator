# Spheroid Validator (Valid vs. Invalid)

Binary image classifier that labels 3D spheroid images as **valid** or **invalid** using a fine-tuned ResNet-50.  
Includes an inference script that scans a `Test/` folder, predicts a class for each image, and writes results to CSV.

## Overview

This repository provides a ResNet50 trained model to classify spheroid images into **valid** or **invalid**.  
Images are preprocessed and passed through a fine-tuned ResNet-50.  
Predictions are saved to `predictions.csv` with class label and confidence.

## Features

- ResNet-50 backbone with a custom classification head (dropout + linear).
- Batch inference over a folder (`./Test/`) of images.
- Outputs a CSV with `filename`, `predicted_class`, `confidence`.
- Flexible class naming:
  - Auto-detect from `Dataset/train/<class_name>/...` (ImageFolder layout), or
  - Provide explicitly via `--classes "invalid,valid"`.
- Supports common image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`.

## Setup and Installation

1) Python 3.9–3.12 recommended.  
2) Install dependencies:
```bash
pip install torch torchvision pillow



