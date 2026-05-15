# Voxera-ISL-RealTime-Translator
A real-time Indian Sign Language (ISL) translator using an auto-scaling ResNet18 architecture and multi-threaded TTS.
# Voxera AI: Real-Time ISL Translation System

Voxera is an end-to-end Computer Vision system designed to translate Indian Sign Language (ISL) into spoken English. It utilizes a custom-trained ResNet18 model and a multi-threaded inference engine to achieve real-time performance on edge devices.

## 🚀 Key Features
- **35-Class Support:** Covers ISL Alphabets (A-Z) and Numerals (1-9).
- **Auto-Scaling Architecture:** Dynamically adjusts the Neural Network head based on dataset directory structure.
- **Aspect-Ratio Correction:** Algorithmic 1:1 center-cropping to eliminate webcam "squish" distortion.
- **Asynchronous TTS:** Multi-threaded Text-to-Speech allows uninterrupted visual inference.

## 📁 Repository Structure

- `train.py`: The training pipeline with Automatic Mixed Precision (AMP).
- `webcam.py`: The live inference script for presentation.


## 📊 Dataset
The 35-Class ISL dataset used for this project is available on Kaggle:
https://www.kaggle.com/datasets/prabheeshsingh/isl-letters-a-to-z-and-numbers-1-9

## 🛠️ Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Voxera-ISL-RealTime-Translator.git](https://github.com/YOUR_USERNAME/Voxera-ISL-RealTime-Translator.git)
