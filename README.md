# MalImg Classifier: Malware Classification using Computer Vision

A project that explores the use of Convolutional Neural Networks (CNNs) to classify malware families by visualizing executable binaries as grayscale images.

---

## 📖 Table of Contents

* [Core Concept](#-core-concept)
* [Technical Stack](#-technical-stack)
* [Project Structure](#-project-structure)
* [Model Architecture](#-model-architecture)
* [Usage](#-usage)
* [Model Limitations & Vulnerabilities](#-model-limitations--vulnerabilities)

---

## 🔬 Core Concept

This project bypasses traditional code analysis by transforming the problem from binary analysis to image classification.

1.  **Binary to Image:** An executable file is read as a 1D stream of 8-bit bytes (0-255). This stream is then reshaped into a 2D matrix (a grayscale image), where each byte becomes a pixel.
2.  **Image Classification:** A Convolutional Neural Network (CNN) is trained on a labeled dataset of these images to learn the unique visual textures and structural patterns associated with different malware families.



## 💻 Technical Stack

This project is built using the following core libraries:

* **Torch:** `2.9.0+cu128`
* **Torchvision:** `0.24.0+cu128`
* **Numpy:** `1.26.4`
* **Matplotlib:** `3.10.7` (for visualization)

---

## 📁 Project Structure

The repository is organized as follows:

INTERNSHIP_PROJECT/ 
├── model/ # Output directory for final, saved model weights (.pth) 
├── old weights/ # Archive of previous model checkpoints 
├── test/ # Image dataset for testing 
├── train/ # Image dataset for training └
── val/ # Image dataset for validation 
├── .gitignore # Git ignore file 
├── check_gpu.py # Utility script to verify CUDA availability
├── clear_cache.py # Utility script to clear pycache 
├── MalImgCNN_Training.py # The main script for training and validating the CNN 
├── multidataset.py # Custom PyTorch Dataset and DataLoader script 
├── overhead_ensemble.py # Script for ensembling or advanced model combination 
├── README.md # This file 
├── test_showcase_MAIN.py # Main script for inference and showcasing model results 
└── versions.txt # File listing all project dependencies

## 🤖 Model Architecture

The classifier is a **Convolutional Neural Network (CNN)**, an architecture specifically designed to find hierarchical patterns in images.

### How It Works:

The model processes an image in two main stages: feature extraction and classification.

**1. Feature Extraction (Convolutional Base):**

This stage consists of several repeating blocks to learn visual features, from simple edges to complex textures. A typical block looks like this:

* **`Conv2d` (Convolution):** A set of filters (kernels) slide over the input image. Each filter is trained to detect a specific pattern (like a horizontal edge or a specific texture). This produces a stack of "feature maps."
* **`ReLU` (Activation):** An activation function is applied to each pixel of the feature maps. It introduces non-linearity, allowing the model to learn complex relationships (it simply turns all negative values to zero).
* **`MaxPool2d` (Max Pooling):** This layer downsamples the feature maps. It slides a window (e.g., 2x2) over the map and keeps only the *maximum* value in that window.
    * **Purpose:** This reduces the spatial dimensions (width & height), decreasing computational cost. It also makes the model more robust by providing "local translation invariance" (it doesn't matter *exactly* where the feature is, as long as it's present in the general area).


**2. Classification (Head):**

After several convolutional and pooling layers, the feature maps are still 2D. To classify them, we must convert them into a 1D vector.

* **`GlobalAveragePooling2d` (Global Average Pooling):** Instead of using a traditional `Flatten` layer (which can have millions of parameters and lead to overfitting), we use Global Average Pooling.
    * **Purpose:** This layer takes each feature map in the final stack and calculates its *average value*, outputting a single number per map. If the previous layer produced 512 feature maps, this layer outputs a 512-element vector. This drastically reduces parameters.
* **`Linear` (Fully Connected Layer):** This 1D vector is fed into one or more dense layers to perform the final classification.
* **`Softmax` (Output):** The final layer outputs a probability score for each of the possible malware families.

This "Conv -> Pool -> GAP" structure is modern, efficient, and less prone to overfitting than older CNN designs.

---

## 🚀 Usage

*(Note: Data in `train/`, `test/`, and `val/` is assumed to be pre-converted to images.)*

### 1. Check GPU Setup

Verify that PyTorch can detect your GPU.
```bash
python check_gpu.py