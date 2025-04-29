# Deep-leaning-project# Variational Autoencoder for Image Generation

## Overview
This project implements a **Variational Autoencoder (VAE)** to generate images similar to those in the **CIFAR-10** dataset.

## Problem Statement
Traditional Autoencoders suffer from limited generative capability. VAEs introduce a probabilistic framework that allows for **meaningful latent representations**, making them suitable for generating realistic images.

## Goals
- Train a VAE to generate **CIFAR-10 style images**.
- Optimize the **latent space representation**.
- Improve image quality using CNN-based encoders and decoders.

## Dataset
- **CIFAR-10** dataset (60,000 images, 10 classes)
- Image size: **32x32x3**
- Preprocessing: **Normalization [-1,1]**

## Model Architecture
- **Encoder**: Extracts feature representation and estimates mean (μ) & variance (σ).
- **Latent Space**: Encodes compressed representation.
- **Decoder**: Reconstructs images from latent space.

## Training Details
- Loss Function: **MSE Loss + KL Divergence**
- Optimizer: **Adam**
- Batch Size: **128**
- Latent Dimension: **128**

## Results
 Successfully trained on CIFAR-10  
 Generated images with VAE  
 ## Data
- The CIFAR-10 dataset is downloaded automatically using torchvision.datasets.
- No manual download is needed. If necessary, you can manually download from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html) and place it in the `data/` folder.

## How to Run

Install the necessary packages:
pip install -r requirements.txt

Running Training
To train the VAE model:
python train.py

Generating and Comparing Images
After training, you can generate and compare images:
python generate_compare.py

Running the Streamlit Web App
To launch the web application for random generation and reconstruction:
streamlit run app.py



