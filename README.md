# Mamba-SemiSeg

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch 1.9+](https://img.shields.io/badge/PyTorch-1.9%2B-orange)

This is the official PyTorch implementation of the paper:

> **Mamba-SemiSeg: A Single-Stage Semi-Supervised Approach for Oceanic Internal Wave Segmentation in Remote Sensing Imagery**  
> [Kaizhe Feng](https://github.com/your_profile), [Dongfang Zhang](https://github.com/your_profile), Chengjun Li, Yaqiong Yu, Junnan Guo, Lei Kuang, Yifan Zhao  
> *Submitted to International Journal of Applied Earth Observation and Geoinformation (JAG)*  

**Abstract:** Oceanic internal waves, as a representative mesoscale dynamic phenomenon, typically manifest in remote sensing imagery as elongated wave-like structures characterized by weak textures, blurred boundaries, and low contrast. To address the challenges of slender targets and scarce annotations, we propose Mamba-SemiSeg, an end-to-end semi-supervised segmentation approach based on the Mamba sequence modeling architecture. Our method employs a patch-wise Mamba encoder to capture long-range dependencies and introduces a pseudo-label generation mechanism to enhance the utilization of limited annotated samples.

## 🚀 Features

- **Mamba-based Architecture:** Leverages a selective state space model for efficient long-range dependency modeling of slender internal waves.
- **Semi-Supervised Learning:** Utilizes a pseudo-label generation mechanism to effectively learn from both labeled and unlabeled data.
- **Boundary Refinement:** Incorporates CRF post-processing and structural enhancement for improved segmentation continuity and boundary sharpness.
- **Reproducibility:** Complete training and evaluation code provided for replicating the results in the paper.

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/fkz6666/Mamba-SemiSeg.git
cd Mamba-SemiSeg

2. Install dependencies (a detailed requirements.txt will be provided soon):
```bash
pip install torch torchvision opencv-python matplotlib numpy tqdm
