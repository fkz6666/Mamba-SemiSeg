# Mamba-SemiSeg

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch 1.9+](https://img.shields.io/badge/PyTorch-1.9%2B-orange)

This is the official PyTorch implementation of the paper:

> **Mamba-SemiSeg: A Single-Stage Semi-Supervised Approach for Oceanic Internal Wave Segmentation in Remote Sensing Imagery**  
> [Kaizhe Feng](https://github.com/fkz6666), Dongfang Zhang, Chengjun Li, Yaqiong Yu, Junnan Guo, Lei Kuang, Yifan Zhao  
> *Submitted to International Journal of Applied Earth Observation and Geoinformation (JAG)*  

**Abstract:** Oceanic internal waves, as a representative mesoscale dynamic phenomenon, typically manifest in remote sensing imagery as elongated wave-like structures characterized by weak textures, blurred boundaries, and low contrast. To address the challenges of slender targets and scarce annotations, we propose Mamba-SemiSeg, an end-to-end semi-supervised segmentation approach based on the Mamba sequence modeling architecture. Our method employs a patch-wise Mamba encoder to capture long-range dependencies and introduces a pseudo-label generation mechanism to enhance the utilization of limited annotated samples.

## 🚀 Features

- **Mamba-based Architecture:** Leverages a selective state space model for efficient long-range dependency modeling of slender internal waves.
- **Semi-Supervised Learning:** Utilizes a pseudo-label generation mechanism to effectively learn from both labeled and unlabeled data.
- **Boundary Refinement:** Incorporates CRF post-processing and structural enhancement for improved segmentation continuity and boundary sharpness.
- **Reproducibility:** Complete training and evaluation code provided for replicating the results in the paper.

## 🧑‍🏫 Overall Architecture of Mamba-SemiSeg

The Mamba-SemiSeg architecture integrates a multi-scale encoder-decoder structure with a patch-wise Mamba encoder, designed to handle the segmentation of oceanic internal waves in remote sensing imagery. The key components of this architecture include:

1. **Patch-wise Mamba Encoder**: Efficiently captures long-range dependencies at the patch level, enabling the model to focus on the elongated wave-like structure typical of oceanic internal waves.
2. **Multi-scale Feature Fusion**: Combines features from different scales to enhance the model’s ability to detect waves at various resolutions.
3. **Decoder**: Uses skip connections to combine shallow features and multi-scale features, enabling high-resolution output segmentation maps.
4. **Pseudo-label Generation**: A semi-supervised approach where pseudo-labels are generated from unlabeled data, increasing the training data without requiring manual annotations.
5. **CRF Boundary Refinement**: A post-processing step that refines boundaries and ensures smooth segmentation transitions.
6. **Post-processing and Structural Enhancement**: Focuses on enhancing boundary transitions and structural details to improve segmentation performance.

### Architecture Diagram

![Mamba-SemiSeg Architecture](https://github.com/fkz6666/Mamba-SemiSeg/blob/main/path_to_your_image.png)

The diagram above shows the complete architecture of the Mamba-SemiSeg model, illustrating how each component works together to provide high-precision segmentation for oceanic internal waves.

---

## 🛠 Installation

To get started with Mamba-SemiSeg, clone the repository and install the required dependencies:

```bash
git clone https://github.com/fkz6666/Mamba-SemiSeg.git
cd Mamba-SemiSeg
pip install -r requirements.txt
