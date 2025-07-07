# HCA-Former: A Context-Aware Transformer for Adaptive Thermal Comfort Prediction

This is the official **PyTorch implementation** of the paper:

> **Sundas Sarwar, Jerry Chun-Wei Lin, Lars Arne Jordanger, Stefano Izzo, Daniela Annunziata, Francesco Piccialli**  

---

## Abstract

Ensuring occupant well-being, improving energy efficiency, and optimizing indoor environments based on accurate thermal comfort predictions constitute key objectives in smart buildings. However, traditional models such as the Predicted Mean Value (PMV) and adaptive approaches often rely on static assumptions and fail to capture the complex interplay between environmental conditions, occupant preferences, and contextual factors such as climate, season, and building characteristics.

We present **HCA-Former**, a transformer-based framework that introduces **Hierarchical Context Attention (HCA)** and **Multi-Head Self-Attention (MHSA)** to dynamically model dependencies across contextual layers. The proposed model eliminates redundancy and enhances generalization by capturing intricate interrelationships among thermal comfort features.

###  Key Contributions

- **Multiclass prediction** of thermal comfort levels.
- **Context-aware deep learning** with transformer architecture.
-  **Scalable** across building types, climates, and seasons.
-  Robust performance verified via ROC and confusion matrix analysis.
-  Ideal for **smart HVAC systems** and real-time thermal comfort control.


---

## Setup & Installation

Install dependencies using **Conda**.


# 1. Create and activate a conda environment
```bash
conda env create -f environment.yml
```

or follow these instructions

```bash
conda create -n hcaformer_env python=3.8 -y
conda activate hcaformer_env
conda install -c conda-forge pytorch pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

## Requirements
Before running the project, ensure have the following dependencies installed:
- Python
- PyTorch
- CUDA Toolkit (for GPU support)
- imbalanced-learn
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn


## Execution Steps

This project consists of two primary steps: **data preprocessing** and **model training with cross-validation**.


### Step 1: Preprocess the Data

Use `ASHRAE_preprocessing.py` to clean and prepare the ASHRAE dataset for training.
```bash
python -u ASHRAE_preprocessing.py
```

Instead, use `Scales_Dataset_preprocessing.py` to clean and prepare the Scales_Dataset for training
```bash
python -u Scales_Dataset_preprocessing.py
```

### Step 2: Train and Evaluate the Model
Run `ASHRAE_train.py` to train and evaluate the model on ASHRAE dataset using k-fold cross-validation.
```bash
python -u ASHRAE_train.py
```

Instead, run `Scales_Dataset_train.py` to train and evaluate the model on Scales_Dataset dataset using k-fold cross-validation.
```bash
python -u Scales_Dataset_train.py
```

## Note

This project requires the following datasets:

1. **ASHRAE Global Thermal Comfort Database II**
2. **Scales Project Dataset**

**Instructions**:

- Both datasets are provided in a zip archive.
- Unzip the archive and **place the extracted contents into the** `./data/` **folder**.





