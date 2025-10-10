# SMART-OM Dataset Technical Validation
---

## Table of Contents

1. [Abstract](#abstract)
2. [Repository Structure](#repository-structure)
3. [Environment and Dependencies](#environment-and-dependencies)
4. [Detailed File Descriptions](#detailed-file-descriptions)

   * [Data Analysis and Split.ipynb](#data-analysis-and-splitipynb)
   * [Training.ipynb](#trainingipynb)
   * [Hyperparameter Tuning.ipynb](#hyperparameter-tuningipynb)
   * [Testing.ipynb](#testingipynb)

---

## Abstract

This repository contains code and notebooks for **data preparation**, **training**, **hyperparameter optimization**, and **evaluation** of convolutional neural network models applied to an **oral pathology image dataset**.

---

## Repository Structure

```plaintext
.
├── Data Analysis and Split.ipynb     # Data exploration, cleaning, and split creation
├── Training.ipynb                    # Model construction and supervised training
├── Hyperparameter Tuning.ipynb       # Automated or manual tuning of training parameters
├── Testing.ipynb                     # Model evaluation, metrics, and visualization
└── README.md                         # Documentation file (this file)
```

Each notebook is modular, allowing independent execution or full-pipeline runs.

---

## Environment and Dependencies

The following dependencies are required for execution:

```bash
python >= 3.8
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
torchvision
Pillow
opencv-python
tqdm
jupyterlab
```

### Example Conda Environment Setup

```bash
conda create -n oralpath python=3.9 numpy pandas matplotlib seaborn scikit-learn pillow opencv tqdm jupyterlab -y
conda activate oralpath
pip install torch torchvision
```

---

## Detailed File Descriptions

### `Data Analysis and Split.ipynb`

**Purpose:** Handles **data exploration**, **class distribution analysis**, and **creation of training/validation/test splits**.

**Core Components:**

* Loads image files and metadata using `pandas`, `glob`, and `cv2`.
* Generates visualizations (histograms, sample image grids, class ratios).
* Creates reproducible train/val/test splits (using `train_test_split` with stratification).
* Exports CSV manifests: `image_path`, `label`.

**Outputs:**

* Summary statistics and data distribution plots.
* CSV manifests for dataset splits.

---

### `Training.ipynb`

**Purpose:** Implements the **end-to-end model training pipeline** for oral pathology classification using ResNet architectures.

**Key Features:**

* Custom `ImageDataset` class for image–label loading and transformations.
* Torch-based data loaders for training.
* Training loops with loss tracking and metric logging.
* Model checkpoint saving and visualization of training dynamics.
* Includes options for transfer learning or training from scratch.

**Functions Defined:**

* `set_seed(seed)` — ensures reproducibility.
* `get_resnet50(num_classes)` — builds and initializes ResNet models.
* `train_epoch()` and — one-epoch training routines.
* `run_training()` — orchestrates full training cycle.

**Outputs:**

* Saved model weights.
* Loss/accuracy plots across epochs.

---

### `Hyperparameter Tuning.ipynb`

**Purpose:** Provides framework for **systematic exploration** of training hyperparameters.

**Includes:**

* Grid and manual search loops over learning rates, batch sizes, optimizers, etc.
* Performance logging and visualization of hyperparameter effects.
* Integration with `run_training()` for consistent evaluation.

**Usage:**

1. Define parameter grid (example: learning rate, batch size).
2. Run tuning loop.
3. Analyze and record best-performing configuration.

**Outputs:**

* Tabulated results (`pandas.DataFrame`).
* Visual comparisons of accuracy/loss across hyperparameters.

---

### `Testing.ipynb`

**Purpose:** Performs **inference and evaluation** on the held-out test set using trained models.

**Core Components:**

* Model loading from checkpoints.
* Prediction generation with batch-wise inference.
* Computation of accuracy, precision, recall, F1-score, and ROC-AUC.
* Visualization: confusion matrix, ROC curves, per-class metrics.
* CSV export of predictions and probabilities.

**Outputs:**

* Quantitative evaluation report.
* Diagnostic plots for qualitative analysis.

---

## Quick Start Guide

### Step 1 — Data Preparation

Run the following notebook sequentially:

```bash
jupyter notebook "Data Analysis and Split.ipynb"
```

Ensure the dataset path variables are correctly set.

### Step 2 — Hyperparameter Search (Optional)

```bash
jupyter notebook "Hyperparameter Tuning.ipynb"
```

Use to fine-tune configurations before final model training.


### Step 3 — Model Training

```bash
jupyter notebook "Training.ipynb"
```

* Adjust hyperparameters as needed.
* Outputs model weights and logs to specified folder.

### Step 4 — Model Evaluation

```bash
jupyter notebook "Testing.ipynb"
```

Generates metrics, plots, and prediction CSVs.

---

## Authorship and Contact

**Maintainer:** *Anwesh Nayak*
**Affiliation:** *International Institute of Information Technology, Bangalore*
**Contact:** *anwesh.Nayak@iiitb.ac.in*
