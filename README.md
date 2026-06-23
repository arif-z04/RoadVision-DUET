# Bangladesh Vehicle Detection Competition

Object detection pipeline for Bangladesh highway surveillance imagery using:

- YOLO (Ultralytics)
- SAHI Sliced Inference
- Weighted Box Fusion (WBF)
- Stratified K-Fold Cross Validation
- Test Time Augmentation (TTA)
- Pseudo Labeling

---

# System Requirements

## Minimum

- Python 3.10+
- 8 GB RAM
- NVIDIA GPU (recommended)

## Recommended

- Python 3.11
- 16+ GB RAM
- NVIDIA GPU with CUDA support
- Google Colab T4 / RTX GPU

---

# Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

---

# Windows Setup

## 1. Install Python

Download Python:

https://www.python.org/downloads/

Verify:

```powershell
python --version
```

---

## 2. Create Virtual Environment

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\activate
```

---

## 3. Upgrade Pip

```powershell
python -m pip install --upgrade pip
```

---

## 4. Install PyTorch

### CUDA GPU

Visit:

https://pytorch.org/get-started/locally/

Example:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### CPU Only

```powershell
pip install torch torchvision torchaudio
```

---

## 5. Install Project Dependencies

```powershell
pip install -r requirements.txt
```

---

# Linux Setup

Tested on Ubuntu 22.04+

## 1. Install Python

```bash
sudo apt update

sudo apt install python3 python3-pip python3-venv -y
```

Verify:

```bash
python3 --version
```

---

## 2. Create Virtual Environment

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

---

## 3. Upgrade Pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install PyTorch

### CUDA GPU

Check GPU:

```bash
nvidia-smi
```

Install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### CPU Only

```bash
pip install torch torchvision torchaudio
```

---

## 5. Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements File

Create `requirements.txt`

```txt
ultralytics>=8.3.0

torch
torchvision
torchaudio

numpy
pandas

opencv-python

matplotlib
seaborn

scikit-learn

sahi

ensemble-boxes

iterative-stratification

Pillow

tqdm

jupyter
ipykernel
```

---

# Verify Installation

Run:

```bash
python
```

Then:

```python
import torch
import ultralytics
import cv2
import pandas as pd
import numpy as np
import sahi

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
```

Expected:

```text
Torch: x.x.x
CUDA: True
```

---

# Jupyter Notebook

Install:

```bash
pip install jupyter
```

Start:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Open:

```text
duet-competition.ipynb
```

---

# GPU Check

## Windows

```powershell
nvidia-smi
```

## Linux

```bash
nvidia-smi
```

Example:

```text
GPU Name: NVIDIA RTX 2050
Memory: 4096 MiB
```

---

# Dataset Structure

```text
dataset/
│
├── train/
│   ├── *.jpg
│
├── test/
│   ├── *.jpg
│
├── train.csv
│
└── sample_submission.csv
```

---

# Recommended Training Configurations

## RTX 2050 4GB

```python
MODEL = "yolo11s.pt"
IMGSZ = 640
BATCH = 4
EPOCHS = 100
```

---

## Google Colab T4

```python
MODEL = "yolo11m.pt"
IMGSZ = 960
BATCH = 8
EPOCHS = 150
```

---

# Common Issues

## OpenCV Import Error

```text
ModuleNotFoundError: No module named 'cv2'
```

Fix:

```bash
pip install opencv-python
```

---

## SAHI Missing

```text
ModuleNotFoundError: No module named 'sahi'
```

Fix:

```bash
pip install sahi
```

---

## WBF Missing

```text
ModuleNotFoundError: No module named 'ensemble_boxes'
```

Fix:

```bash
pip install ensemble-boxes
```

---

## CUDA Not Detected

Check:

```bash
nvidia-smi
```

If CUDA is unavailable:

```python
import torch
print(torch.cuda.is_available())
```

returns:

```text
False
```

Reinstall the correct PyTorch CUDA build.

---

# Reproduce Environment

Export:

```bash
pip freeze > requirements-lock.txt
```

Restore:

```bash
pip install -r requirements-lock.txt
```

---

# Notebook Features

- Stratified K-Fold Training
- YOLO Training
- SAHI Inference
- Test-Time Augmentation
- Weighted Box Fusion
- Pseudo Labeling
- Competition Submission Generation
