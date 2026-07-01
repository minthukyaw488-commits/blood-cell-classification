# Blood Cell Classification

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Automated 8-class blood cell subtype classification using EfficientNetB3 with two-phase transfer learning in PyTorch.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [License](#license)

## Overview

| Attribute     | Detail                                      |
|---------------|---------------------------------------------|
| Task          | Multi-class image classification            |
| Dataset       | Peripheral blood cell microscopy images     |
| Architecture  | EfficientNetB3 (pretrained on ImageNet)     |
| Classes       | 8 blood cell subtypes                       |
| Input size    | 224 x 224 RGB                               |
| Framework     | PyTorch + timm                              |

The training approach uses two phases: first training only the classification head on frozen ImageNet features, then performing full fine-tuning with discriminative learning rates (higher LR for the head, lower for earlier layers). Class imbalance is handled with `WeightedRandomSampler` and label smoothing (0.1) is applied via `CrossEntropyLoss`.

## Dataset

The model classifies 8 peripheral blood cell types, each with distinct clinical significance:

| Class         | Description                                                    |
|---------------|----------------------------------------------------------------|
| basophil      | Rarest granulocyte; involved in allergic responses             |
| eosinophil    | Granulocyte active in parasitic infections and allergies       |
| erythroblast  | Immature red blood cell precursor (nucleated)                  |
| ig            | Immature granulocytes; elevated in infection or bone marrow disorders |
| lymphocyte    | Key adaptive immunity cell; B and T cell lineages             |
| monocyte      | Large mononuclear phagocyte; precursor to macrophages          |
| neutrophil    | Most abundant WBC; first responder to bacterial infection      |
| platelet      | Thrombocyte fragment essential for clotting                    |

Data is split 70% train / 15% validation / 15% test with a fixed random seed (42) for reproducibility.

## Model Architecture

**Backbone:** EfficientNetB3 pretrained on ImageNet (via `timm`), fine-tuned end-to-end.

**Two-Phase Training:**

1. **Phase 1 — Head Training:** The backbone is frozen; only the final linear classifier is trained with Adam at `lr=1e-3` for up to 10 epochs (early stopping, patience=5). This quickly adapts the classification head to the blood cell domain.

2. **Phase 2 — Full Fine-tuning:** All layers are unfrozen and trained with discriminative learning rates via `CosineAnnealingLR`:
   - Classifier head: `lr = 1e-4`
   - Later blocks: `lr = 1e-5`
   - Early layers: `lr = 1e-6`

Mixed-precision training (`torch.cuda.amp`) is enabled automatically when a CUDA GPU is available.

## Results

Results vary by hardware, random seed, and dataset version. The table below shows representative placeholder values; run `train.py` to obtain results on your data.

| Metric            | Value (example) |
|-------------------|-----------------|
| Test Accuracy     | ~97%            |
| Weighted F1       | ~0.97           |
| Best Val Accuracy | logged per run  |

Training produces `training_history.png` (loss and accuracy curves across both phases) and `confusion_matrix.png` (counts and normalized).

## Project Structure

```
blood-cell-classification/
├── blood_cell_classification_8class.ipynb   # Original exploratory notebook
├── train.py                                 # CLI: end-to-end training pipeline
├── predict.py                               # CLI: single-image inference
├── requirements.txt                         # Python dependencies
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py      # Hyperparameters and constants
│   ├── data.py        # Dataset, transforms, DataLoader construction
│   ├── model.py       # EfficientNetB3 builder and param-group helper
│   ├── train.py       # train_one_epoch, evaluate, run_training
│   └── evaluate.py    # Plotting, metrics, Grad-CAM
├── tests/
│   ├── __init__.py
│   ├── test_model.py  # Unit tests for model building and param groups
│   └── test_data.py   # Unit tests for transforms and dataset splitting
└── .github/
    └── workflows/
        └── ci.yml     # Lint (ruff) + test (pytest) on push/PR
```

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU training is supported but slow)

### Installation

```bash
git clone https://github.com/<your-username>/blood-cell-classification.git
cd blood-cell-classification
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Training

```bash
# Basic run (uses defaults from src/config.py)
python train.py --data_dir /path/to/bloodcells_dataset

# Custom epochs and checkpoint path
python train.py --data_dir /path/to/bloodcells_dataset --epochs 30 --checkpoint my_model.pth
```

The dataset directory must follow `torchvision.datasets.ImageFolder` layout:

```
bloodcells_dataset/
├── basophil/
├── eosinophil/
├── erythroblast/
├── ig/
├── lymphocyte/
├── monocyte/
├── neutrophil/
└── platelet/
```

Outputs saved to the working directory: `best_model.pth`, `training_history.png`, `confusion_matrix.png`.

### Inference

```bash
python predict.py --image /path/to/cell_image.jpg
python predict.py --image /path/to/cell_image.jpg --checkpoint my_model.pth
```

Outputs: console probability table and `prediction_result.png`.

### Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Configuration

All key hyperparameters live in `src/config.py`:

| Parameter     | Default | Description                                  |
|---------------|---------|----------------------------------------------|
| `SEED`        | 42      | Global random seed for reproducibility        |
| `IMG_SIZE`    | 224     | Input image resolution (pixels)               |
| `BATCH_SIZE`  | 32      | Mini-batch size                               |
| `EPOCHS`      | 20      | Maximum fine-tuning epochs (Phase 2)          |
| `LR`          | 1e-3    | Phase 1 learning rate (head only)             |
| `LR_FINETUNE` | 1e-5    | Base learning rate for Phase 2 fine-tuning    |
| `NUM_CLASSES` | 8       | Number of output classes                      |
| `TRAIN_SPLIT` | 0.70    | Fraction of data used for training            |
| `VAL_SPLIT`   | 0.15    | Fraction of data used for validation          |
| `TEST_SPLIT`  | 0.15    | Fraction of data used for final evaluation    |

## License

This project is licensed under the MIT License.

## Author

**NOVEM (MIN THU KYAW)**
Medical AI · Konyang University, Daejeon, South Korea
