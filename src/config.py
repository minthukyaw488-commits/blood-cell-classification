import os
from dataclasses import dataclass, field
from typing import List

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
LR_FINETUNE = 1e-5
NUM_CLASSES = 8
CLASS_NAMES = [
    'basophil', 'eosinophil', 'erythroblast', 'ig',
    'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
]
COLORS = [
    '#8338EC', '#E63946', '#FF6B6B', '#F4A261',
    '#457B9D', '#2A9D8F', '#E9C46A', '#264653'
]
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
