import torch
import pytest
from src.model import build_model, get_finetune_param_groups
from src.config import NUM_CLASSES


def test_model_output_shape():
    model = build_model(num_classes=NUM_CLASSES, freeze_base=True)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_frozen_base():
    model = build_model(freeze_base=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all('classifier' in n for n in trainable), "Only classifier should be trainable"


def test_unfrozen_model():
    model = build_model(freeze_base=False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    assert trainable == total


def test_finetune_param_groups():
    model  = build_model(freeze_base=False)
    groups = get_finetune_param_groups(model, lr_finetune=1e-5)
    assert len(groups) == 3
    lrs = [g['lr'] for g in groups]
    assert lrs[0] > lrs[1] > lrs[2]
