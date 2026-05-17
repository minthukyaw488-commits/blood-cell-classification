import torch
import pytest
from PIL import Image
import tempfile
import os
from src.data import DatasetWithTransform, get_transforms
from src.config import CLASS_NAMES, IMG_SIZE


def _make_fake_dataset(root, class_names, n_per_class=5):
    for cls in class_names:
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir)
        for i in range(n_per_class):
            img = Image.new('RGB', (64, 64), color=(i * 20, 100, 200))
            img.save(os.path.join(cls_dir, f'{i}.jpg'))


def test_transforms_output_shape():
    train_t, val_t = get_transforms()
    img = Image.new('RGB', (100, 100))
    tensor = val_t(img)
    assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)


def test_dataset_with_transform_length():
    train_t, _ = get_transforms()
    samples = [('/fake/path.jpg', 0), ('/fake/path2.jpg', 1)]
    ds = DatasetWithTransform(samples, transform=None)
    assert len(ds) == 2


def test_build_dataloaders_splits():
    """Integration test using a tiny fake dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_fake_dataset(tmpdir, CLASS_NAMES, n_per_class=5)
        from src.data import build_dataloaders
        train_loader, val_loader, test_loader = build_dataloaders(tmpdir)
        total = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
        assert total == len(CLASS_NAMES) * 5
