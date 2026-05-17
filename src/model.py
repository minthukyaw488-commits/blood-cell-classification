import torch
import torch.nn as nn
import timm

from src.config import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, freeze_base: bool = True) -> nn.Module:
    model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
    if freeze_base:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    return model


def get_finetune_param_groups(model: nn.Module, lr_finetune: float):
    return [
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n],
         'lr': lr_finetune * 10},
        {'params': [p for n, p in model.named_parameters()
                    if 'blocks' in n and 'classifier' not in n],
         'lr': lr_finetune},
        {'params': [p for n, p in model.named_parameters()
                    if 'blocks' not in n and 'classifier' not in n],
         'lr': lr_finetune * 0.1},
    ]
