"""MobileNetV2 pre-entraine sur ImageNet, adapte a la classification real / spoof.

Transfer learning : on reutilise l'extracteur de caracteristiques appris sur
1,2 million d'images (ImageNet) et on remplace uniquement la derniere couche
(1000 classes ImageNet -> 2 classes : real, spoof).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

NUM_CLASSES = 2


def build_model(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features          # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Gele (False) ou degele (True) l'extracteur de caracteristiques."""
    for p in model.features.parameters():
        p.requires_grad = trainable


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


@torch.no_grad()
def predict_proba(model: nn.Module, loader) -> tuple[torch.Tensor, torch.Tensor]:
    """Renvoie (probabilite spoof, etiquettes) sur tout un DataLoader."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x), dim=1)[:, 1])
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)
