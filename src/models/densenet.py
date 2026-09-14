"""
DenseNet-121 for 8-class dermoscopy classification.

This architecture is deliberately standard and unmodified. Keeping it conventional is a
design decision, not a shortcut: if rare-class performance improves, that improvement must
be attributable to the aggregation mechanism rather than to a stronger backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(n_classes: int = 8, pretrained: bool = True,
                freeze_backbone: bool = False) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, n_classes)
    return model


def classifier_keys(model: nn.Module) -> list[str]:
    """State-dict keys belonging to the classifier head.

    The class-conditional method aggregates the head row-by-row using per-class weights,
    while the feature extractor aggregates by standard FedAvg. This function marks the
    boundary between those two treatments.
    """
    return [k for k in model.state_dict().keys() if k.startswith("classifier")]


def split_state_dict(state: dict, head_keys: list[str]) -> tuple[dict, dict]:
    """Split a state dict into (feature extractor, classifier head)."""
    head = {k: v for k, v in state.items() if k in head_keys}
    body = {k: v for k, v in state.items() if k not in head_keys}
    return body, head
