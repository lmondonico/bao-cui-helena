"""Loss functions for CE prediction tasks."""

import torch.nn as nn


def get_regression_loss(delta=1.0):
    """Huber loss for robust regression."""
    return nn.HuberLoss(delta=delta)


def get_classification_loss():
    return nn.CrossEntropyLoss()
