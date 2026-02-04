"""Модуль обучения моделей рекомендательных систем."""

from .losses import BPRLoss, BCELoss, RegularizedLoss
from .metrics import (
    recall_at_k,
    ndcg_at_k,
    precision_at_k,
    coverage,
    compute_all_metrics
)
from .trainer import Trainer

__all__ = [
    'BPRLoss',
    'BCELoss',
    'RegularizedLoss',
    'recall_at_k',
    'ndcg_at_k',
    'precision_at_k',
    'coverage',
    'compute_all_metrics',
    'Trainer',
]
