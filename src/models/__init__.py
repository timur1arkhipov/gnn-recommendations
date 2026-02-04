"""Модели рекомендательных систем."""

from .base import BaseRecommender
from .orthogonal_bundle import OrthogonalBundleGNN, GroupShuffleLayer
from .baselines import (
    LightGCN,
    NGCF,
    GAT,
    UltraGCN,
    KGTORe,
)

__all__ = [
    'BaseRecommender',
    'OrthogonalBundleGNN',
    'GroupShuffleLayer',
    'LightGCN',
    'NGCF',
    'GAT',
    'UltraGCN',
    'KGTORe',
]
