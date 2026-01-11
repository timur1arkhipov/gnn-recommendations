"""Модели рекомендательных систем."""

from .base import BaseRecommender
from .orthogonal_bundle import OrthogonalBundleGNN, GroupShuffleLayer
from .baselines import (
    BPR_MF,
    LightGCN,
    GCNII,
    DGR,
    SVD_GCN,
    LayerGCN,
)

__all__ = [
    'BaseRecommender',
    'OrthogonalBundleGNN',
    'GroupShuffleLayer',
    'BPR_MF',
    'LightGCN',
    'GCNII',
    'DGR',
    'SVD_GCN',
    'LayerGCN',
]
