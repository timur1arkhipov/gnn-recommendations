"""Модели рекомендательных систем."""

from .base import BaseRecommender
from .group_shuffle import GroupShuffleGNN, GroupShuffleLayer
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
    'GroupShuffleGNN',
    'GroupShuffleLayer',
    'BPR_MF',
    'LightGCN',
    'GCNII',
    'DGR',
    'SVD_GCN',
    'LayerGCN',
]
