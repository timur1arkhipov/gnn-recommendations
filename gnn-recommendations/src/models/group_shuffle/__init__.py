"""Group and Shuffle модель для GNN рекомендательных систем."""

from .layers import GroupShuffleLayer
from .model import GroupShuffleGNN

__all__ = [
    'GroupShuffleLayer',
    'GroupShuffleGNN',
]
