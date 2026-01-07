"""Data pipeline module"""

from .dataset import RecommendationDataset
from .preprocessing import (
    filter_by_min_interactions,
    binarize_interactions,
    normalize_ids,
    remove_duplicates,
    get_statistics
)
from .graph_builder import (
    build_bipartite_graph,
    normalize_adjacency_matrix,
    convert_to_torch_sparse,
    save_adjacency_matrix,
    load_adjacency_matrix
)
from .loaders import get_loader, BaseDatasetLoader, LOADER_REGISTRY

__all__ = [
    'RecommendationDataset',
    'filter_by_min_interactions',
    'binarize_interactions',
    'normalize_ids',
    'remove_duplicates',
    'get_statistics',
    'build_bipartite_graph',
    'normalize_adjacency_matrix',
    'convert_to_torch_sparse',
    'save_adjacency_matrix',
    'load_adjacency_matrix',
    'get_loader',
    'BaseDatasetLoader',
    'LOADER_REGISTRY',
]
