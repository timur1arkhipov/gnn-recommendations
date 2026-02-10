# src/models/orthogonal_bundle/bundle_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BundleConnectionLayer(nn.Module):
    """
    Bundle Connection Layer: вычисляет ортогональные матрицы связи W_{ij}
    для параллельного переноса вдоль рёбер графа.

    Ключевая идея:
    - Каждое ребро (i,j) имеет свою матрицу связи W_{ij}
    - W_{ij} ортогональна → сохраняет норму при переносе
    - Построена через механизм Group & Shuffle
    """

    def __init__(self, embedding_dim, block_size, n_blocks=None):
        """
        Args:
            embedding_dim: размерность эмбеддингов
            block_size: размер блоков для механизма Group
            n_blocks: количество блоков (если None, вычисляется автоматически)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.n_blocks = n_blocks or (embedding_dim // block_size)
        
        # Параметры для кососимметричных матриц (для каждого блока)
        self.skew_params = nn.ParameterList([
            nn.Parameter(torch.randn(block_size, block_size) * 0.01)
            for _ in range(self.n_blocks)
        ])

        # Перестановка для Shuffle (фиксированная для этого слоя)
        self.register_buffer('shuffle_perm', self._create_shuffle_permutation())

    def _create_shuffle_permutation(self):
        """Создаёт перестановку для shuffle"""
        return torch.randperm(self.embedding_dim)
    
    def forward(self, edge_index=None):
        """
        Строит ортогональную матрицу связи

        Args:
            edge_index: [2, num_edges] - опционально, для специфичных рёбрам матриц

        Returns:
            W_connection: [embedding_dim, embedding_dim] ортогональная матрица
        """
        # Строим блочно-диагональную ортогональную матрицу
        blocks = []

        for skew_param in self.skew_params:
            # Делаем кососимметричной: A = A - A^T
            A_skew = skew_param - skew_param.transpose(-2, -1)

            # Экспоненциальное отображение: W = exp(A_skew)
            # Это гарантирует ортогональность!
            W_block = torch.matrix_exp(A_skew)

            blocks.append(W_block)

        # Собираем блочно-диагональную матрицу
        W_orth = torch.block_diag(*blocks)

        # Применяем перестановку shuffle
        W_shuffled = W_orth[:, self.shuffle_perm]

        return W_shuffled
    
    def get_connection_matrix_for_edge(self, src_node, dst_node):
        """
        Получает матрицу связи для конкретного ребра
        (сейчас используется одна матрица для всех рёбер, может быть расширено)

        Args:
            src_node: индекс исходной вершины
            dst_node: индекс целевой вершины

        Returns:
            W: [embedding_dim, embedding_dim]
        """
        return self.forward()

    def get_orthogonality_metrics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисляет метрики ортогональности для матрицы связи.

        Returns:
            Tuple (frobenius_error, max_deviation)
        """
        W_orth = self.forward()
        identity = torch.eye(self.embedding_dim, device=W_orth.device, dtype=W_orth.dtype)
        diff = W_orth.T @ W_orth - identity
        fro_error = torch.norm(diff, p='fro')
        max_deviation = diff.abs().max()
        return fro_error, max_deviation


class EdgeSpecificBundleConnection(nn.Module):
    """
    Расширенная версия: разные матрицы связи для разных типов рёбер
    (например, user-item vs item-user)
    """

    def __init__(self, embedding_dim, block_size, n_edge_types=2):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Создаём отдельные слои связи для каждого типа рёбер
        self.connection_layers = nn.ModuleList([
            BundleConnectionLayer(embedding_dim, block_size)
            for _ in range(n_edge_types)
        ])

    def forward(self, edge_index, edge_type):
        """
        Args:
            edge_index: [2, num_edges]
            edge_type: [num_edges] - тип каждого ребра (0 или 1)

        Returns:
            W_connections: [num_edges, embedding_dim, embedding_dim]
        """
        num_edges = edge_index.size(1)
        device = edge_index.device

        # Получаем матрицы связи для каждого типа
        W_type_0 = self.connection_layers[0]()
        W_type_1 = self.connection_layers[1]()

        # Собираем для каждого ребра
        W_connections = torch.zeros(num_edges, self.embedding_dim, self.embedding_dim,
                                     device=device)
        
        mask_0 = (edge_type == 0)
        mask_1 = (edge_type == 1)
        
        W_connections[mask_0] = W_type_0.unsqueeze(0)
        W_connections[mask_1] = W_type_1.unsqueeze(0)
        
        return W_connections

