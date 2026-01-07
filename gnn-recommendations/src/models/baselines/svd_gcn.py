"""
SVD-GCN - GCN с использованием SVD декомпозиции.

Использует спектральную декомпозицию для более эффективной графовой свертки.

Ссылка: Метод основан на спектральных подходах к GCN
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..base import BaseRecommender


class SVD_GCN(BaseRecommender):
    """
    SVD-GCN модель - GCN с SVD декомпозицией adjacency matrix.
    
    Использует низкоранговое приближение adjacency matrix для эффективности.
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        rank: Optional[int] = None,
        dropout: float = 0.0,
        init_scale: float = 0.01
    ):
        """
        Инициализация SVD-GCN модели.
        
        Args:
            n_users: количество пользователей
            n_items: количество айтемов
            embedding_dim: размерность embeddings
            n_layers: количество слоев
            rank: ранг для SVD (если None, используется embedding_dim)
            dropout: вероятность dropout
            init_scale: масштаб инициализации параметров
        """
        super().__init__(n_users, n_items, embedding_dim)
        
        self.n_layers = n_layers
        self.rank = rank if rank is not None else embedding_dim
        self.dropout = dropout
        self.init_scale = init_scale
        
        # Начальные embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Веса для каждого слоя
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(embedding_dim, embedding_dim) * init_scale)
            for _ in range(n_layers)
        ])
        
        # SVD компоненты (будут вычислены при первом forward)
        self.register_buffer('U', None)
        self.register_buffer('S', None)
        self.register_buffer('V', None)
        self._svd_computed = False
        
        # Dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Инициализация параметров."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=self.init_scale)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=self.init_scale)
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
    
    def _compute_svd(self, adj_matrix: torch.Tensor):
        """Вычислить SVD декомпозицию adjacency matrix."""
        if self._svd_computed and self.U is not None:
            return
        
        # Преобразуем в dense, если sparse
        if adj_matrix.is_sparse:
            adj_dense = adj_matrix.to_dense()
        else:
            adj_dense = adj_matrix
        
        # SVD декомпозиция
        # Используем torch.linalg.svd (современный API)
        try:
            U, S, Vh = torch.linalg.svd(adj_dense, full_matrices=False)
            V = Vh.T  # Vh - это V^T, нужно транспонировать
        except AttributeError:
            # Для старых версий PyTorch используем torch.svd
            U, S, V = torch.svd(adj_dense)
        
        # Сохраняем только top-k компонент
        self.U = U[:, :self.rank]
        self.S = S[:self.rank]
        self.V = V[:, :self.rank]
        self._svd_computed = True
    
    def forward(
        self,
        adj_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass через SVD-GCN.
        
        Args:
            adj_matrix: normalized adjacency matrix [N, N]
        
        Returns:
            Tuple из (user_embeddings, item_embeddings)
        """
        # Вычисляем SVD при первом вызове
        self._compute_svd(adj_matrix)
        
        # Начальные embeddings
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        # Прохождение через слои
        for weight in self.weights:
            # Графовая свертка через SVD: A @ x ≈ U @ S @ V^T @ x
            # Более эффективно: U @ (S * (V^T @ x))
            Vtx = torch.mm(self.V.T, x)  # [rank, embedding_dim]
            SVtx = self.S.unsqueeze(1) * Vtx  # [rank, embedding_dim]
            x_conv = torch.mm(self.U, SVtx)  # [N, embedding_dim]
            
            # Линейное преобразование
            x = x_conv @ weight
            
            # Dropout
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)
        
        # Разделить обратно
        user_emb, item_emb = torch.split(
            x,
            [self.n_users, self.n_items],
            dim=0
        )
        
        return user_emb, item_emb
    
    def predict(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Предсказание scores."""
        if adj_matrix is None:
            raise ValueError("adj_matrix должен быть передан для SVD_GCN")
        user_emb, item_emb = self.get_all_embeddings(adj_matrix)
        scores = (user_emb[users] * item_emb[items]).sum(dim=1)
        return scores
    
    def get_all_embeddings(
        self,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Получить все embeddings."""
        if adj_matrix is None:
            raise ValueError("adj_matrix должен быть передан для SVD_GCN")
        return self.forward(adj_matrix)

