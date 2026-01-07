"""
GCNII - GCN с residual connections и identity mapping.

Использует residual connections для борьбы с over-smoothing.
Классический метод для глубоких GNN.

Ссылка: Chen et al. "Simple and Deep Graph Convolutional Networks" (2020)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..base import BaseRecommender


class GCNII(BaseRecommender):
    """
    GCNII модель - GCN с residual connections и identity mapping.
    
    Архитектура:
    1. Начальные embeddings
    2. Графовая свертка с residual connections
    3. Identity mapping для сохранения информации
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        alpha: float = 0.1,
        beta: float = 0.5,
        dropout: float = 0.0,
        init_scale: float = 0.01
    ):
        """
        Инициализация GCNII модели.
        
        Args:
            n_users: количество пользователей
            n_items: количество айтемов
            embedding_dim: размерность embeddings
            n_layers: количество слоев
            alpha: коэффициент для identity mapping (0.0-1.0)
            beta: коэффициент для residual connection (0.0-1.0)
            dropout: вероятность dropout
            init_scale: масштаб инициализации параметров
        """
        super().__init__(n_users, n_items, embedding_dim)
        
        self.n_layers = n_layers
        self.alpha = alpha
        self.beta = beta
        self.dropout = dropout
        self.init_scale = init_scale
        
        # Начальные embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Веса для каждого слоя (в GCNII есть веса)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(embedding_dim, embedding_dim) * init_scale)
            for _ in range(n_layers)
        ])
        
        # Dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
        
        # Инициализация
        self.reset_parameters()
    
    def reset_parameters(self):
        """Инициализация параметров."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=self.init_scale)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=self.init_scale)
        
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
    
    def forward(
        self,
        adj_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass через GCNII.
        
        Args:
            adj_matrix: normalized adjacency matrix [N, N]
        
        Returns:
            Tuple из (user_embeddings, item_embeddings)
        """
        # Начальные embeddings
        x_init = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)  # [N, embedding_dim]
        
        x = x_init
        
        # Прохождение через слои
        for i, weight in enumerate(self.weights):
            # Графовая свертка
            if adj_matrix.is_sparse:
                x_conv = torch.sparse.mm(adj_matrix, x)
            else:
                x_conv = torch.mm(adj_matrix, x)
            
            # Линейное преобразование
            x_transformed = x_conv @ weight
            
            # Identity mapping: x = (1 - alpha) * x_transformed + alpha * x_init
            x = (1 - self.alpha) * x_transformed + self.alpha * x_init
            
            # Residual connection: x = (1 - beta) * x + beta * x_prev
            if i > 0:
                x = (1 - self.beta) * x + self.beta * x_prev
            
            # Dropout
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)
            
            x_prev = x
        
        # Разделить обратно на users и items
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
            raise ValueError("adj_matrix должен быть передан для GCNII")
        
        user_emb, item_emb = self.get_all_embeddings(adj_matrix)
        user_emb_selected = user_emb[users]
        item_emb_selected = item_emb[items]
        scores = (user_emb_selected * item_emb_selected).sum(dim=1)
        return scores
    
    def get_all_embeddings(
        self,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Получить все embeddings."""
        if adj_matrix is None:
            raise ValueError("adj_matrix должен быть передан для GCNII")
        return self.forward(adj_matrix)

