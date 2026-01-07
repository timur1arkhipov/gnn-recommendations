"""
DGR (Desmoothing Graph Representation) - современный метод борьбы с over-smoothing.

Использует desmoothing framework для предотвращения over-smoothing в глубоких GNN.

Ссылка: Bei et al. "DGR: Tackling Over-Smoothing in Graph Neural Networks via Deep Graph Regularization" (2024)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..base import BaseRecommender


class DGR(BaseRecommender):
    """
    DGR модель - Desmoothing Graph Representation.
    
    Использует regularization для предотвращения over-smoothing.
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        lambda_reg: float = 0.1,
        dropout: float = 0.0,
        init_scale: float = 0.01
    ):
        """
        Инициализация DGR модели.
        
        Args:
            n_users: количество пользователей
            n_items: количество айтемов
            embedding_dim: размерность embeddings
            n_layers: количество слоев
            lambda_reg: коэффициент регуляризации для desmoothing
            dropout: вероятность dropout
            init_scale: масштаб инициализации параметров
        """
        super().__init__(n_users, n_items, embedding_dim)
        
        self.n_layers = n_layers
        self.lambda_reg = lambda_reg
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
    
    def forward(
        self,
        adj_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass через DGR.
        
        Args:
            adj_matrix: normalized adjacency matrix [N, N]
        
        Returns:
            Tuple из (user_embeddings, item_embeddings)
        """
        # Начальные embeddings
        x_init = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        x = x_init
        x_prev = x_init
        
        # Прохождение через слои
        for weight in self.weights:
            # Графовая свертка
            if adj_matrix.is_sparse:
                x_conv = torch.sparse.mm(adj_matrix, x)
            else:
                x_conv = torch.mm(adj_matrix, x)
            
            # Линейное преобразование
            x_transformed = x_conv @ weight
            
            # Desmoothing: комбинация текущего и предыдущего слоя
            x = (1 - self.lambda_reg) * x_transformed + self.lambda_reg * x_prev
            
            # Dropout
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)
            
            x_prev = x
        
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
            raise ValueError("adj_matrix должен быть передан для DGR")
        user_emb, item_emb = self.get_all_embeddings(adj_matrix)
        scores = (user_emb[users] * item_emb[items]).sum(dim=1)
        return scores
    
    def get_all_embeddings(
        self,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Получить все embeddings."""
        if adj_matrix is None:
            raise ValueError("adj_matrix должен быть передан для DGR")
        return self.forward(adj_matrix)

