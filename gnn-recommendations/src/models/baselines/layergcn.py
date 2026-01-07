"""
LayerGCN - Layer-wise Graph Convolutional Network.

Использует layer-wise refinement для постепенного уточнения представлений.

Ссылка: Chen et al. "LayerGCN: Layer-wise Refinement for Graph Convolutional Networks" (2020)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..base import BaseRecommender


class LayerGCN(BaseRecommender):
    """
    LayerGCN модель - layer-wise refinement.
    
    Каждый слой постепенно уточняет представления, используя информацию
    из предыдущих слоев.
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        alpha: float = 0.5,
        dropout: float = 0.0,
        init_scale: float = 0.01
    ):
        """
        Инициализация LayerGCN модели.
        
        Args:
            n_users: количество пользователей
            n_items: количество айтемов
            embedding_dim: размерность embeddings
            n_layers: количество слоев
            alpha: коэффициент для layer-wise refinement (0.0-1.0)
            dropout: вероятность dropout
            init_scale: масштаб инициализации параметров
        """
        super().__init__(n_users, n_items, embedding_dim)
        
        self.n_layers = n_layers
        self.alpha = alpha
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
        
        # Веса для layer-wise refinement
        self.layer_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * alpha)
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
        for layer_weight in self.layer_weights:
            nn.init.constant_(layer_weight, self.alpha)
    
    def forward(
        self,
        adj_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass через LayerGCN.
        
        Args:
            adj_matrix: normalized adjacency matrix [N, N]
        
        Returns:
            Tuple из (user_embeddings, item_embeddings)
        """
        # Начальные embeddings
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        # Сохраняем представления из всех слоев
        layer_embeddings = [x]
        
        # Прохождение через слои
        for i, (weight, layer_weight) in enumerate(zip(self.weights, self.layer_weights)):
            # Графовая свертка
            if adj_matrix.is_sparse:
                x_conv = torch.sparse.mm(adj_matrix, x)
            else:
                x_conv = torch.mm(adj_matrix, x)
            
            # Линейное преобразование
            x_transformed = x_conv @ weight
            
            # Layer-wise refinement: комбинация текущего и предыдущих слоев
            # x = layer_weight * x_transformed + (1 - layer_weight) * avg(previous_layers)
            if i > 0:
                prev_avg = torch.mean(torch.stack(layer_embeddings), dim=0)
                x = layer_weight * x_transformed + (1 - layer_weight) * prev_avg
            else:
                x = x_transformed
            
            # Dropout
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)
            
            layer_embeddings.append(x)
        
        # Финальное представление - среднее всех слоев
        x_final = torch.mean(torch.stack(layer_embeddings), dim=0)
        
        # Разделить обратно
        user_emb, item_emb = torch.split(
            x_final,
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
            raise ValueError("adj_matrix должен быть передан для LayerGCN")
        user_emb, item_emb = self.get_all_embeddings(adj_matrix)
        scores = (user_emb[users] * item_emb[items]).sum(dim=1)
        return scores
    
    def get_all_embeddings(
        self,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Получить все embeddings."""
        if adj_matrix is None:
            raise ValueError("adj_matrix должен быть передан для LayerGCN")
        return self.forward(adj_matrix)

