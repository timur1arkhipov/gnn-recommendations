"""
LightGCN - упрощенная версия GCN для рекомендательных систем.

Убраны нелинейности и веса, осталась только графовая свертка и layer aggregation.
Один из самых эффективных baseline методов.

Ссылка: He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" (2020)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from ..base import BaseRecommender


class LightGCN(BaseRecommender):
    """
    LightGCN модель - упрощенный GCN без нелинейностей и весов.
    
    Архитектура:
    1. Начальные embeddings
    2. Графовая свертка через несколько слоев (без весов и нелинейностей)
    3. Layer aggregation (среднее арифметическое всех слоев)
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        init_scale: float = 0.01
    ):
        """
        Инициализация LightGCN модели.
        
        Args:
            n_users: количество пользователей
            n_items: количество айтемов
            embedding_dim: размерность embeddings
            n_layers: количество слоев графовой свертки
            init_scale: масштаб инициализации параметров
        """
        super().__init__(n_users, n_items, embedding_dim)
        
        self.n_layers = n_layers
        self.init_scale = init_scale
        
        # Начальные embeddings (единственные параметры модели)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Инициализация
        self.reset_parameters()
    
    def reset_parameters(self):
        """Инициализация параметров."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=self.init_scale)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=self.init_scale)
    
    def forward(
        self,
        adj_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass через LightGCN.
        
        Args:
            adj_matrix: normalized adjacency matrix [N, N], где N = n_users + n_items
        
        Returns:
            Tuple из (user_embeddings, item_embeddings)
        """
        # Начальные embeddings
        x = torch.cat([
            self.user_embedding.weight,  # [n_users, embedding_dim]
            self.item_embedding.weight   # [n_items, embedding_dim]
        ], dim=0)  # [N, embedding_dim]
        
        # Сохраняем embeddings из всех слоев для layer aggregation
        all_embeddings = [x]
        
        # Графовая свертка через n_layers слоев
        # В LightGCN нет весов и нелинейностей, только графовая свертка
        for _ in range(self.n_layers):
            if adj_matrix.is_sparse:
                x = torch.sparse.mm(adj_matrix, x)
            else:
                x = torch.mm(adj_matrix, x)
            all_embeddings.append(x)
        
        # Layer aggregation - среднее арифметическое всех слоев
        all_embeddings_stack = torch.stack(all_embeddings, dim=0)  # [n_layers+1, N, embedding_dim]
        x_final = torch.mean(all_embeddings_stack, dim=0)  # [N, embedding_dim]
        
        # Разделить обратно на users и items
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
        """
        Предсказание scores для user-item пар.
        
        Args:
            users: тензор с ID пользователей [batch_size]
            items: тензор с ID айтемов [batch_size]
            adj_matrix: normalized adjacency matrix
        
        Returns:
            Тензор с предсказанными scores [batch_size]
        """
        if adj_matrix is None:
            raise ValueError("adj_matrix должен быть передан для LightGCN")
        
        user_emb, item_emb = self.get_all_embeddings(adj_matrix)
        
        user_emb_selected = user_emb[users]
        item_emb_selected = item_emb[items]
        
        scores = (user_emb_selected * item_emb_selected).sum(dim=1)
        
        return scores
    
    def get_all_embeddings(
        self,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получить все embeddings пользователей и айтемов.
        
        Args:
            adj_matrix: normalized adjacency matrix
        
        Returns:
            Tuple из (user_embeddings, item_embeddings)
        """
        if adj_matrix is None:
            raise ValueError("adj_matrix должен быть передан для LightGCN")
        
        return self.forward(adj_matrix)
    
    def get_layer_embeddings(
        self,
        adj_matrix: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Получить embeddings для каждого слоя (для анализа over-smoothing).
        
        Args:
            adj_matrix: normalized adjacency matrix [N, N]
        
        Returns:
            Список тензоров embeddings для каждого слоя [n_layers+1]
            Каждый тензор имеет размер [N, embedding_dim]
        """
        # Начальные embeddings
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        all_embeddings = [x.clone()]
        
        # Графовая свертка через n_layers слоев
        for _ in range(self.n_layers):
            if adj_matrix.is_sparse:
                x = torch.sparse.mm(adj_matrix, x)
            else:
                x = torch.mm(adj_matrix, x)
            all_embeddings.append(x.clone())
        
        return all_embeddings

