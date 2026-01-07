"""
BPR-MF (Bayesian Personalized Ranking - Matrix Factorization)

Базовый метод матричной факторизации с BPR loss.
Самый простой baseline для рекомендательных систем.

Ссылка: Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback" (2009)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..base import BaseRecommender


class BPR_MF(BaseRecommender):
    """
    BPR-MF модель - матричная факторизация без графовой структуры.
    
    Простое скалярное произведение embeddings пользователей и айтемов.
    Не использует графовую структуру, только embeddings.
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        init_scale: float = 0.01
    ):
        """
        Инициализация BPR-MF модели.
        
        Args:
            n_users: количество пользователей
            n_items: количество айтемов
            embedding_dim: размерность embeddings
            init_scale: масштаб инициализации параметров
        """
        super().__init__(n_users, n_items, embedding_dim)
        
        self.init_scale = init_scale
        
        # Embeddings пользователей и айтемов
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
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (для совместимости с интерфейсом).
        
        BPR-MF не использует граф, просто возвращает embeddings.
        
        Args:
            adj_matrix: игнорируется (для совместимости)
        
        Returns:
            Tuple из (user_embeddings, item_embeddings)
        """
        user_emb = self.user_embedding.weight  # [n_users, embedding_dim]
        item_emb = self.item_embedding.weight   # [n_items, embedding_dim]
        
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
            adj_matrix: игнорируется
        
        Returns:
            Тензор с предсказанными scores [batch_size]
        """
        user_emb = self.user_embedding(users)  # [batch_size, embedding_dim]
        item_emb = self.item_embedding(items)  # [batch_size, embedding_dim]
        
        # Scores как скалярное произведение
        scores = (user_emb * item_emb).sum(dim=1)  # [batch_size]
        
        return scores
    
    def get_all_embeddings(
        self,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получить все embeddings пользователей и айтемов.
        
        Args:
            adj_matrix: игнорируется
        
        Returns:
            Tuple из (user_embeddings, item_embeddings)
        """
        return self.forward(adj_matrix)

