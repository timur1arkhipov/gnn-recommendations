"""
UltraGCN - Ultra Simplification of Graph Convolutional Networks.

UltraGCN использует математический прокси для бесконечного количества GCN слоев
без явных graph convolutions. Модель напрямую вычисляет финальные embeddings
через constraint loss, который аппроксимирует результат infinite-layer GCN.

Ключевая идея: борьба с over-smoothing через constraint loss вместо явных convolutions.

Ссылка: Mao et al. "UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..base import BaseRecommender


class UltraGCN(BaseRecommender):
    """
    UltraGCN модель - прокси для infinite-layer GCN без явных convolutions.

    Архитектура:
    1. Прямые embeddings (без graph convolution layers)
    2. Constraint loss для аппроксимации GCN: Ω = ||E_u - Σ E_i||²
    3. Negative sampling с весами для stable training
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        lambda_1: float = 1.0,  # Weight for constraint loss
        lambda_2: float = 1.0,  # Weight for L2 regularization
        gamma: float = 1e-4,    # L2 regularization coefficient
        neg_weight: float = 0.5,  # Weight for negative samples
        init_scale: float = 0.01
    ):
        """
        Инициализация UltraGCN модели.

        Args:
            n_users: количество пользователей
            n_items: количество айтемов
            embedding_dim: размерность embeddings
            lambda_1: вес constraint loss
            lambda_2: вес L2 регуляризации
            gamma: коэффициент L2 регуляризации
            neg_weight: вес для negative samples
            init_scale: масштаб инициализации параметров
        """
        super().__init__(n_users, n_items, embedding_dim)

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.gamma = gamma
        self.neg_weight = neg_weight
        self.init_scale = init_scale

        # Только embeddings - никаких весов для graph convolution
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
        Forward pass через UltraGCN.

        UltraGCN не использует явные graph convolutions, поэтому forward
        просто возвращает текущие embeddings. Вся магия происходит в loss функции.

        Args:
            adj_matrix: не используется в UltraGCN (для совместимости с API)

        Returns:
            Tuple из (user_embeddings, item_embeddings)
        """
        return self.user_embedding.weight, self.item_embedding.weight

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
            adj_matrix: не используется (для совместимости)

        Returns:
            Тензор с предсказанными scores [batch_size]
        """
        user_emb = self.user_embedding(users)  # [batch_size, embedding_dim]
        item_emb = self.item_embedding(items)  # [batch_size, embedding_dim]

        scores = (user_emb * item_emb).sum(dim=1)  # [batch_size]

        return scores

    def get_all_embeddings(
        self,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получить все embeddings пользователей и айтемов.

        Args:
            adj_matrix: не используется (для совместимости)

        Returns:
            Tuple из (user_embeddings, item_embeddings)
        """
        return self.forward(adj_matrix)

    def compute_constraint_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление constraint loss для аппроксимации infinite-layer GCN.

        Constraint loss: Ω = Σ ||E_u - (1/|N(u)|) * Σ_{i ∈ N(u)} E_i||²

        Это заставляет user embeddings быть близкими к среднему их neighbor item embeddings,
        что аппроксимирует результат infinite graph convolutions.

        Args:
            users: тензор с ID пользователей [batch_size]
            pos_items: тензор с ID positive items [batch_size]
            adj_matrix: normalized adjacency matrix [n_users, n_items]

        Returns:
            Constraint loss
        """
        # User embeddings
        user_emb = self.user_embedding(users)  # [batch_size, embedding_dim]

        # Для каждого пользователя: среднее embeddings его positive items
        # Используем adj_matrix для получения neighbors
        batch_size = users.size(0)
        constraint_loss = 0.0

        for i in range(batch_size):
            user_idx = users[i]

            # Получить все items, с которыми взаимодействовал user
            # adj_matrix[user_idx] содержит веса связей с items
            if adj_matrix.is_sparse:
                user_neighbors = adj_matrix[user_idx].to_dense()
            else:
                user_neighbors = adj_matrix[user_idx]

            # Найти non-zero items (positive interactions)
            neighbor_items = torch.nonzero(user_neighbors, as_tuple=True)[0]

            if neighbor_items.size(0) > 0:
                # Embeddings всех neighbor items
                neighbor_emb = self.item_embedding(neighbor_items)  # [n_neighbors, embedding_dim]

                # Среднее neighbor embeddings
                mean_neighbor_emb = neighbor_emb.mean(dim=0)  # [embedding_dim]

                # Constraint: ||E_u - mean(E_neighbors)||²
                constraint_loss += torch.sum((user_emb[i] - mean_neighbor_emb) ** 2)

        # Усредняем по batch
        constraint_loss = constraint_loss / batch_size

        return self.lambda_1 * constraint_loss

    def compute_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Вычисление полного loss для UltraGCN.

        Total loss = BPR loss + constraint loss + L2 regularization

        Args:
            users: тензор с ID пользователей [batch_size]
            pos_items: тензор с ID positive items [batch_size]
            neg_items: тензор с ID negative items [batch_size]
            adj_matrix: normalized adjacency matrix [n_users, n_items]

        Returns:
            Tuple из (total_loss, loss_dict) где loss_dict содержит компоненты loss
        """
        # User embeddings
        user_emb = self.user_embedding(users)  # [batch_size, embedding_dim]

        # Positive и negative item embeddings
        pos_item_emb = self.item_embedding(pos_items)  # [batch_size, embedding_dim]
        neg_item_emb = self.item_embedding(neg_items)  # [batch_size, embedding_dim]

        # BPR loss
        pos_scores = (user_emb * pos_item_emb).sum(dim=1)  # [batch_size]
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)  # [batch_size]

        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # Constraint loss (если есть adj_matrix)
        if adj_matrix is not None:
            constraint_loss = self.compute_constraint_loss(users, pos_items, adj_matrix)
        else:
            constraint_loss = torch.tensor(0.0, device=user_emb.device)

        # L2 regularization
        l2_loss = self.lambda_2 * self.gamma * (
            torch.norm(user_emb) ** 2 +
            torch.norm(pos_item_emb) ** 2 +
            torch.norm(neg_item_emb) ** 2
        ) / user_emb.size(0)

        # Total loss
        total_loss = bpr_loss + constraint_loss + l2_loss

        loss_dict = {
            'bpr_loss': bpr_loss.item(),
            'constraint_loss': constraint_loss.item() if isinstance(constraint_loss, torch.Tensor) else constraint_loss,
            'l2_loss': l2_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict
