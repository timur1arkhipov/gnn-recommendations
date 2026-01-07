"""
GroupShuffleGNN - основная модель с применением метода Group and Shuffle.

Использует GroupShuffleLayer для борьбы с over-smoothing в GNN рекомендательных системах.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..base import BaseRecommender
from .layers import GroupShuffleLayer


class GroupShuffleGNN(BaseRecommender):
    """
    GNN модель с применением метода Group and Shuffle.
    
    Архитектура:
    1. Начальные embeddings (user и item)
    2. Несколько слоев GroupShuffleLayer
    3. Residual connections для сохранения информации
    4. Layer aggregation для объединения представлений из всех слоев
    
    Метод из статьи Gorbunov and Yudin "Group and Shuffle".
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        block_size: int = 8,
        residual_alpha: float = 0.1,
        dropout: float = 0.0,
        init_scale: float = 0.01
    ):
        """
        Инициализация GroupShuffleGNN.
        
        Args:
            n_users: количество пользователей
            n_items: количество айтемов
            embedding_dim: размерность embeddings (должна делиться на block_size)
            n_layers: количество слоев GroupShuffleLayer
            block_size: размер блока для ортогональной матрицы
            residual_alpha: коэффициент для residual connections (0.0 = нет residual, 1.0 = только начальные embeddings)
            dropout: вероятность dropout (0.0 = нет dropout)
            init_scale: масштаб инициализации параметров
        """
        super().__init__(n_users, n_items, embedding_dim)
        
        # Проверяем, что embedding_dim делится на block_size
        if embedding_dim % block_size != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) должно делиться на block_size ({block_size})"
            )
        
        self.n_layers = n_layers
        self.block_size = block_size
        self.residual_alpha = residual_alpha
        self.dropout = dropout
        
        # Начальные embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # ⭐ ВАШИ СЛОИ - GroupShuffleLayer
        self.layers = nn.ModuleList([
            GroupShuffleLayer(embedding_dim, block_size, init_scale)
            for _ in range(n_layers)
        ])
        
        # Dropout (опционально)
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
        
        # Инициализация параметров
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Инициализация параметров модели.
        """
        # Инициализация embeddings
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        
        # Инициализация слоев (уже делается в GroupShuffleLayer)
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def forward(
        self,
        adj_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass через модель.
        
        Args:
            adj_matrix: normalized adjacency matrix [N, N], где N = n_users + n_items
                       Может быть sparse или dense tensor
        
        Returns:
            Tuple из (user_embeddings, item_embeddings)
            - user_embeddings: [n_users, embedding_dim]
            - item_embeddings: [n_items, embedding_dim]
        """
        # Начальные embeddings
        # Объединяем user и item embeddings в один тензор
        x_init = torch.cat([
            self.user_embedding.weight,  # [n_users, embedding_dim]
            self.item_embedding.weight   # [n_items, embedding_dim]
        ], dim=0)  # [N, embedding_dim], где N = n_users + n_items
        
        # Сохраняем начальные embeddings для residual connections
        x = x_init
        all_embeddings = [x]  # Сохраняем embeddings из всех слоев для layer aggregation
        
        # Прохождение через слои
        for layer in self.layers:
            # Forward через GroupShuffleLayer
            x_transformed = layer(x, adj_matrix)  # [N, embedding_dim]
            
            # Residual connection
            # x = (1 - alpha) * x_transformed + alpha * x_init
            # alpha = 0.0 → только трансформированные embeddings
            # alpha = 1.0 → только начальные embeddings
            # alpha = 0.1 → 90% трансформированных + 10% начальных
            x = (1 - self.residual_alpha) * x_transformed + \
                self.residual_alpha * x_init
            
            # Dropout (если включен)
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)
            
            # Сохраняем embeddings для layer aggregation
            all_embeddings.append(x)
        
        # Layer aggregation - объединение представлений из всех слоев
        # Используем среднее арифметическое (можно также использовать взвешенную сумму)
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
            adj_matrix: normalized adjacency matrix (если None, используется последний)
        
        Returns:
            Тензор с предсказанными scores [batch_size]
        """
        # Получаем все embeddings
        user_emb, item_emb = self.get_all_embeddings(adj_matrix)
        
        # Выбираем embeddings для указанных пользователей и айтемов
        user_emb_selected = user_emb[users]  # [batch_size, embedding_dim]
        item_emb_selected = item_emb[items]   # [batch_size, embedding_dim]
        
        # Вычисляем scores как скалярное произведение
        scores = (user_emb_selected * item_emb_selected).sum(dim=1)  # [batch_size]
        
        return scores
    
    def get_all_embeddings(
        self,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получить все embeddings пользователей и айтемов.
        
        Args:
            adj_matrix: normalized adjacency matrix (если None, нужно передать в forward)
        
        Returns:
            Tuple из (user_embeddings, item_embeddings)
            - user_embeddings: [n_users, embedding_dim]
            - item_embeddings: [n_items, embedding_dim]
        """
        if adj_matrix is None:
            raise ValueError(
                "adj_matrix должен быть передан для получения embeddings. "
                "Используйте forward(adj_matrix) или передайте adj_matrix в get_all_embeddings(adj_matrix)."
            )
        
        # Forward pass через модель
        user_emb, item_emb = self.forward(adj_matrix)
        
        return user_emb, item_emb
    
    def get_orthogonality_errors(self) -> torch.Tensor:
        """
        Получить ошибки ортогональности для всех слоев.
        
        Полезно для мониторинга во время обучения.
        
        Returns:
            Тензор с ошибками ортогональности [n_layers]
        """
        errors = []
        for layer in self.layers:
            error = layer.get_orthogonality_error()
            errors.append(error)
        return torch.stack(errors)

