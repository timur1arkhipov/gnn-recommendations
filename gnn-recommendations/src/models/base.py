"""
Базовый класс для всех моделей рекомендательных систем.

Все модели наследуются от BaseRecommender и должны реализовать:
- forward() - forward pass через модель
- predict() - предсказание scores для user-item пар
- get_all_embeddings() - получение всех embeddings для evaluation
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class BaseRecommender(nn.Module):
    """
    Базовый класс для всех моделей рекомендательных систем.
    
    Все модели должны наследоваться от этого класса и реализовать
    методы forward(), predict() и get_all_embeddings().
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int
    ):
        """
        Инициализация базовой модели.
        
        Args:
            n_users: количество пользователей
            n_items: количество айтемов
            embedding_dim: размерность embeddings
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
    
    def forward(self, *args, **kwargs):
        """
        Forward pass через модель.
        
        Должен быть реализован в каждой конкретной модели.
        Разные модели могут принимать разные аргументы:
        - GNN модели: adj_matrix
        - MF модели: user_ids, item_ids
        
        Raises:
            NotImplementedError: если метод не реализован в подклассе
        """
        raise NotImplementedError(
            f"Метод forward() должен быть реализован в классе {self.__class__.__name__}"
        )
    
    def predict(
        self,
        users: torch.Tensor,
        items: torch.Tensor
    ) -> torch.Tensor:
        """
        Предсказание scores для user-item пар.
        
        Args:
            users: тензор с ID пользователей [batch_size]
            items: тензор с ID айтемов [batch_size]
        
        Returns:
            Тензор с предсказанными scores [batch_size]
        
        Raises:
            NotImplementedError: если метод не реализован в подклассе
        """
        raise NotImplementedError(
            f"Метод predict() должен быть реализован в классе {self.__class__.__name__}"
        )
    
    def get_all_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получить все embeddings пользователей и айтемов.
        
        Используется для evaluation, когда нужно вычислить scores
        для всех возможных user-item пар.
        
        Returns:
            Tuple из (user_embeddings, item_embeddings)
            - user_embeddings: [n_users, embedding_dim]
            - item_embeddings: [n_items, embedding_dim]
        
        Raises:
            NotImplementedError: если метод не реализован в подклассе
        """
        raise NotImplementedError(
            f"Метод get_all_embeddings() должен быть реализован в классе {self.__class__.__name__}"
        )
    
    def get_parameters_count(self) -> int:
        """
        Получить количество обучаемых параметров модели.
        
        Returns:
            Количество параметров
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reset_parameters(self):
        """
        Сброс параметров модели (для инициализации).
        
        Должен быть реализован в подклассе, если нужна кастомная инициализация.
        По умолчанию использует стандартную инициализацию PyTorch.
        """
        # Стандартная инициализация PyTorch
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

