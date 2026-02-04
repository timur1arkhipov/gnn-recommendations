"""
NGCF - Neural Graph Collaborative Filtering.

NGCF использует граф-свертки с интеграцией collaborative signals.
Ключевая особенность: embedding propagation с учетом взаимодействий
между users и items через message passing с нелинейностями.

Ссылка: Wang et al. "Neural Graph Collaborative Filtering" (2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from ..base import BaseRecommender


class NGCFLayer(nn.Module):
    """
    Один слой NGCF с message passing и feature transformation.

    Message passing: m_{u←i} = W₁x_i + W₂(x_i ⊙ x_u)
    Aggregation: x_u' = LeakyReLU(W₁ * Σ(m_{u←i}) + W₂ * x_u)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0
    ):
        """
        Инициализация NGCF слоя.

        Args:
            in_dim: входная размерность
            out_dim: выходная размерность
            dropout: вероятность dropout
        """
        super().__init__()

        # W1 для линейного преобразования соседей
        self.W1 = nn.Linear(in_dim, out_dim, bias=True)

        # W2 для element-wise product между embeddings
        self.W2 = nn.Linear(in_dim, out_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass через NGCF слой.

        Args:
            x: node embeddings [N, in_dim]
            adj_matrix: normalized adjacency matrix [N, N]

        Returns:
            updated embeddings [N, out_dim]
        """
        # Линейное преобразование соседей
        # neighbor_emb = Σ(m_{u←i}) через adj_matrix
        if adj_matrix.is_sparse:
            neighbor_emb = torch.sparse.mm(adj_matrix, x)
        else:
            neighbor_emb = torch.mm(adj_matrix, x)

        # Element-wise interaction между self и neighbors
        # Для каждого узла: x_i ⊙ Σ(x_neighbors)
        self_emb = x
        interaction = self_emb * neighbor_emb

        # Message passing: W1 * neighbors + W2 * (self ⊙ neighbors)
        out = self.W1(neighbor_emb) + self.W2(interaction)

        # Activation и dropout
        out = self.activation(out)
        out = self.dropout(out)

        return out


class NGCF(BaseRecommender):
    """
    NGCF модель - GCN с collaborative signal integration.

    Архитектура:
    1. Начальные embeddings
    2. Несколько NGCF слоев с message passing и interactions
    3. Layer aggregation (конкатенация всех слоев)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        layer_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
        init_scale: float = 0.01
    ):
        """
        Инициализация NGCF модели.

        Args:
            n_users: количество пользователей
            n_items: количество айтемов
            embedding_dim: размерность начальных embeddings
            layer_sizes: список размерностей для каждого слоя [64, 64, 64]
            dropout: вероятность dropout
            init_scale: масштаб инициализации параметров
        """
        super().__init__(n_users, n_items, embedding_dim)

        if layer_sizes is None:
            layer_sizes = [64, 64, 64]  # 3 слоя по умолчанию

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.dropout = dropout
        self.init_scale = init_scale

        # Начальные embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # NGCF слои
        self.layers = nn.ModuleList()
        in_dim = embedding_dim
        for out_dim in layer_sizes:
            self.layers.append(NGCFLayer(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Инициализация
        self.reset_parameters()

    def reset_parameters(self):
        """Инициализация параметров."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=self.init_scale)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=self.init_scale)

        # Инициализация весов слоев
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.W1.weight)
            nn.init.xavier_uniform_(layer.W2.weight)
            if layer.W1.bias is not None:
                nn.init.zeros_(layer.W1.bias)
            if layer.W2.bias is not None:
                nn.init.zeros_(layer.W2.bias)

    def forward(
        self,
        adj_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass через NGCF.

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

        # Прогоняем через NGCF слои
        for layer in self.layers:
            x = layer(x, adj_matrix)
            all_embeddings.append(x)

        # Layer aggregation - конкатенация всех слоев
        # Это позволяет модели использовать информацию с разных уровней абстракции
        x_final = torch.cat(all_embeddings, dim=1)  # [N, embedding_dim + sum(layer_sizes)]

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
            raise ValueError("adj_matrix должен быть передан для NGCF")

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
            raise ValueError("adj_matrix должен быть передан для NGCF")

        return self.forward(adj_matrix)
