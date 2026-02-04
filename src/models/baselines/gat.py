"""
GAT - Graph Attention Network для Collaborative Filtering.

GAT использует attention механизм для взвешивания важности соседей.
Модель автоматически обучается, какие соседи более релевантны для каждого узла.

Ссылка: Veličković et al. "Graph Attention Networks" (2018)
Адаптация для CF.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..base import BaseRecommender


class GATLayer(nn.Module):
    """
    Один слой GAT с multi-head attention.

    Attention weights: α_{ij} = softmax(LeakyReLU(a^T [W*h_i || W*h_j]))
    Output: h_i' = σ(Σ_j α_{ij} * W * h_j)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 1,
        dropout: float = 0.0,
        alpha: float = 0.2,
        concat_heads: bool = True
    ):
        """
        Инициализация GAT слоя.

        Args:
            in_dim: входная размерность
            out_dim: выходная размерность на один head
            n_heads: количество attention heads
            dropout: вероятность dropout
            alpha: negative slope для LeakyReLU
            concat_heads: True - конкатенация heads, False - усреднение
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.concat_heads = concat_heads
        self.dropout = dropout
        self.alpha = alpha

        # Линейное преобразование для каждого head
        self.W = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(n_heads)
        ])

        # Attention параметры для каждого head
        # a^T [W*h_i || W*h_j] = a_self^T * W*h_i + a_neigh^T * W*h_j
        self.a_self = nn.ParameterList([
            nn.Parameter(torch.zeros(size=(out_dim, 1)))
            for _ in range(n_heads)
        ])
        self.a_neigh = nn.ParameterList([
            nn.Parameter(torch.zeros(size=(out_dim, 1)))
            for _ in range(n_heads)
        ])

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass через GAT слой.

        Args:
            x: node embeddings [N, in_dim]
            adj_matrix: adjacency matrix [N, N] (может быть sparse)

        Returns:
            updated embeddings [N, out_dim * n_heads] если concat
                               [N, out_dim] если average
        """
        N = x.size(0)

        # Compute attention для каждого head
        head_outputs = []

        for i in range(self.n_heads):
            # Линейное преобразование
            h = self.W[i](x)  # [N, out_dim]

            # Compute attention coefficients
            # e_ij = LeakyReLU(a^T [W*h_i || W*h_j])
            #      = LeakyReLU(a_self^T * W*h_i + a_neigh^T * W*h_j)

            # Self attention: a_self^T * W*h_i для каждого узла
            attn_self = torch.mm(h, self.a_self[i])  # [N, 1]

            # Neighbor attention: a_neigh^T * W*h_j для каждого узла
            attn_neigh = torch.mm(h, self.a_neigh[i])  # [N, 1]

            # e_ij = attn_self[i] + attn_neigh[j]
            # Создаем матрицу attention scores
            attn_scores = attn_self + attn_neigh.t()  # [N, N]

            # Применяем LeakyReLU
            attn_scores = self.leakyrelu(attn_scores)

            # Маскируем несуществующие ребра
            # Создаем маску из adj_matrix
            if adj_matrix.is_sparse:
                # Для sparse matrix
                indices = adj_matrix.coalesce().indices()
                mask = torch.sparse.FloatTensor(
                    indices,
                    torch.ones(indices.size(1), device=x.device),
                    adj_matrix.size()
                ).to_dense()
            else:
                mask = adj_matrix

            # Маскируем: -inf где нет ребра, чтобы после softmax получился 0
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

            # Softmax по соседям (по строкам)
            attn_weights = F.softmax(attn_scores, dim=1)  # [N, N]

            # Dropout на attention weights
            attn_weights = self.dropout_layer(attn_weights)

            # Aggregate neighbors с весами attention
            h_prime = torch.mm(attn_weights, h)  # [N, out_dim]

            head_outputs.append(h_prime)

        # Combine heads
        if self.concat_heads:
            output = torch.cat(head_outputs, dim=1)  # [N, out_dim * n_heads]
        else:
            output = torch.stack(head_outputs, dim=0).mean(dim=0)  # [N, out_dim]

        return output


class GAT(BaseRecommender):
    """
    GAT модель для рекомендательных систем.

    Архитектура:
    1. Начальные embeddings
    2. Несколько GAT слоев с multi-head attention
    3. Layer aggregation (среднее всех слоев)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
        init_scale: float = 0.01
    ):
        """
        Инициализация GAT модели.

        Args:
            n_users: количество пользователей
            n_items: количество айтемов
            embedding_dim: размерность embeddings
            n_layers: количество GAT слоев
            n_heads: количество attention heads в каждом слое
            dropout: вероятность dropout
            alpha: negative slope для LeakyReLU
            init_scale: масштаб инициализации параметров
        """
        super().__init__(n_users, n_items, embedding_dim)

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha
        self.init_scale = init_scale

        # Начальные embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # GAT слои
        self.layers = nn.ModuleList()

        # Первый слой: embedding_dim -> embedding_dim (с multi-head concat)
        self.layers.append(
            GATLayer(
                in_dim=embedding_dim,
                out_dim=embedding_dim // n_heads,  # Чтобы после concat получился embedding_dim
                n_heads=n_heads,
                dropout=dropout,
                alpha=alpha,
                concat_heads=True
            )
        )

        # Средние слои
        for _ in range(n_layers - 2):
            self.layers.append(
                GATLayer(
                    in_dim=embedding_dim,
                    out_dim=embedding_dim // n_heads,
                    n_heads=n_heads,
                    dropout=dropout,
                    alpha=alpha,
                    concat_heads=True
                )
            )

        # Последний слой: averaging вместо concat
        if n_layers > 1:
            self.layers.append(
                GATLayer(
                    in_dim=embedding_dim,
                    out_dim=embedding_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    alpha=alpha,
                    concat_heads=False  # Average для последнего слоя
                )
            )

        # Инициализация
        self.reset_parameters()

    def reset_parameters(self):
        """Инициализация параметров."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=self.init_scale)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=self.init_scale)

        # Инициализация GAT layers
        for layer in self.layers:
            for w in layer.W:
                nn.init.xavier_uniform_(w.weight)
            for a in layer.a_self:
                nn.init.xavier_uniform_(a.data)
            for a in layer.a_neigh:
                nn.init.xavier_uniform_(a.data)

    def forward(
        self,
        adj_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass через GAT.

        Args:
            adj_matrix: adjacency matrix [N, N], где N = n_users + n_items

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

        # Прогоняем через GAT слои
        for layer in self.layers:
            x = layer(x, adj_matrix)
            x = F.elu(x)  # Нелинейность между слоями
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
            adj_matrix: adjacency matrix

        Returns:
            Тензор с предсказанными scores [batch_size]
        """
        if adj_matrix is None:
            raise ValueError("adj_matrix должен быть передан для GAT")

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
            adj_matrix: adjacency matrix

        Returns:
            Tuple из (user_embeddings, item_embeddings)
        """
        if adj_matrix is None:
            raise ValueError("adj_matrix должен быть передан для GAT")

        return self.forward(adj_matrix)
