"""
OrthogonalBundleGNN - Ортогональная векторная расслоенная графовая нейронная сеть.

Реализует предложенный метод, объединяющий:
1. Структура расслоения: Каждая вершина имеет своё пространство волокон (локальное пространство признаков)
2. Ортогональные матрицы связи W_{ij}: Переносят эмбеддинги между пространствами волокон
3. Параллельный перенос: Перемещает представления вдоль рёбер, сохраняя геометрическую структуру
4. Локальные преобразования: Механизм Group & Shuffle для выразительности
5. Предотвращение over-smoothing: Ортогональность обеспечивает ||W_{ij} · x|| = ||x||

Архитектура на каждом слое:
    1. Параллельный перенос: Передача эмбеддингов вдоль рёбер через W_{ij}
    2. Локальное преобразование: Ортогональное преобразование внутри пространства волокон
    3. Residual-связь: Стабилизация обучения
    4. Агрегация слоёв: Взвешенная комбинация всех слоёв
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict

from ..base import BaseRecommender
from .bundle_layer import BundleConnectionLayer
from .parallel_transport import parallel_transport_along_edges
from .group_shuffle_layer import GroupShuffleLayer


class OrthogonalBundleGNN(BaseRecommender):
    """
    Ортогональная векторная расслоенная GNN для рекомендательных систем

    Предложенный метод:
    - Структура расслоения: Каждая вершина (пользователь/товар) имеет своё пространство волокон
    - Матрицы связи W_{ij}: Ортогональные преобразования для параллельного переноса
    - Group & Shuffle: Ортогональная параметризация, обеспечивающая ||W_{ij} · x|| = ||x||
    - Предотвращение over-smoothing: Ортогональность предотвращает коллапс представлений

    Архитектура (L слоёв):
        Для каждого слоя l:
            1. Параллельный перенос: x^(l) переносится вдоль рёбер через W_{ij}
            2. Локальное преобразование: Ортогональный Group & Shuffle внутри пространства волокон
            3. Residual-связь: x^(l+1) = (1-α)·transformed + α·x^(0)

        Финал: Взвешенная агрегация всех слоёв

    Ключевые свойства:
    - Сохранение нормы: ||W_{ij} · x|| = ||x||
    - Предотвращает over-smoothing даже с глубокими слоями
    - Сохраняет выразительность через локальные преобразования
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
        init_scale: float = 0.01,
        use_parallel_transport: bool = True,
        use_edge_index: bool = False
    ):
        """
        Инициализирует OrthogonalBundleGNN.

        Args:
            n_users: количество пользователей
            n_items: количество товаров
            embedding_dim: размерность эмбеддинга (должна делиться на block_size)
            n_layers: количество слоёв (L в статье)
            block_size: размер блока для ортогональных матриц
            residual_alpha: α в residual-связи (0.0 = нет residual, 1.0 = только начальные)
            dropout: вероятность dropout
            init_scale: масштаб инициализации для параметров
            use_parallel_transport: использовать ли параллельный перенос с W_{ij}
            use_edge_index: использовать ли формат edge_index (True) или adj_matrix (False)
        """
        super().__init__(n_users, n_items, embedding_dim)
        
        if embedding_dim % block_size != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by block_size ({block_size})"
            )
        
        self.n_layers = n_layers
        self.block_size = block_size
        self.residual_alpha = residual_alpha
        self.dropout = dropout
        self.use_parallel_transport = use_parallel_transport
        self.use_edge_index = use_edge_index

        # 1. БАЗОВЫЕ ЭМБЕДДИНГИ - Пространства волокон для каждой вершины
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        # 2. МАТРИЦЫ СВЯЗИ W_{ij} - Ортогональные преобразования для параллельного переноса
        if use_parallel_transport:
            self.connection_layers = nn.ModuleList([
                BundleConnectionLayer(embedding_dim, block_size)
                for _ in range(n_layers)
            ])

        # 3. СЛОИ ЛОКАЛЬНЫХ ПРЕОБРАЗОВАНИЙ - Group & Shuffle внутри пространств волокон
        self.local_transform_layers = nn.ModuleList([
            GroupShuffleLayer(embedding_dim, block_size, init_scale)
            for _ in range(n_layers)
        ])

        # 4. DROPOUT
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

        # 5. ВЕСА АГРЕГАЦИИ СЛОЁВ (обучаемые)
        self.layer_weights = nn.Parameter(torch.ones(n_layers + 1))
    
    def forward(
        self,
        adj_matrix: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход через Orthogonal Bundle GNN

        Реализует предложенную архитектуру:
        Для каждого слоя l:
            1. Параллельный перенос: x переносится вдоль рёбер через W_{ij}
            2. Локальное преобразование: Group & Shuffle внутри пространства волокон
            3. Residual-связь: комбинируется с начальными эмбеддингами

        Args:
            adj_matrix: нормализованная матрица смежности [N, N] (если не используется edge_index)
            edge_index: список рёбер [2, num_edges] (если не используется adj_matrix)

        Returns:
            Кортеж из (user_embeddings, item_embeddings)
        """
        # Проверяем формат входных данных
        if self.use_edge_index:
            if edge_index is None:
                raise ValueError("edge_index должен быть передан при use_edge_index=True")
        else:
            if adj_matrix is None:
                raise ValueError("adj_matrix должна быть передана при use_edge_index=False")

        # Начальные эмбеддинги (пространства волокон в каждой вершине)
        x_init = torch.cat([
            self.user_embedding.weight,  # [n_users, embedding_dim]
            self.item_embedding.weight   # [n_items, embedding_dim]
        ], dim=0)  # [N, embedding_dim], N = n_users + n_items

        x = x_init
        all_layer_embeddings = [x]

        # Проходим через L слоёв
        for layer_idx in range(self.n_layers):
            if self.use_parallel_transport:
                # Получаем матрицу связи W_{ij} для этого слоя
                W_connection = self.connection_layers[layer_idx]()

                if self.use_edge_index:
                    # ШАГ 1: ПАРАЛЛЕЛЬНЫЙ ПЕРЕНОС через edge_index
                    # Переносим эмбеддинги вдоль рёбер: x_j = Σ_{i: (i,j)∈E} W_{ij} · x_i
                    x_transported = parallel_transport_along_edges(x, edge_index, W_connection)
                else:
                    # ШАГ 1: ПАРАЛЛЕЛЬНЫЙ ПЕРЕНОС через матрицу смежности
                    # Сначала: graph convolution (передача сообщений)
                    if adj_matrix.is_sparse:
                        x_conv = torch.sparse.mm(adj_matrix, x)
                    else:
                        x_conv = torch.mm(adj_matrix, x)
                    # Затем: применяем матрицу связи W_{ij}
                    x_transported = x_conv @ W_connection
            else:
                # Без параллельного переноса, просто graph convolution
                if self.use_edge_index:
                    # Имитируем graph convolution через edge_index
                    x_transported = self._graph_conv_edge_index(x, edge_index)
                else:
                    if adj_matrix.is_sparse:
                        x_transported = torch.sparse.mm(adj_matrix, x)
                    else:
                        x_transported = torch.mm(adj_matrix, x)

            # ШАГ 2: ЛОКАЛЬНОЕ ПРЕОБРАЗОВАНИЕ (Group & Shuffle внутри пространства волокон)
            x_transformed = self.local_transform_layers[layer_idx](x_transported)

            # ШАГ 3: RESIDUAL-СВЯЗЬ
            # x^(l+1) = (1-α)·transformed + α·x^(0)
            # Это предотвращает over-smoothing, сохраняя начальную информацию
            x = (1 - self.residual_alpha) * x_transformed + \
                self.residual_alpha * x_init

            # ШАГ 4: DROPOUT
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)

            all_layer_embeddings.append(x)

        # АГРЕГАЦИЯ СЛОЁВ: Взвешенная комбинация всех слоёв
        layer_weights_normalized = F.softmax(self.layer_weights, dim=0)
        x_final = sum([
            w * emb for w, emb in zip(layer_weights_normalized, all_layer_embeddings)
        ])

        # Разделяем обратно на пользователей и товары
        user_embeddings = x_final[:self.n_users]
        item_embeddings = x_final[self.n_users:]
        
        return user_embeddings, item_embeddings
    
    def _graph_conv_edge_index(self, x, edge_index):
        """Простая graph convolution через edge_index (запасной вариант без параллельного переноса)"""
        src, dst = edge_index
        x_aggregated = torch.zeros_like(x)
        x_aggregated.index_add_(0, dst, x[src])
        return x_aggregated

    def predict(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Предсказывает оценки для пар пользователь-товар.
        """
        user_emb, item_emb = self.get_all_embeddings(adj_matrix, edge_index)
        user_emb_selected = user_emb[users]
        item_emb_selected = item_emb[items]
        scores = (user_emb_selected * item_emb_selected).sum(dim=1)
        return scores
    
    def get_all_embeddings(
        self,
        adj_matrix: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Получает все эмбеддинги для пользователей и товаров."""
        return self.forward(adj_matrix, edge_index)

    def get_orthogonality_errors(self) -> torch.Tensor:
        """
        Получает ошибки ортогональности для всех слоёв.

        Полезно для мониторинга во время обучения.

        Returns:
            Тензор с ошибками ортогональности [n_layers]
        """
        errors = []
        for layer in self.local_transform_layers:
            error = layer.get_orthogonality_error()
            errors.append(error)
        return torch.stack(errors)

    def get_orthogonality_metrics(self) -> Dict[str, torch.Tensor]:
        """
        Получает метрики ортогональности во время выполнения для мониторинга.

        Returns:
            Словарь с агрегированной статистикой ортогональности.
        """
        metrics: Dict[str, torch.Tensor] = {}

        local_fro = []
        local_max = []
        for layer in self.local_transform_layers:
            if hasattr(layer, 'get_orthogonality_metrics'):
                fro_error, max_deviation = layer.get_orthogonality_metrics()
                local_max.append(max_deviation)
            else:
                fro_error = layer.get_orthogonality_error()
            local_fro.append(fro_error)

        if local_fro:
            local_fro_tensor = torch.stack(local_fro)
            metrics['local_fro_mean'] = local_fro_tensor.mean()
            metrics['local_fro_max'] = local_fro_tensor.max()
        if local_max:
            metrics['local_max_dev'] = torch.stack(local_max).max()

        if self.use_parallel_transport:
            conn_fro = []
            conn_max = []
            for layer in self.connection_layers:
                if hasattr(layer, 'get_orthogonality_metrics'):
                    fro_error, max_deviation = layer.get_orthogonality_metrics()
                    conn_fro.append(fro_error)
                    conn_max.append(max_deviation)
            if conn_fro:
                conn_fro_tensor = torch.stack(conn_fro)
                metrics['conn_fro_mean'] = conn_fro_tensor.mean()
                metrics['conn_fro_max'] = conn_fro_tensor.max()
            if conn_max:
                metrics['conn_max_dev'] = torch.stack(conn_max).max()

        return metrics
    
    def get_layer_embeddings(
        self,
        adj_matrix: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Получает эмбеддинги для каждого слоя (для анализа over-smoothing).

        Возвращает эмбеддинги на каждом слое для анализа:
        - Over-smoothing: схожесть между эмбеддингами по слоям
        - Сохранение нормы: ||x^(l)|| должна оставаться стабильной
        """
        # Начальные эмбеддинги
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)

        all_embeddings = [x.clone()]

        # Проходим через каждый слой
        for layer_idx in range(self.n_layers):
            if self.use_parallel_transport:
                W_connection = self.connection_layers[layer_idx]()
                
                if self.use_edge_index and edge_index is not None:
                    x_transported = parallel_transport_along_edges(x, edge_index, W_connection)
                elif adj_matrix is not None:
                    if adj_matrix.is_sparse:
                        x_conv = torch.sparse.mm(adj_matrix, x)
                    else:
                        x_conv = torch.mm(adj_matrix, x)
                    x_transported = x_conv @ W_connection
                else:
                    raise ValueError("Должна быть передана либо adj_matrix, либо edge_index")
            else:
                if self.use_edge_index and edge_index is not None:
                    x_transported = self._graph_conv_edge_index(x, edge_index)
                elif adj_matrix is not None:
                    if adj_matrix.is_sparse:
                        x_transported = torch.sparse.mm(adj_matrix, x)
                    else:
                        x_transported = torch.mm(adj_matrix, x)
                else:
                    raise ValueError("Должна быть передана либо adj_matrix, либо edge_index")

            # Локальное преобразование
            x = self.local_transform_layers[layer_idx](x_transported)
            all_embeddings.append(x.clone())

        return all_embeddings

    def reset_parameters(self):
        """
        Сбрасывает параметры к начальным значениям.
        """
        # Сбрасываем эмбеддинги
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

        # Сбрасываем слои
        for layer in self.local_transform_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

