"""
Evaluator класс для оценки моделей рекомендательных систем.

Вычисляет метрики качества рекомендаций:
- Recall@K
- NDCG@K
- Precision@K
- Coverage
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

# Импорты - поддерживаем оба варианта импорта
try:
    # Сначала пробуем абсолютный импорт (когда src в sys.path)
    from training.metrics import compute_all_metrics
    from data import RecommendationDataset
except ImportError:
    # Если не работает, пробуем относительный (когда импортируем как модуль)
    from ..training.metrics import compute_all_metrics
    from ..data import RecommendationDataset


class Evaluator:
    """
    Класс для оценки моделей рекомендательных систем.
    
    Вычисляет метрики качества на test/validation set.
    """
    
    def __init__(
        self,
        k_values: List[int] = [10, 20, 50],
        device: Optional[torch.device] = None
    ):
        """
        Инициализация Evaluator.
        
        Args:
            k_values: список значений K для метрик (Recall@K, NDCG@K)
            device: устройство для вычислений
        """
        self.k_values = k_values
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataset: RecommendationDataset,
        test_data: Optional = None
    ) -> Dict[str, float]:
        """
        Оценивает модель на test set.
        
        Args:
            model: обученная модель
            dataset: датасет с данными
            test_data: test данные (если None, используется dataset.test_data)
        
        Returns:
            Словарь с метриками
        """
        model.eval()
        
        if test_data is None:
            test_data = dataset.test_data
        
        with torch.no_grad():
            # Получаем adjacency matrix
            adj_matrix = dataset.get_torch_adjacency(normalized=True)
            adj_matrix = adj_matrix.to(self.device)
            
            # Получаем все embeddings
            user_emb, item_emb = model.get_all_embeddings(adj_matrix)
            
            # Вычисляем scores для всех пар
            scores = user_emb @ item_emb.T  # [n_users, n_items]
            
            # ВАЖНО: Маскируем train items (и valid items если это test)
            # чтобы они не попадали в рекомендации
            train_mask = self._get_train_mask(dataset, scores.shape)
            scores = scores.masked_fill(train_mask.to(self.device), float('-inf'))
            
            # Подготавливаем ground truth из test_data
            ground_truth = self._prepare_ground_truth(test_data)
            
            # Вычисляем метрики
            metrics = compute_all_metrics(
                scores.cpu(),
                ground_truth,
                k_values=self.k_values
            )
            
            return metrics
    
    def _get_train_mask(self, dataset: RecommendationDataset, shape: tuple) -> torch.Tensor:
        """
        Создаёт маску для train (и valid) items.
        
        Args:
            dataset: датасет
            shape: размер матрицы scores (n_users, n_items)
        
        Returns:
            Булева маска [n_users, n_items], где True = train/valid item
        """
        n_users, n_items = shape
        mask = torch.zeros(n_users, n_items, dtype=torch.bool)
        
        # Маскируем train items
        train_data = dataset.train_data
        if train_data is not None:
            if isinstance(train_data, list):
                for row in train_data:
                    user_id = int(row.get('userId', row.get('user_id')))
                    item_id = int(row.get('itemId', row.get('item_id')))
                    if user_id < n_users and item_id < n_items:
                        mask[user_id, item_id] = True
            else:
                # DataFrame
                for _, row in train_data.iterrows():
                    user_id = int(row['userId'])
                    item_id = int(row['itemId'])
                    if user_id < n_users and item_id < n_items:
                        mask[user_id, item_id] = True
        
        # Также маскируем valid items при оценке на test
        valid_data = dataset.valid_data
        if valid_data is not None:
            if isinstance(valid_data, list):
                for row in valid_data:
                    user_id = int(row.get('userId', row.get('user_id')))
                    item_id = int(row.get('itemId', row.get('item_id')))
                    if user_id < n_users and item_id < n_items:
                        mask[user_id, item_id] = True
            else:
                # DataFrame
                for _, row in valid_data.iterrows():
                    user_id = int(row['userId'])
                    item_id = int(row['itemId'])
                    if user_id < n_users and item_id < n_items:
                        mask[user_id, item_id] = True
        
        return mask
    
    def _prepare_ground_truth(self, test_data) -> Dict[int, List[int]]:
        """
        Подготавливает ground truth из test данных.
        
        Args:
            test_data: test данные (DataFrame или список словарей)
        
        Returns:
            Словарь {user_id: [item_id1, item_id2, ...]}
        """
        ground_truth = defaultdict(list)
        
        if isinstance(test_data, list):
            # Список словарей
            for row in test_data:
                user_id = row.get('userId', row.get('user_id'))
                item_id = row.get('itemId', row.get('item_id'))
                if user_id is not None and item_id is not None:
                    ground_truth[int(user_id)].append(int(item_id))
        else:
            # DataFrame
            for _, row in test_data.iterrows():
                user_id = int(row['userId'])
                item_id = int(row['itemId'])
                ground_truth[user_id].append(item_id)
        
        return dict(ground_truth)
    
    def evaluate_batch(
        self,
        model: torch.nn.Module,
        users: torch.Tensor,
        items: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Оценивает модель на батче user-item пар.
        
        Args:
            model: модель
            users: тензор с ID пользователей [batch_size]
            items: тензор с ID айтемов [batch_size]
            adj_matrix: adjacency matrix
        
        Returns:
            Тензор с scores [batch_size]
        """
        model.eval()
        with torch.no_grad():
            scores = model.predict(users, items, adj_matrix)
        return scores

