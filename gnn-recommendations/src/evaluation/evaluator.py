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

# Импорты
from ..training.metrics import (
    compute_metrics_from_topk,
    mean_cosine_similarity,
    mean_average_distance,
    embedding_variance
)
from ..data.dataset import RecommendationDataset


class Evaluator:
    """
    Класс для оценки моделей рекомендательных систем.
    
    Вычисляет метрики качества на test/validation set.
    """
    
    def __init__(
        self,
        k_values: List[int] = [10, 20],
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
            
            # Объединяем embeddings для анализа over-smoothing
            all_embeddings = torch.cat([user_emb, item_emb], dim=0)
            
            # Подготавливаем ground truth из test_data
            ground_truth = self._prepare_ground_truth(test_data)
            eval_users = sorted(ground_truth.keys())
            if not eval_users:
                return {}

            max_k = max(self.k_values) if self.k_values else 10
            topk_items = []
            train_items = self._get_train_items_by_user(dataset)
            batch_size = 2048

            for i in range(0, len(eval_users), batch_size):
                batch_users = eval_users[i:i + batch_size]
                batch_tensor = torch.tensor(batch_users, device=self.device)
                scores = user_emb[batch_tensor] @ item_emb.T
                for row_idx, user_id in enumerate(batch_users):
                    if user_id in train_items:
                        items = list(train_items[user_id])
                        if items:
                            scores[row_idx, items] = float('-inf')
                batch_topk = torch.topk(scores, k=max_k, dim=1).indices
                topk_items.append(batch_topk.cpu())

            topk_items = torch.cat(topk_items, dim=0)
            metrics = compute_metrics_from_topk(
                topk_items=topk_items,
                user_ids=eval_users,
                ground_truth=ground_truth,
                n_items=dataset.n_items,
                k_values=self.k_values
            )

            # Добавляем embedding метрики (анализ over-smoothing)
            if all_embeddings is not None:
                embeddings_cpu = all_embeddings.cpu()
                metrics['mcs'] = mean_cosine_similarity(embeddings_cpu)
                metrics['mad'] = mean_average_distance(embeddings_cpu)
                metrics['variance'] = embedding_variance(embeddings_cpu)

            return metrics
    
    def _get_train_items_by_user(self, dataset: RecommendationDataset) -> Dict[int, set]:
        """Создаёт маппинг user_id -> set(train/valid items)."""
        train_items = defaultdict(set)

        train_data = dataset.train_data
        if train_data is not None:
            if isinstance(train_data, list):
                for row in train_data:
                    user_id = int(row.get('userId', row.get('user_id')))
                    item_id = int(row.get('itemId', row.get('item_id')))
                    train_items[user_id].add(item_id)
            else:
                for _, row in train_data.iterrows():
                    user_id = int(row['userId'])
                    item_id = int(row['itemId'])
                    train_items[user_id].add(item_id)

        valid_data = dataset.valid_data
        if valid_data is not None:
            if isinstance(valid_data, list):
                for row in valid_data:
                    user_id = int(row.get('userId', row.get('user_id')))
                    item_id = int(row.get('itemId', row.get('item_id')))
                    train_items[user_id].add(item_id)
            else:
                for _, row in valid_data.iterrows():
                    user_id = int(row['userId'])
                    item_id = int(row['itemId'])
                    train_items[user_id].add(item_id)

        return train_items
    
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

