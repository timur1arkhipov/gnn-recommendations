"""
Метрики для оценки качества рекомендательных систем.

Основные метрики:
- Recall@K: доля релевантных айтемов в топ-K рекомендациях
- NDCG@K: Normalized Discounted Cumulative Gain
- Precision@K: точность топ-K рекомендаций
- Coverage: доля уникальных айтемов в рекомендациях
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


def recall_at_k(
    scores: torch.Tensor,
    ground_truth: Dict[int, List[int]],
    k: int = 10
) -> float:
    """
    Вычисляет Recall@K.
    
    Recall@K = (количество релевантных айтемов в топ-K) / (общее количество релевантных)
    
    Args:
        scores: матрица scores [n_users, n_items]
        ground_truth: словарь {user_id: [item_id1, item_id2, ...]}
        k: количество топ рекомендаций
    
    Returns:
        Средний Recall@K по всем пользователям
    """
    n_users = scores.shape[0]
    recalls = []
    
    # Топ-K айтемов для каждого пользователя
    _, top_k_items = torch.topk(scores, k, dim=1)  # [n_users, k]
    
    for user_id in range(n_users):
        if user_id not in ground_truth:
            continue
        
        # Релевантные айтемы для пользователя
        relevant_items = set(ground_truth[user_id])
        if len(relevant_items) == 0:
            continue
        
        # Предсказанные айтемы (топ-K)
        predicted_items = set(top_k_items[user_id].cpu().numpy().tolist())
        
        # Вычисляем recall
        hits = len(relevant_items & predicted_items)
        recall = hits / len(relevant_items)
        recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def ndcg_at_k(
    scores: torch.Tensor,
    ground_truth: Dict[int, List[int]],
    k: int = 10
) -> float:
    """
    Вычисляет NDCG@K (Normalized Discounted Cumulative Gain).
    
    NDCG учитывает позицию релевантных айтемов в ранжированном списке.
    
    Args:
        scores: матрица scores [n_users, n_items]
        ground_truth: словарь {user_id: [item_id1, item_id2, ...]}
        k: количество топ рекомендаций
    
    Returns:
        Средний NDCG@K по всем пользователям
    """
    n_users = scores.shape[0]
    ndcgs = []
    
    # Топ-K айтемов для каждого пользователя
    _, top_k_items = torch.topk(scores, k, dim=1)  # [n_users, k]
    
    for user_id in range(n_users):
        if user_id not in ground_truth:
            continue
        
        relevant_items = set(ground_truth[user_id])
        if len(relevant_items) == 0:
            continue
        
        # Предсказанные айтемы (топ-K)
        predicted_items = top_k_items[user_id].cpu().numpy().tolist()
        
        # Вычисляем DCG
        dcg = 0.0
        for i, item_id in enumerate(predicted_items):
            if item_id in relevant_items:
                # DCG: rel / log2(i + 2), где rel = 1 (релевантный)
                dcg += 1.0 / np.log2(i + 2)
        
        # Вычисляем IDCG (Ideal DCG)
        idcg = 0.0
        num_relevant = min(len(relevant_items), k)
        for i in range(num_relevant):
            idcg += 1.0 / np.log2(i + 2)
        
        # NDCG = DCG / IDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    
    return np.mean(ndcgs) if ndcgs else 0.0


def precision_at_k(
    scores: torch.Tensor,
    ground_truth: Dict[int, List[int]],
    k: int = 10
) -> float:
    """
    Вычисляет Precision@K.
    
    Precision@K = (количество релевантных айтемов в топ-K) / K
    
    Args:
        scores: матрица scores [n_users, n_items]
        ground_truth: словарь {user_id: [item_id1, item_id2, ...]}
        k: количество топ рекомендаций
    
    Returns:
        Средний Precision@K по всем пользователям
    """
    n_users = scores.shape[0]
    precisions = []
    
    # Топ-K айтемов для каждого пользователя
    _, top_k_items = torch.topk(scores, k, dim=1)  # [n_users, k]
    
    for user_id in range(n_users):
        if user_id not in ground_truth:
            continue
        
        relevant_items = set(ground_truth[user_id])
        if len(relevant_items) == 0:
            continue
        
        predicted_items = set(top_k_items[user_id].cpu().numpy().tolist())
        
        # Precision = hits / k
        hits = len(relevant_items & predicted_items)
        precision = hits / k
        precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0


def coverage(
    scores: torch.Tensor,
    k: int = 10
) -> float:
    """
    Вычисляет Coverage - долю уникальных айтемов в рекомендациях.
    
    Coverage = (количество уникальных айтемов в топ-K) / (общее количество айтемов)
    
    Args:
        scores: матрица scores [n_users, n_items]
        k: количество топ рекомендаций
    
    Returns:
        Coverage (0.0 - 1.0)
    """
    n_items = scores.shape[1]
    
    # Топ-K айтемов для каждого пользователя
    _, top_k_items = torch.topk(scores, k, dim=1)  # [n_users, k]
    
    # Уникальные айтемы в рекомендациях
    unique_items = set(top_k_items.cpu().numpy().flatten().tolist())
    
    # Coverage
    coverage_score = len(unique_items) / n_items
    
    return coverage_score


def gini_index(
    scores: torch.Tensor,
    k: int = 10
) -> float:
    """
    Вычисляет Gini index - меру неравномерности распределения рекомендаций.
    
    Gini index показывает, насколько неравномерно items рекомендуются пользователям.
    - Gini = 0: все items рекомендуются одинаково часто (идеальное равенство)
    - Gini = 1: один item рекомендуется всем (максимальное неравенство)
    
    Args:
        scores: матрица scores [n_users, n_items]
        k: количество топ рекомендаций
    
    Returns:
        Gini index (0.0 - 1.0)
    """
    n_items = scores.shape[1]
    
    # Топ-K айтемов для каждого пользователя
    _, top_k_items = torch.topk(scores, k, dim=1)  # [n_users, k]
    
    # Подсчитываем частоту рекомендаций для каждого item
    item_counts = np.zeros(n_items)
    for item_id in top_k_items.cpu().numpy().flatten():
        item_counts[item_id] += 1
    
    # Сортируем частоты
    item_counts = np.sort(item_counts)
    
    # Вычисляем Gini index
    n = len(item_counts)
    cumsum = np.cumsum(item_counts)
    
    # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    gini = (2 * np.sum((np.arange(n) + 1) * item_counts)) / (n * cumsum[-1]) - (n + 1) / n
    
    return float(gini)


def compute_all_metrics(
    scores: torch.Tensor,
    ground_truth: Dict[int, List[int]],
    k_values: List[int] = [10, 20]
) -> Dict[str, float]:
    """
    Вычисляет все метрики для заданных значений K.
    
    Args:
        scores: матрица scores [n_users, n_items]
        ground_truth: словарь {user_id: [item_id1, item_id2, ...]}
        k_values: список значений K для метрик (по умолчанию [10, 20])
    
    Returns:
        Словарь с метриками: {'recall@10': ..., 'ndcg@10': ..., ...}
    """
    metrics = {}
    
    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(scores, ground_truth, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(scores, ground_truth, k)
        metrics[f'precision@{k}'] = precision_at_k(scores, ground_truth, k)
        metrics[f'coverage@{k}'] = coverage(scores, k)
        metrics[f'gini@{k}'] = gini_index(scores, k)
    
    return metrics

