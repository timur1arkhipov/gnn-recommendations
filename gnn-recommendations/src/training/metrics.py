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


def mean_cosine_similarity(embeddings: torch.Tensor) -> float:
    """
    Вычисляет Mean Cosine Similarity (MCS) между embeddings.
    
    MCS показывает среднюю похожесть между всеми парами embeddings.
    Используется для анализа over-smoothing:
    - MCS близко к 1: сильный over-smoothing (все embeddings похожи)
    - MCS близко к 0: хорошая дифференциация
    
    Args:
        embeddings: тензор embeddings [n_nodes, embedding_dim]
    
    Returns:
        Средняя косинусная похожесть (0.0 - 1.0)
    """
    # Нормализуем embeddings
    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Косинусная похожесть между всеми парами
    similarity_matrix = embeddings_norm @ embeddings_norm.T  # [n_nodes, n_nodes]
    
    # Убираем диагональ (похожесть с самим собой)
    n_nodes = embeddings.shape[0]
    mask = ~torch.eye(n_nodes, dtype=torch.bool, device=embeddings.device)
    similarities = similarity_matrix[mask]
    
    # Средняя похожесть
    mcs = similarities.mean().item()
    
    return float(mcs)


def mean_average_distance(embeddings: torch.Tensor) -> float:
    """
    Вычисляет Mean Average Distance (MAD) между embeddings.
    
    MAD показывает среднее расстояние между всеми парами embeddings.
    Используется для анализа разделимости:
    - MAD большое: embeddings хорошо разделены
    - MAD маленькое: embeddings слишком близко (возможно over-smoothing)
    
    Args:
        embeddings: тензор embeddings [n_nodes, embedding_dim]
    
    Returns:
        Среднее расстояние
    """
    n_nodes = embeddings.shape[0]
    
    # Евклидово расстояние между всеми парами
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j
    norms_sq = (embeddings ** 2).sum(dim=1, keepdim=True)  # [n_nodes, 1]
    distances_sq = norms_sq + norms_sq.T - 2 * (embeddings @ embeddings.T)
    distances_sq = torch.clamp(distances_sq, min=0)  # Избегаем отрицательных из-за точности
    distances = torch.sqrt(distances_sq)
    
    # Убираем диагональ
    mask = ~torch.eye(n_nodes, dtype=torch.bool, device=embeddings.device)
    distances_flat = distances[mask]
    
    # Среднее расстояние
    mad = distances_flat.mean().item()
    
    return float(mad)


def embedding_variance(embeddings: torch.Tensor) -> float:
    """
    Вычисляет Variance (дисперсию) embeddings.
    
    Variance показывает разброс значений в embeddings:
    - Variance большая: embeddings разнообразны
    - Variance маленькая: embeddings коллапсировали (over-smoothing)
    
    Args:
        embeddings: тензор embeddings [n_nodes, embedding_dim]
    
    Returns:
        Средняя дисперсия по всем измерениям
    """
    # Дисперсия по каждому измерению
    variance_per_dim = embeddings.var(dim=0)  # [embedding_dim]
    
    # Средняя дисперсия
    avg_variance = variance_per_dim.mean().item()
    
    return float(avg_variance)


def compute_all_metrics(
    scores: torch.Tensor,
    ground_truth: Dict[int, List[int]],
    k_values: List[int] = [10, 20],
    embeddings: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Вычисляет все метрики для заданных значений K.
    
    Args:
        scores: матрица scores [n_users, n_items]
        ground_truth: словарь {user_id: [item_id1, item_id2, ...]}
        k_values: список значений K для метрик (по умолчанию [10, 20])
        embeddings: опциональные embeddings для анализа [n_nodes, embedding_dim]
    
    Returns:
        Словарь с метриками: {'recall@10': ..., 'ndcg@10': ..., ...}
    """
    metrics = {}
    
    # Метрики для рекомендаций
    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(scores, ground_truth, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(scores, ground_truth, k)
        metrics[f'precision@{k}'] = precision_at_k(scores, ground_truth, k)
        metrics[f'coverage@{k}'] = coverage(scores, k)
        metrics[f'gini@{k}'] = gini_index(scores, k)
    
    # Метрики для embeddings (анализ over-smoothing)
    if embeddings is not None:
        metrics['mcs'] = mean_cosine_similarity(embeddings)
        metrics['mad'] = mean_average_distance(embeddings)
        metrics['variance'] = embedding_variance(embeddings)
    
    return metrics


def compute_metrics_from_topk(
    topk_items: torch.Tensor,
    user_ids: List[int],
    ground_truth: Dict[int, List[int]],
    n_items: int,
    k_values: List[int] = [10, 20]
) -> Dict[str, float]:
    """
    Вычисляет метрики по заранее рассчитанным top-K рекомендациям.
    
    Args:
        topk_items: тензор [n_eval_users, max_k] с рекомендациями
        user_ids: список user_id, соответствующий строкам topk_items
        ground_truth: словарь {user_id: [item_id1, item_id2, ...]}
        n_items: общее число айтемов
        k_values: список значений K
    
    Returns:
        Словарь метрик
    """
    if topk_items.numel() == 0:
        return {}

    max_k = topk_items.shape[1]
    topk_np = topk_items.cpu().numpy()
    metrics = {}

    for k in k_values:
        k = min(k, max_k)
        recalls = []
        ndcgs = []
        precisions = []

        for idx, user_id in enumerate(user_ids):
            if user_id not in ground_truth:
                continue
            relevant_items = set(ground_truth[user_id])
            if not relevant_items:
                continue

            predicted_items = topk_np[idx, :k].tolist()
            predicted_set = set(predicted_items)

            hits = len(relevant_items & predicted_set)
            recalls.append(hits / len(relevant_items))
            precisions.append(hits / k)

            dcg = 0.0
            for rank, item_id in enumerate(predicted_items):
                if item_id in relevant_items:
                    dcg += 1.0 / np.log2(rank + 2)
            idcg = 0.0
            num_relevant = min(len(relevant_items), k)
            for rank in range(num_relevant):
                idcg += 1.0 / np.log2(rank + 2)
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        metrics[f'recall@{k}'] = float(np.mean(recalls)) if recalls else 0.0
        metrics[f'ndcg@{k}'] = float(np.mean(ndcgs)) if ndcgs else 0.0
        metrics[f'precision@{k}'] = float(np.mean(precisions)) if precisions else 0.0

        # Coverage и Gini считаем по всем рекомендациям
        unique_items = set(topk_np[:, :k].flatten().tolist())
        metrics[f'coverage@{k}'] = len(unique_items) / max(1, n_items)

        item_counts = np.zeros(n_items, dtype=np.int64)
        for item_id in topk_np[:, :k].flatten():
            item_counts[item_id] += 1
        item_counts = np.sort(item_counts)
        if item_counts.sum() > 0:
            n = len(item_counts)
            cumsum = np.cumsum(item_counts)
            gini = (2 * np.sum((np.arange(n) + 1) * item_counts)) / (n * cumsum[-1]) - (n + 1) / n
            metrics[f'gini@{k}'] = float(gini)
        else:
            metrics[f'gini@{k}'] = 0.0

    return metrics

