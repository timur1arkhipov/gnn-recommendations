"""
Анализ over-smoothing в GNN моделях.

Метрики:
- MCS (Mean Cosine Similarity): средняя косинусная схожесть между всеми парами embeddings
- MAD (Mean Average Distance): средняя L2 дистанция между embeddings
- Embedding Variance: дисперсия embeddings по измерениям
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F


class OversmoothingAnalyzer:
    """
    Анализатор over-smoothing для GNN моделей.
    
    Over-smoothing - проблема, когда при увеличении числа слоёв GNN
    все node embeddings становятся очень похожими друг на друга.
    """
    
    def __init__(self):
        """Инициализация анализатора."""
        pass
    
    def compute_mcs(
        self,
        embeddings: torch.Tensor,
        sample_size: Optional[int] = None
    ) -> float:
        """
        Вычисляет Mean Cosine Similarity (MCS) между всеми парами embeddings.
        
        MCS близкий к 1.0 означает сильный over-smoothing (все embeddings похожи).
        MCS близкий к 0.0 означает хорошую различимость embeddings.
        
        Args:
            embeddings: тензор embeddings [n_nodes, embedding_dim]
            sample_size: количество узлов для сэмплирования (для ускорения на больших графах)
        
        Returns:
            Средняя косинусная схожесть
        """
        # Нормализуем embeddings для косинусной схожести
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Если граф слишком большой, сэмплируем узлы
        if sample_size is not None and embeddings.shape[0] > sample_size:
            indices = torch.randperm(embeddings.shape[0])[:sample_size]
            embeddings_norm = embeddings_norm[indices]
        
        # Вычисляем матрицу косинусных схожестей
        # cos_sim[i,j] = dot(emb[i], emb[j]) / (||emb[i]|| * ||emb[j]||)
        cos_sim_matrix = embeddings_norm @ embeddings_norm.T  # [n, n]
        
        # Убираем диагональ (схожесть узла с самим собой = 1.0)
        n = cos_sim_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=cos_sim_matrix.device)
        cos_sim_values = cos_sim_matrix[mask]
        
        # Средняя косинусная схожесть
        mcs = cos_sim_values.mean().item()
        
        return mcs
    
    def compute_mad(
        self,
        embeddings: torch.Tensor,
        sample_size: Optional[int] = None
    ) -> float:
        """
        Вычисляет Mean Average Distance (MAD) - среднюю L2 дистанцию между embeddings.
        
        MAD близкий к 0.0 означает сильный over-smoothing (все embeddings в одной точке).
        MAD большой означает хорошую различимость embeddings.
        
        Args:
            embeddings: тензор embeddings [n_nodes, embedding_dim]
            sample_size: количество узлов для сэмплирования
        
        Returns:
            Средняя L2 дистанция
        """
        # Сэмплируем, если нужно
        if sample_size is not None and embeddings.shape[0] > sample_size:
            indices = torch.randperm(embeddings.shape[0])[:sample_size]
            embeddings = embeddings[indices]
        
        # Вычисляем попарные L2 дистанции
        # dist[i,j] = ||emb[i] - emb[j]||_2
        n = embeddings.shape[0]
        
        # Эффективное вычисление через broadcasting
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*<a,b>
        emb_squared_norm = (embeddings ** 2).sum(dim=1, keepdim=True)  # [n, 1]
        distances_squared = emb_squared_norm + emb_squared_norm.T - 2 * (embeddings @ embeddings.T)
        distances_squared = torch.clamp(distances_squared, min=0.0)  # Избегаем отрицательных из-за численных ошибок
        distances = torch.sqrt(distances_squared)
        
        # Убираем диагональ (дистанция узла до самого себя = 0)
        mask = ~torch.eye(n, dtype=torch.bool, device=distances.device)
        distance_values = distances[mask]
        
        # Средняя дистанция
        mad = distance_values.mean().item()
        
        return mad
    
    def compute_variance(self, embeddings: torch.Tensor) -> float:
        """
        Вычисляет дисперсию embeddings по всем измерениям.
        
        Низкая дисперсия означает, что все embeddings сконцентрированы в узкой области.
        
        Args:
            embeddings: тензор embeddings [n_nodes, embedding_dim]
        
        Returns:
            Средняя дисперсия по измерениям
        """
        # Дисперсия по каждому измерению
        variance_per_dim = embeddings.var(dim=0)  # [embedding_dim]
        
        # Средняя дисперсия
        mean_variance = variance_per_dim.mean().item()
        
        return mean_variance
    
    def compute_node_similarity_distribution(
        self,
        embeddings: torch.Tensor,
        sample_size: int = 1000
    ) -> Dict[str, float]:
        """
        Вычисляет статистику распределения попарных схожестей.
        
        Args:
            embeddings: тензор embeddings [n_nodes, embedding_dim]
            sample_size: количество пар для сэмплирования
        
        Returns:
            Словарь со статистиками: mean, std, min, max, median
        """
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Сэмплируем пары узлов
        n = embeddings.shape[0]
        n_pairs = min(sample_size, n * (n - 1) // 2)
        
        # Генерируем случайные пары
        similarities = []
        for _ in range(n_pairs):
            i, j = torch.randint(0, n, (2,))
            if i != j:
                sim = (embeddings_norm[i] * embeddings_norm[j]).sum().item()
                similarities.append(sim)
        
        similarities = np.array(similarities)
        
        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'median': float(np.median(similarities)),
            'q25': float(np.percentile(similarities, 25)),
            'q75': float(np.percentile(similarities, 75))
        }
    
    def analyze_layer_embeddings(
        self,
        layer_embeddings: List[torch.Tensor],
        layer_names: Optional[List[str]] = None,
        sample_size: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """
        Анализирует embeddings по слоям для выявления over-smoothing.
        
        Args:
            layer_embeddings: список тензоров embeddings для каждого слоя
            layer_names: названия слоёв (опционально)
            sample_size: размер сэмпла для вычислений
        
        Returns:
            Словарь с метриками для каждого слоя
        """
        if layer_names is None:
            layer_names = [f'layer_{i}' for i in range(len(layer_embeddings))]
        
        results = {}
        
        for layer_name, embeddings in zip(layer_names, layer_embeddings):
            # Переводим на CPU для вычислений
            embeddings = embeddings.detach().cpu()
            
            layer_metrics = {
                'mcs': self.compute_mcs(embeddings, sample_size=sample_size),
                'mad': self.compute_mad(embeddings, sample_size=sample_size),
                'variance': self.compute_variance(embeddings),
            }
            
            # Добавляем статистику распределения
            sim_stats = self.compute_node_similarity_distribution(embeddings, sample_size=sample_size)
            layer_metrics.update({f'sim_{k}': v for k, v in sim_stats.items()})
            
            results[layer_name] = layer_metrics
        
        return results
    
    def analyze_model(
        self,
        model: torch.nn.Module,
        adj_matrix: torch.Tensor,
        sample_size: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """
        Анализирует модель на over-smoothing.
        
        Требует, чтобы модель имела метод get_layer_embeddings().
        
        Args:
            model: GNN модель
            adj_matrix: матрица смежности графа
            sample_size: размер сэмпла для вычислений
        
        Returns:
            Словарь с метриками для каждого слоя
        """
        model.eval()
        
        with torch.no_grad():
            # Получаем embeddings по слоям
            if hasattr(model, 'get_layer_embeddings'):
                layer_embeddings = model.get_layer_embeddings(adj_matrix)
            else:
                raise AttributeError(
                    f"Модель {model.__class__.__name__} не имеет метода get_layer_embeddings(). "
                    "Добавьте этот метод для анализа over-smoothing."
                )
        
        # Анализируем
        results = self.analyze_layer_embeddings(
            layer_embeddings,
            sample_size=sample_size
        )
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        adj_matrix: torch.Tensor,
        sample_size: int = 1000
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Сравнивает несколько моделей по over-smoothing метрикам.
        
        Args:
            models: словарь {model_name: model}
            adj_matrix: матрица смежности графа
            sample_size: размер сэмпла для вычислений
        
        Returns:
            Словарь {model_name: {layer_name: metrics}}
        """
        results = {}
        
        for model_name, model in models.items():
            print(f"Анализ модели: {model_name}")
            try:
                model_results = self.analyze_model(model, adj_matrix, sample_size)
                results[model_name] = model_results
            except Exception as e:
                print(f"  Ошибка при анализе {model_name}: {e}")
                results[model_name] = {}
        
        return results
    
    def get_final_layer_mcs(
        self,
        model: torch.nn.Module,
        adj_matrix: torch.Tensor,
        sample_size: int = 1000
    ) -> float:
        """
        Получает MCS для финального слоя модели (для таблицы результатов).
        
        Args:
            model: GNN модель
            adj_matrix: матрица смежности
            sample_size: размер сэмпла
        
        Returns:
            MCS значение для последнего слоя
        """
        try:
            results = self.analyze_model(model, adj_matrix, sample_size)
            # Берём последний слой
            last_layer = list(results.keys())[-1]
            return results[last_layer]['mcs']
        except Exception as e:
            print(f"Ошибка при вычислении MCS: {e}")
            return float('nan')


def compute_oversmoothing_metrics(
    model: torch.nn.Module,
    adj_matrix: torch.Tensor,
    sample_size: int = 1000
) -> Dict[str, float]:
    """
    Удобная функция для быстрого вычисления over-smoothing метрик.
    
    Args:
        model: GNN модель
        adj_matrix: матрица смежности
        sample_size: размер сэмпла
    
    Returns:
        Словарь с метриками финального слоя
    """
    analyzer = OversmoothingAnalyzer()
    
    try:
        results = analyzer.analyze_model(model, adj_matrix, sample_size)
        # Возвращаем метрики последнего слоя
        last_layer = list(results.keys())[-1]
        return results[last_layer]
    except Exception as e:
        print(f"Ошибка при вычислении метрик: {e}")
        return {
            'mcs': float('nan'),
            'mad': float('nan'),
            'variance': float('nan')
        }

