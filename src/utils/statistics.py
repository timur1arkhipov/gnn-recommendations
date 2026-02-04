"""
Статистические тесты для сравнения моделей.

Включает:
- Paired t-test для сравнения двух моделей
- Агрегация результатов (mean ± std)
- Форматирование результатов с p-values
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json


def paired_t_test(
    results_a: List[float],
    results_b: List[float],
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Выполняет paired t-test для сравнения двух моделей.
    
    Paired t-test используется когда мы сравниваем две модели на одних и тех же данных
    (например, с разными random seeds).
    
    Args:
        results_a: результаты модели A (список метрик для каждого seed)
        results_b: результаты модели B (список метрик для каждого seed)
        alternative: тип теста ('two-sided', 'less', 'greater')
    
    Returns:
        Tuple из (t_statistic, p_value)
    """
    if len(results_a) != len(results_b):
        raise ValueError(f"Размеры должны совпадать: {len(results_a)} != {len(results_b)}")
    
    if len(results_a) < 2:
        raise ValueError("Нужно минимум 2 запуска для t-test")
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(results_a, results_b, alternative=alternative)
    
    return float(t_stat), float(p_value)


def compute_mean_std(results: List[float]) -> Tuple[float, float]:
    """
    Вычисляет mean ± std для списка результатов.
    
    Args:
        results: список метрик
    
    Returns:
        Tuple из (mean, std)
    """
    if len(results) == 0:
        return 0.0, 0.0
    
    mean = np.mean(results)
    std = np.std(results, ddof=1) if len(results) > 1 else 0.0
    
    return float(mean), float(std)


def aggregate_multiple_runs(
    runs: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Агрегирует результаты нескольких запусков (mean ± std).
    
    Args:
        runs: список словарей с метриками для каждого запуска
              Например: [{'recall@10': 0.05, 'ndcg@10': 0.03}, ...]
    
    Returns:
        Словарь {metric_name: {'mean': ..., 'std': ..., 'values': [...]}}
    """
    if len(runs) == 0:
        return {}
    
    # Собираем все метрики
    all_metrics = set()
    for run in runs:
        all_metrics.update(run.keys())
    
    # Агрегируем каждую метрику
    aggregated = {}
    for metric in all_metrics:
        values = [run.get(metric, float('nan')) for run in runs]
        # Фильтруем NaN
        values = [v for v in values if not np.isnan(v)]
        
        if len(values) > 0:
            mean, std = compute_mean_std(values)
            aggregated[metric] = {
                'mean': mean,
                'std': std,
                'values': values,
                'n_runs': len(values)
            }
        else:
            aggregated[metric] = {
                'mean': float('nan'),
                'std': float('nan'),
                'values': [],
                'n_runs': 0
            }
    
    return aggregated


def compare_models_statistical(
    model_a_runs: List[Dict[str, float]],
    model_b_runs: List[Dict[str, float]],
    metrics: Optional[List[str]] = None,
    alpha: float = 0.05
) -> Dict[str, Dict]:
    """
    Сравнивает две модели статистически по всем метрикам.
    
    Args:
        model_a_runs: результаты модели A (список запусков)
        model_b_runs: результаты модели B (список запусков)
        metrics: список метрик для сравнения (если None, берутся все)
        alpha: уровень значимости (по умолчанию 0.05)
    
    Returns:
        Словарь с результатами сравнения для каждой метрики
    """
    # Агрегируем результаты
    agg_a = aggregate_multiple_runs(model_a_runs)
    agg_b = aggregate_multiple_runs(model_b_runs)
    
    # Определяем метрики для сравнения
    if metrics is None:
        metrics = list(set(agg_a.keys()) & set(agg_b.keys()))
    
    comparison = {}
    
    for metric in metrics:
        if metric not in agg_a or metric not in agg_b:
            continue
        
        values_a = agg_a[metric]['values']
        values_b = agg_b[metric]['values']
        
        if len(values_a) < 2 or len(values_b) < 2:
            comparison[metric] = {
                'mean_a': agg_a[metric]['mean'],
                'std_a': agg_a[metric]['std'],
                'mean_b': agg_b[metric]['mean'],
                'std_b': agg_b[metric]['std'],
                't_stat': float('nan'),
                'p_value': float('nan'),
                'significant': False,
                'note': 'Недостаточно запусков для t-test'
            }
            continue
        
        # Выполняем paired t-test
        # Проверяем, что A лучше B (greater)
        try:
            t_stat, p_value = paired_t_test(values_a, values_b, alternative='greater')
            
            comparison[metric] = {
                'mean_a': agg_a[metric]['mean'],
                'std_a': agg_a[metric]['std'],
                'mean_b': agg_b[metric]['mean'],
                'std_b': agg_b[metric]['std'],
                't_stat': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'improvement': ((agg_a[metric]['mean'] - agg_b[metric]['mean']) / agg_b[metric]['mean'] * 100)
                               if agg_b[metric]['mean'] != 0 else float('inf')
            }
        except Exception as e:
            comparison[metric] = {
                'mean_a': agg_a[metric]['mean'],
                'std_a': agg_a[metric]['std'],
                'mean_b': agg_b[metric]['mean'],
                'std_b': agg_b[metric]['std'],
                't_stat': float('nan'),
                'p_value': float('nan'),
                'significant': False,
                'note': f'Ошибка: {str(e)}'
            }
    
    return comparison


def format_result_with_std(mean: float, std: float, decimals: int = 4) -> str:
    """
    Форматирует результат в виде "mean ± std".
    
    Args:
        mean: среднее значение
        std: стандартное отклонение
        decimals: количество знаков после запятой
    
    Returns:
        Строка вида "0.0623 ± 0.0015"
    """
    format_str = f"{{:.{decimals}f}}"
    return f"{format_str.format(mean)} ± {format_str.format(std)}"


def format_result_with_significance(
    mean: float,
    std: float,
    p_value: float,
    alpha: float = 0.05,
    decimals: int = 4
) -> str:
    """
    Форматирует результат с отметкой значимости.
    
    Args:
        mean: среднее значение
        std: стандартное отклонение
        p_value: p-value из статистического теста
        alpha: уровень значимости
        decimals: количество знаков после запятой
    
    Returns:
        Строка вида "0.0623 ± 0.0015*" (звёздочка если significant)
    """
    result = format_result_with_std(mean, std, decimals)
    
    if p_value < alpha:
        if p_value < 0.001:
            result += "***"
        elif p_value < 0.01:
            result += "**"
        else:
            result += "*"
    
    return result


def find_best_model(
    all_results: Dict[str, List[Dict[str, float]]],
    metric: str = 'recall@10'
) -> Tuple[str, float, float]:
    """
    Находит лучшую модель по заданной метрике.
    
    Args:
        all_results: словарь {model_name: [runs]}
        metric: метрика для сравнения
    
    Returns:
        Tuple из (best_model_name, mean, std)
    """
    best_model = None
    best_mean = -float('inf')
    best_std = 0.0
    
    for model_name, runs in all_results.items():
        agg = aggregate_multiple_runs(runs)
        if metric in agg:
            mean = agg[metric]['mean']
            std = agg[metric]['std']
            
            if mean > best_mean:
                best_model = model_name
                best_mean = mean
                best_std = std
    
    return best_model, best_mean, best_std


def compare_all_models(
    all_results: Dict[str, List[Dict[str, float]]],
    baseline_model: str,
    metrics: Optional[List[str]] = None,
    alpha: float = 0.05
) -> Dict[str, Dict[str, Dict]]:
    """
    Сравнивает все модели с baseline моделью.
    
    Args:
        all_results: словарь {model_name: [runs]}
        baseline_model: название baseline модели для сравнения
        metrics: список метрик для сравнения
        alpha: уровень значимости
    
    Returns:
        Словарь {model_name: {metric: comparison_results}}
    """
    if baseline_model not in all_results:
        raise ValueError(f"Baseline модель '{baseline_model}' не найдена в результатах")
    
    baseline_runs = all_results[baseline_model]
    
    comparisons = {}
    
    for model_name, model_runs in all_results.items():
        if model_name == baseline_model:
            continue
        
        comparison = compare_models_statistical(
            model_runs,
            baseline_runs,
            metrics=metrics,
            alpha=alpha
        )
        
        comparisons[model_name] = comparison
    
    return comparisons


def save_statistical_results(
    all_results: Dict[str, List[Dict[str, float]]],
    output_file: str,
    baseline_model: Optional[str] = None
):
    """
    Сохраняет статистические результаты в JSON файл.
    
    Args:
        all_results: словарь {model_name: [runs]}
        output_file: путь к выходному файлу
        baseline_model: название baseline модели (опционально)
    """
    # Агрегируем результаты
    aggregated = {}
    for model_name, runs in all_results.items():
        aggregated[model_name] = aggregate_multiple_runs(runs)
    
    # Сравниваем с baseline, если указан
    comparisons = {}
    if baseline_model and baseline_model in all_results:
        comparisons = compare_all_models(all_results, baseline_model)
    
    # Сохраняем
    output = {
        'aggregated_results': aggregated,
        'comparisons': comparisons if comparisons else None,
        'baseline_model': baseline_model
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Статистические результаты сохранены: {output_file}")


def print_comparison_table(
    all_results: Dict[str, List[Dict[str, float]]],
    metrics: List[str] = ['recall@10', 'ndcg@10', 'precision@10'],
    baseline_model: Optional[str] = None,
    alpha: float = 0.05
):
    """
    Выводит таблицу сравнения моделей в консоль.
    
    Args:
        all_results: словарь {model_name: [runs]}
        metrics: список метрик для отображения
        baseline_model: название baseline модели
        alpha: уровень значимости
    """
    print("\n" + "="*100)
    print("СТАТИСТИЧЕСКОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*100)
    
    # Агрегируем результаты
    aggregated = {}
    for model_name, runs in all_results.items():
        aggregated[model_name] = aggregate_multiple_runs(runs)
    
    # Сравниваем с baseline
    comparisons = {}
    if baseline_model and baseline_model in all_results:
        comparisons = compare_all_models(all_results, baseline_model, metrics, alpha)
    
    # Выводим таблицу
    header = f"{'Model':<20}"
    for metric in metrics:
        header += f" | {metric:>20}"
    print(header)
    print("-" * 100)
    
    for model_name in sorted(all_results.keys()):
        row = f"{model_name:<20}"
        
        for metric in metrics:
            if metric in aggregated[model_name]:
                mean = aggregated[model_name][metric]['mean']
                std = aggregated[model_name][metric]['std']
                
                # Добавляем звёздочку если significant
                if model_name in comparisons and metric in comparisons[model_name]:
                    p_value = comparisons[model_name][metric]['p_value']
                    result_str = format_result_with_significance(mean, std, p_value, alpha)
                else:
                    result_str = format_result_with_std(mean, std)
                
                row += f" | {result_str:>20}"
            else:
                row += f" | {'N/A':>20}"
        
        print(row)
    
    print("-" * 100)
    if baseline_model:
        print(f"Baseline: {baseline_model}")
    print(f"* p < {alpha}, ** p < 0.01, *** p < 0.001")
    print("="*100 + "\n")

