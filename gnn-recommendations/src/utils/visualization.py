"""
Визуализация результатов экспериментов.

Включает:
- Bar charts для сравнения моделей
- Over-smoothing графики (MCS по слоям)
- Training curves
- Depth analysis plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

# Настройка стиля
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def plot_model_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str] = ['recall@10', 'ndcg@10'],
    output_file: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Создаёт bar chart для сравнения моделей по метрикам.
    
    Args:
        results: словарь {model_name: {metric: {'mean': ..., 'std': ...}}}
        metrics: список метрик для отображения
        output_file: путь для сохранения графика
        title: заголовок графика
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    model_names = list(results.keys())
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        means = []
        stds = []
        labels = []
        
        for model_name in model_names:
            if metric in results[model_name]:
                means.append(results[model_name][metric]['mean'])
                stds.append(results[model_name][metric]['std'])
                labels.append(model_name)
        
        x_pos = np.arange(len(labels))
        
        # Bar chart с error bars
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8)
        
        # Цвета для разных моделей
        colors = sns.color_palette("husl", len(labels))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Добавляем значения на столбцы
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std, f'{mean:.4f}', ha='center', va='bottom', fontsize=9)
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_oversmoothing_by_layers(
    layer_results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = 'mcs',
    output_file: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Создаёт график over-smoothing метрики по слоям для разных моделей.
    
    Args:
        layer_results: словарь {model_name: {layer_name: {metric: value}}}
        metric: метрика для отображения ('mcs', 'mad', 'variance')
        output_file: путь для сохранения
        title: заголовок
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name, layers in layer_results.items():
        layer_names = sorted(layers.keys())
        values = [layers[layer][metric] for layer in layer_names if metric in layers[layer]]
        
        if len(values) > 0:
            x = list(range(len(values)))
            ax.plot(x, values, marker='o', linewidth=2, markersize=8, label=model_name)
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
    ax.set_title(title or f'{metric.upper()} by Layer', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    training_history: Dict[str, List[float]],
    metrics: List[str] = ['train_loss', 'recall@10', 'ndcg@10'],
    output_file: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Создаёт графики обучения (loss и метрики по эпохам).
    
    Args:
        training_history: словарь {metric_name: [values_per_epoch]}
        metrics: список метрик для отображения
        output_file: путь для сохранения
        title: заголовок
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        if metric in training_history:
            values = training_history[metric]
            epochs = list(range(1, len(values) + 1))
            
            ax.plot(epochs, values, linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} over Epochs', fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(metric, fontsize=13, fontweight='bold')
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_depth_analysis(
    depth_results: Dict[int, Dict[str, float]],
    metrics: List[str] = ['recall@10', 'mcs'],
    output_file: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Создаёт график зависимости метрик от глубины сети.
    
    Args:
        depth_results: словарь {n_layers: {metric: value}}
        metrics: список метрик для отображения
        output_file: путь для сохранения
        title: заголовок
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    depths = sorted(depth_results.keys())
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        values = [depth_results[d].get(metric, np.nan) for d in depths]
        
        ax.plot(depths, values, marker='o', linewidth=2, markersize=10)
        ax.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper()} vs Depth', fontsize=13, fontweight='bold')
        ax.set_xticks(depths)
        ax.grid(alpha=0.3)
        
        # Добавляем значения на точки
        for d, v in zip(depths, values):
            if not np.isnan(v):
                ax.text(d, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_ablation_study(
    ablation_results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['recall@10', 'ndcg@10'],
    output_file: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Создаёт bar chart для ablation study.
    
    Args:
        ablation_results: словарь {variant_name: {metric: value}}
        metrics: список метрик
        output_file: путь для сохранения
        title: заголовок
    """
    plot_model_comparison(
        {k: {m: {'mean': v[m], 'std': 0.0} for m in metrics if m in v}
         for k, v in ablation_results.items()},
        metrics=metrics,
        output_file=output_file,
        title=title or 'Ablation Study'
    )


def plot_heatmap(
    data: Dict[str, Dict[str, float]],
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = 'YlOrRd',
    annot: bool = True
):
    """
    Создаёт heatmap для визуализации матрицы значений.
    
    Args:
        data: словарь {row_name: {col_name: value}}
        output_file: путь для сохранения
        title: заголовок
        cmap: цветовая схема
        annot: показывать ли значения в ячейках
    """
    # Преобразуем в матрицу
    row_names = list(data.keys())
    col_names = list(next(iter(data.values())).keys())
    
    matrix = np.zeros((len(row_names), len(col_names)))
    for i, row_name in enumerate(row_names):
        for j, col_name in enumerate(col_names):
            matrix[i, j] = data[row_name].get(col_name, np.nan)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(matrix, annot=annot, fmt='.4f', cmap=cmap,
                xticklabels=col_names, yticklabels=row_names,
                cbar_kws={'label': 'Value'}, ax=ax)
    
    ax.set_title(title or 'Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {output_file}")
    else:
        plt.show()
    
    plt.close()


def create_results_summary_figure(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    oversmoothing_results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    dataset_name: str
):
    """
    Создаёт комплексную фигуру с несколькими графиками для статьи.
    
    Args:
        all_results: результаты моделей {model: {metric: {'mean': ..., 'std': ...}}}
        oversmoothing_results: результаты over-smoothing {model: {layer: {metric: value}}}
        output_dir: директория для сохранения
        dataset_name: название датасета
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Сравнение моделей по Recall@10 и NDCG@10
    plot_model_comparison(
        all_results,
        metrics=['recall@10', 'ndcg@10'],
        output_file=str(output_dir / f'{dataset_name}_model_comparison.png'),
        title=f'Model Comparison on {dataset_name.upper()}'
    )
    
    # 2. Over-smoothing по слоям (MCS)
    plot_oversmoothing_by_layers(
        oversmoothing_results,
        metric='mcs',
        output_file=str(output_dir / f'{dataset_name}_oversmoothing_mcs.png'),
        title=f'Over-smoothing Analysis (MCS) on {dataset_name.upper()}'
    )
    
    # 3. Over-smoothing по слоям (MAD)
    plot_oversmoothing_by_layers(
        oversmoothing_results,
        metric='mad',
        output_file=str(output_dir / f'{dataset_name}_oversmoothing_mad.png'),
        title=f'Over-smoothing Analysis (MAD) on {dataset_name.upper()}'
    )
    
    print(f"\nВсе графики сохранены в: {output_dir}")


def save_latex_table(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str],
    output_file: str,
    caption: str = "Model Comparison",
    label: str = "tab:results"
):
    """
    Сохраняет результаты в формате LaTeX таблицы.
    
    Args:
        all_results: результаты моделей
        metrics: список метрик
        output_file: путь к выходному файлу
        caption: заголовок таблицы
        label: метка для ссылки
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Заголовок таблицы
        n_cols = len(metrics) + 1
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{l{'c' * len(metrics)}}}\n")
        f.write("\\hline\n")
        
        # Заголовок столбцов
        header = "Model"
        for metric in metrics:
            header += f" & {metric.replace('@', '@')}"
        f.write(header + " \\\\\n")
        f.write("\\hline\n")
        
        # Строки с данными
        for model_name in sorted(all_results.keys()):
            row = model_name.replace('_', '\\_')
            for metric in metrics:
                if metric in all_results[model_name]:
                    mean = all_results[model_name][metric]['mean']
                    std = all_results[model_name][metric]['std']
                    row += f" & ${mean:.4f} \\pm {std:.4f}$"
                else:
                    row += " & N/A"
            f.write(row + " \\\\\n")
        
        # Конец таблицы
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX таблица сохранена: {output_file}")

