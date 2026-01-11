"""
Скрипт для анализа и визуализации всех результатов экспериментов.

Создаёт:
- Таблицы сравнения моделей
- Графики производительности
- Over-smoothing анализ
- LaTeX таблицы для статьи
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict, List
import pandas as pd

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.statistics import (
    aggregate_multiple_runs,
    compare_all_models,
    print_comparison_table,
    format_result_with_std
)
from src.utils.visualization import (
    plot_model_comparison,
    create_results_summary_figure,
    save_latex_table
)


def load_multiple_seeds_results(results_dir: Path) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Загружает результаты экспериментов с несколькими seeds.
    
    Args:
        results_dir: директория с результатами
    
    Returns:
        Словарь {dataset: {model: [runs]}}
    """
    all_results = {}
    
    # Ищем файл all_results_multiple_seeds.json
    results_file = results_dir / "all_results_multiple_seeds.json"
    
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Преобразуем в нужный формат
        for dataset_name, dataset_results in data.items():
            all_results[dataset_name] = {}
            for model_name, runs in dataset_results.items():
                successful_runs = [r['metrics'] for r in runs if r.get('status') == 'success']
                if len(successful_runs) > 0:
                    all_results[dataset_name][model_name] = successful_runs
    
    return all_results


def create_summary_table(
    all_results: Dict[str, Dict[str, List[Dict]]],
    metrics: List[str],
    output_file: Path
):
    """
    Создаёт сводную таблицу результатов.
    
    Args:
        all_results: словарь {dataset: {model: [runs]}}
        metrics: список метрик
        output_file: путь к выходному файлу
    """
    rows = []
    
    for dataset_name, dataset_results in all_results.items():
        for model_name, runs in dataset_results.items():
            agg = aggregate_multiple_runs(runs)
            
            row = {
                'Dataset': dataset_name,
                'Model': model_name
            }
            
            for metric in metrics:
                if metric in agg:
                    mean = agg[metric]['mean']
                    std = agg[metric]['std']
                    row[metric] = format_result_with_std(mean, std)
                else:
                    row[metric] = 'N/A'
            
            rows.append(row)
    
    # Создаём DataFrame
    df = pd.DataFrame(rows)
    
    # Сохраняем в CSV
    df.to_csv(output_file, index=False)
    print(f"Сводная таблица сохранена: {output_file}")
    
    # Выводим в консоль
    print("\n" + "="*100)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Анализ и визуализация результатов экспериментов"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Директория с результатами multiple seeds"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Директория для сохранения графиков и таблиц"
    )
    
    parser.add_argument(
        "--baseline_model",
        type=str,
        default='layergcn',
        help="Baseline модель для сравнения"
    )
    
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        default=['recall@10', 'ndcg@10', 'precision@10', 'coverage'],
        help="Метрики для анализа"
    )
    
    args = parser.parse_args()
    
    # Определяем директории
    if args.results_dir is None:
        results_dir = project_root / "results" / "multiple_seeds"
    else:
        results_dir = Path(args.results_dir)
    
    if args.output_dir is None:
        output_dir = project_root / "results" / "figures"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("АНАЛИЗ И ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print(f"{'='*80}")
    print(f"Директория с результатами: {results_dir}")
    print(f"Директория для графиков: {output_dir}")
    print(f"Baseline модель: {args.baseline_model}")
    print(f"{'='*80}\n")
    
    # Загружаем результаты
    print("Загрузка результатов...")
    all_results = load_multiple_seeds_results(results_dir)
    
    if not all_results:
        print("❌ Результаты не найдены!")
        print(f"Проверьте, что файл {results_dir / 'all_results_multiple_seeds.json'} существует")
        return
    
    print(f"✅ Загружено результатов для {len(all_results)} датасетов")
    
    # Создаём сводную таблицу
    print("\nСоздание сводной таблицы...")
    create_summary_table(
        all_results,
        metrics=args.metrics,
        output_file=output_dir / "summary_table.csv"
    )
    
    # Для каждого датасета
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{'='*80}")
        print(f"ОБРАБОТКА: {dataset_name.upper()}")
        print(f"{'='*80}\n")
        
        # Агрегируем результаты
        aggregated = {}
        for model_name, runs in dataset_results.items():
            aggregated[model_name] = aggregate_multiple_runs(runs)
        
        # Создаём графики сравнения
        print("Создание графиков сравнения...")
        plot_model_comparison(
            aggregated,
            metrics=['recall@10', 'ndcg@10'],
            output_file=str(output_dir / f"{dataset_name}_comparison.png"),
            title=f"Model Comparison on {dataset_name.upper()}"
        )
        
        # Создаём LaTeX таблицу
        print("Создание LaTeX таблицы...")
        save_latex_table(
            aggregated,
            metrics=['recall@10', 'ndcg@10', 'precision@10', 'coverage'],
            output_file=str(output_dir / f"{dataset_name}_table.tex"),
            caption=f"Results on {dataset_name.replace('_', ' ').title()}",
            label=f"tab:{dataset_name}"
        )
        
        # Статистическое сравнение
        if args.baseline_model in dataset_results:
            print(f"Статистическое сравнение с {args.baseline_model}...")
            print_comparison_table(
                dataset_results,
                metrics=args.metrics,
                baseline_model=args.baseline_model
            )
    
    print(f"\n{'='*80}")
    print("ГОТОВО!")
    print(f"Все графики и таблицы сохранены в: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

