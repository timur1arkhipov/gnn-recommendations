"""
Скрипт для запуска экспериментов с несколькими random seeds.

Запускает каждую модель на каждом датасете N раз с разными seeds,
затем агрегирует результаты (mean ± std) и выполняет статистические тесты.
"""

import sys
from pathlib import Path
import argparse
import json
import time
from typing import Dict, List
import torch

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import RecommendationDataset
from src.evaluation.evaluator import Evaluator
from src.training.trainer import Trainer
from src.utils.statistics import (
    aggregate_multiple_runs,
    compare_all_models,
    save_statistical_results,
    print_comparison_table
)
from scripts.run_all_experiments import create_model, load_config


# Списки моделей и датасетов
ALL_MODELS = [
    'bpr_mf',
    'lightgcn',
    'gcnii',
    'dgr',
    'svd_gcn',
    'layergcn',
    'groupshuffle_gnn'
]

ALL_DATASETS = [
    'movie_lens',
    'book_crossing',
    # 'gowalla',  # Слишком большой для RTX 4060
]


def train_and_evaluate_with_seed(
    model_name: str,
    dataset_name: str,
    root_dir: Path,
    seed: int
) -> Dict:
    """
    Обучает и оценивает модель с заданным seed.
    
    Args:
        model_name: название модели
        dataset_name: название датасета
        root_dir: корневая директория проекта
        seed: random seed
    
    Returns:
        Словарь с метриками
    """
    print(f"\n{'='*80}")
    print(f"Модель: {model_name} | Датасет: {dataset_name} | Seed: {seed}")
    print(f"{'='*80}")
    
    try:
        # Устанавливаем seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 1. Загружаем датасет
        normalized_dataset_name = dataset_name.replace('-', '_')
        dataset = RecommendationDataset(name=normalized_dataset_name, root_dir=str(root_dir))
        dataset.load_processed_data()
        
        # 2. Создаем модель
        model = create_model(
            model_name=model_name,
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            root_dir=root_dir
        )
        
        # 3. Загружаем конфигурацию обучения
        config = load_config(model_name, root_dir)
        config['checkpoint_dir'] = str(
            root_dir / "results" / "checkpoints" / model_name / normalized_dataset_name / f"seed_{seed}"
        )
        
        # 4. Создаем Trainer
        trainer = Trainer(model, dataset, config)
        
        # 5. Обучение
        training_results = trainer.train()
        
        # 6. Финальная оценка на test set
        evaluator = Evaluator()
        test_metrics = evaluator.evaluate(model, dataset, dataset.test_data)
        
        print(f"\n✅ Успешно! Recall@10: {test_metrics.get('recall@10', 0):.4f}")
        
        return {
            'status': 'success',
            'seed': seed,
            'model': model_name,
            'dataset': dataset_name,
            'training_time': training_results['training_time'],
            'best_epoch': training_results['best_epoch'],
            'metrics': test_metrics
        }
        
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
        return {
            'status': 'failed',
            'seed': seed,
            'model': model_name,
            'dataset': dataset_name,
            'error': str(e)
        }


def run_multiple_seeds_experiment(
    models: List[str],
    datasets: List[str],
    seeds: List[int],
    root_dir: Path,
    output_dir: Path
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Запускает эксперименты с несколькими seeds.
    
    Args:
        models: список моделей
        datasets: список датасетов
        seeds: список seeds
        root_dir: корневая директория проекта
        output_dir: директория для сохранения результатов
    
    Returns:
        Словарь {dataset: {model: [results_per_seed]}}
    """
    all_results = {}
    
    total_experiments = len(models) * len(datasets) * len(seeds)
    current_experiment = 0
    
    start_time = time.time()
    
    for dataset_name in datasets:
        all_results[dataset_name] = {}
        
        for model_name in models:
            all_results[dataset_name][model_name] = []
            
            for seed in seeds:
                current_experiment += 1
                
                print(f"\n{'#'*80}")
                print(f"ЭКСПЕРИМЕНТ {current_experiment}/{total_experiments}")
                print(f"{'#'*80}")
                
                result = train_and_evaluate_with_seed(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    root_dir=root_dir,
                    seed=seed
                )
                
                all_results[dataset_name][model_name].append(result)
                
                # Сохраняем промежуточные результаты
                intermediate_file = output_dir / f"intermediate_{dataset_name}_{model_name}.json"
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results[dataset_name][model_name], f, indent=2)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print(f"Общее время: {total_time/60:.2f} минут")
    print(f"{'='*80}\n")
    
    return all_results


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Запуск экспериментов с несколькими random seeds"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Список моделей для обучения"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=ALL_DATASETS,
        choices=ALL_DATASETS + ['gowalla'],
        help="Список датасетов"
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        nargs='+',
        default=[42, 43, 44, 45, 46],
        help="Список random seeds (по умолчанию 5 seeds)"
    )
    
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Корневая директория проекта"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Директория для сохранения результатов"
    )
    
    parser.add_argument(
        "--baseline_model",
        type=str,
        default='layergcn',
        help="Baseline модель для статистического сравнения"
    )
    
    args = parser.parse_args()
    
    # Определяем root_dir
    if args.root_dir is None:
        root_dir = project_root
    else:
        root_dir = Path(args.root_dir)
    
    # Определяем output_dir
    if args.output_dir is None:
        output_dir = root_dir / "results" / "multiple_seeds"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ С НЕСКОЛЬКИМИ SEEDS")
    print(f"{'='*80}")
    print(f"Модели: {', '.join(args.models)}")
    print(f"Датасеты: {', '.join(args.datasets)}")
    print(f"Seeds: {args.seeds}")
    print(f"Всего экспериментов: {len(args.models) * len(args.datasets) * len(args.seeds)}")
    print(f"Результаты будут сохранены в: {output_dir}")
    print(f"{'='*80}\n")
    
    # Запускаем эксперименты
    all_results = run_multiple_seeds_experiment(
        models=args.models,
        datasets=args.datasets,
        seeds=args.seeds,
        root_dir=root_dir,
        output_dir=output_dir
    )
    
    # Обрабатываем результаты для каждого датасета
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{'='*80}")
        print(f"ОБРАБОТКА РЕЗУЛЬТАТОВ: {dataset_name.upper()}")
        print(f"{'='*80}\n")
        
        # Преобразуем в формат для статистических функций
        model_runs = {}
        for model_name, runs in dataset_results.items():
            successful_runs = [r['metrics'] for r in runs if r['status'] == 'success']
            if len(successful_runs) > 0:
                model_runs[model_name] = successful_runs
        
        # Агрегируем результаты
        aggregated = {}
        for model_name, runs in model_runs.items():
            aggregated[model_name] = aggregate_multiple_runs(runs)
        
        # Сохраняем агрегированные результаты
        agg_file = output_dir / f"{dataset_name}_aggregated.json"
        with open(agg_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2)
        print(f"Агрегированные результаты сохранены: {agg_file}")
        
        # Статистическое сравнение с baseline
        if args.baseline_model in model_runs:
            comparisons = compare_all_models(
                model_runs,
                baseline_model=args.baseline_model,
                metrics=['recall@10', 'ndcg@10', 'precision@10']
            )
            
            # Сохраняем сравнения
            comp_file = output_dir / f"{dataset_name}_comparisons.json"
            with open(comp_file, 'w', encoding='utf-8') as f:
                json.dump(comparisons, f, indent=2)
            print(f"Статистические сравнения сохранены: {comp_file}")
        
        # Выводим таблицу
        print_comparison_table(
            model_runs,
            metrics=['recall@10', 'ndcg@10', 'precision@10', 'coverage'],
            baseline_model=args.baseline_model
        )
    
    # Сохраняем все результаты
    all_results_file = output_dir / "all_results_multiple_seeds.json"
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nВсе результаты сохранены: {all_results_file}")
    
    print(f"\n{'='*80}")
    print("ГОТОВО!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

