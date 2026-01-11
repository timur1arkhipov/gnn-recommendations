"""
Скрипт для запуска всех моделей на всех датасетах.

Использование:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --models lightgcn groupshuffle_gnn
    python scripts/run_all_experiments.py --datasets movie_lens book_crossing
    python scripts/run_all_experiments.py --seed 42
"""

import sys
from pathlib import Path
import yaml
import torch
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import traceback

# Определяем корневую директорию проекта
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Добавляем путь к src в PYTHONPATH
sys.path.insert(0, str(project_root))

from src.models import (
    BPR_MF, LightGCN, GCNII, DGR, SVD_GCN, LayerGCN, OrthogonalBundleGNN
)
from src.data import RecommendationDataset
from src.training import Trainer
from src.evaluation import Evaluator


# Регистр моделей
MODEL_REGISTRY = {
    'bpr_mf': BPR_MF,
    'lightgcn': LightGCN,
    'gcnii': GCNII,
    'dgr': DGR,
    'svd_gcn': SVD_GCN,
    'layergcn': LayerGCN,
    'orthogonal_bundle': OrthogonalBundleGNN,
}

# Все датасеты
ALL_DATASETS = ['movie_lens', 'book_crossing', 'gowalla']

# Все модели (без дубликатов алиасов)
ALL_MODELS = ['bpr_mf', 'lightgcn', 'gcnii', 'dgr', 'svd_gcn', 'layergcn', 'orthogonal_bundle']


def load_config(model_name: str, root_dir: Path) -> dict:
    """
    Загружает конфигурацию для модели.
    
    Args:
        model_name: название модели
        root_dir: корневая директория проекта
    
    Returns:
        Словарь с конфигурацией
    """
    # Загружаем конфигурацию модели
    model_config_path = root_dir / "config" / "models" / f"{model_name}.yaml"
    if model_config_path.exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
    else:
        model_config = {'model': {}, 'training': {}}
    
    # Загружаем общую конфигурацию обучения
    training_config_path = root_dir / "config" / "training.yaml"
    if training_config_path.exists():
        with open(training_config_path, 'r', encoding='utf-8') as f:
            training_config = yaml.safe_load(f)
    else:
        training_config = {}
    
    # Объединяем конфигурации
    config = training_config.copy()
    if 'training' in model_config:
        config.update(model_config['training'])
    
    return config


def create_model(
    model_name: str,
    n_users: int,
    n_items: int,
    root_dir: Path
) -> torch.nn.Module:
    """
    Создает модель по имени.
    
    Args:
        model_name: название модели
        n_users: количество пользователей
        n_items: количество айтемов
        root_dir: корневая директория проекта
    
    Returns:
        Инициализированная модель
    """
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        raise ValueError(f"Неизвестная модель: {model_name}")
    
    # Загружаем конфигурацию модели
    model_config_path = root_dir / "config" / "models" / f"{model_name}.yaml"
    if model_config_path.exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        model_params = config.get('model', {})
    else:
        model_params = {}
    
    # Параметры по умолчанию (только для моделей, которые их поддерживают)
    default_params = {
        'embedding_dim': 64,
    }
    
    # n_layers только для GNN моделей (не для BPR-MF)
    if model_name != 'bpr_mf':
        default_params['n_layers'] = 3
    
    default_params.update(model_params)
    
    # Убираем параметры, которые модель не поддерживает
    import inspect
    sig = inspect.signature(model_class.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    filtered_params = {k: v for k, v in default_params.items() if k in valid_params}
    
    # Создаем модель
    model = model_class(
        n_users=n_users,
        n_items=n_items,
        **filtered_params
    )
    
    return model


def train_and_evaluate(
    model_name: str,
    dataset_name: str,
    root_dir: Path,
    seed: int = 42
) -> Dict:
    """
    Обучает модель на датасете и возвращает результаты.
    
    Args:
        model_name: название модели
        dataset_name: название датасета
        root_dir: корневая директория проекта
        seed: random seed
    
    Returns:
        Словарь с результатами обучения и оценки
    """
    result = {
        'model': model_name,
        'dataset': dataset_name,
        'seed': seed,
        'status': 'failed',
        'error': None,
        'training_time': None,
        'best_epoch': None,
        'best_metric': None,
        'test_metrics': None,
    }
    
    try:
        # Устанавливаем seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 1. Загружаем датасет
        # Нормализуем имя датасета (book-crossing -> book_crossing)
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
        # Преобразуем числовые значения из строк (если YAML загрузил их как строки)
        if isinstance(config.get('weight_decay'), str):
            config['weight_decay'] = float(config['weight_decay'])
        if isinstance(config.get('learning_rate'), str):
            config['learning_rate'] = float(config['learning_rate'])
        if isinstance(config.get('batch_size'), str):
            config['batch_size'] = int(config['batch_size'])
        if isinstance(config.get('epochs'), str):
            config['epochs'] = int(config['epochs'])
        
        config['checkpoint_dir'] = str(
            root_dir / "results" / "checkpoints" / model_name / normalized_dataset_name
        )
        
        # 4. Создаем Trainer
        trainer = Trainer(model, dataset, config)
        
        # 5. Обучение
        training_results = trainer.train()
        
        # 6. Финальная оценка на test set
        evaluator = Evaluator()
        test_metrics = evaluator.evaluate(model, dataset, dataset.test_data)
        
        # Формируем результат
        result.update({
            'status': 'success',
            'training_time': training_results['training_time'],
            'best_epoch': training_results['best_epoch'],
            'best_metric': training_results['best_metric'],
            'test_metrics': test_metrics,
        })
        
    except Exception as e:
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        print(f" Ошибка при обучении {model_name} на {dataset_name}: {e}")
    
    return result


def save_results(
    all_results: List[Dict],
    root_dir: Path,
    output_file: Optional[str] = None
):
    """
    Сохраняет все результаты в файлы.
    
    Args:
        all_results: список всех результатов
        root_dir: корневая директория проекта
        output_file: имя файла для сохранения (если None, генерируется автоматически)
    """
    results_dir = root_dir / "results" / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Генерируем имя файла
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"all_experiments_{timestamp}.json"
    
    output_path = results_dir / output_file
    
    # Сохраняем все результаты
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n Все результаты сохранены в: {output_path}")
    
    # Также сохраняем сводную таблицу (CSV)
    csv_path = results_dir / output_file.replace('.json', '.csv')
    save_summary_csv(all_results, csv_path)
    
    # Сохраняем отдельные файлы для каждой модели на каждом датасете
    save_individual_results(all_results, results_dir)


def save_summary_csv(all_results: List[Dict], csv_path: Path):
    """
    Сохраняет сводную таблицу результатов в CSV.
    
    Args:
        all_results: список всех результатов
        csv_path: путь к CSV файлу
    """
    import csv
    
    # Подготавливаем данные для CSV
    rows = []
    for result in all_results:
        if result['status'] == 'success':
            row = {
                'model': result['model'],
                'dataset': result['dataset'],
                'seed': result['seed'],
                'best_epoch': result['best_epoch'],
                'best_metric': result['best_metric'],
                'training_time': result['training_time'],
            }
            
            # Добавляем метрики
            if result['test_metrics']:
                for metric_name, value in result['test_metrics'].items():
                    row[metric_name] = value
            
            rows.append(row)
    
    if rows:
        # Записываем CSV
        fieldnames = ['model', 'dataset', 'seed', 'best_epoch', 'best_metric', 'training_time']
        if rows:
            # Добавляем все метрики
            for metric_name in rows[0].keys():
                if metric_name not in fieldnames:
                    fieldnames.append(metric_name)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f" Сводная таблица сохранена в: {csv_path}")


def save_individual_results(all_results: List[Dict], results_dir: Path):
    """
    Сохраняет результаты для каждой модели на каждом датасете в отдельные файлы.
    
    Args:
        all_results: список всех результатов
        results_dir: директория для сохранения
    """
    # Группируем результаты по модели и датасету
    grouped = {}
    for result in all_results:
        key = f"{result['model']}_{result['dataset']}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)
    
    # Сохраняем для каждой комбинации
    for key, results in grouped.items():
        output_file = results_dir / f"{key}_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def print_summary(all_results: List[Dict]):
    """
    Выводит сводку результатов.
    
    Args:
        all_results: список всех результатов
    """
    print(f"\n{'='*100}")
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print(f"{'='*100}\n")
    
    # Группируем по статусу
    successful = [r for r in all_results if r['status'] == 'success']
    failed = [r for r in all_results if r['status'] == 'failed']
    
    print(f" Успешно: {len(successful)}/{len(all_results)}")
    print(f" Ошибок: {len(failed)}/{len(all_results)}\n")
    
    if failed:
        print("ОШИБКИ:")
        for result in failed:
            print(f"  - {result['model']} на {result['dataset']}: {result.get('error', 'Unknown error')}")
        print()
    
    if successful:
        print("ЛУЧШИЕ РЕЗУЛЬТАТЫ (Recall@10):")
        print("-" * 100)
        print(f"{'Модель':<20} {'Датасет':<20} {'Recall@10':<12} {'NDCG@10':<12} {'Время (сек)':<12}")
        print("-" * 100)
        
        # Сортируем по Recall@10
        sorted_results = sorted(
            successful,
            key=lambda x: x.get('test_metrics', {}).get('recall@10', 0),
            reverse=True
        )
        
        for result in sorted_results[:20]:  # Топ-20
            metrics = result.get('test_metrics', {})
            print(
                f"{result['model']:<20} "
                f"{result['dataset']:<20} "
                f"{metrics.get('recall@10', 0):<12.4f} "
                f"{metrics.get('ndcg@10', 0):<12.4f} "
                f"{result.get('training_time', 0):<12.2f}"
            )
        
        print("-" * 100)


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Запуск всех моделей на всех датасетах"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Список моделей для обучения (по умолчанию все)"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Список датасетов (по умолчанию все)"
    )
    
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Корневая директория проекта"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed для воспроизводимости"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Имя файла для сохранения результатов"
    )
    
    parser.add_argument(
        "--skip_existing",
        action='store_true',
        help="Пропускать уже обученные модели"
    )
    
    args = parser.parse_args()
    
    # Определяем root_dir
    if args.root_dir is None:
        root_dir = project_root
    else:
        root_dir = Path(args.root_dir)
    
    print(f"\n{'='*100}")
    print("ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ")
    print(f"{'='*100}")
    print(f"Модели: {', '.join(args.models)}")
    print(f"Датасеты: {', '.join(args.datasets)}")
    print(f"Seed: {args.seed}")
    print(f"Всего экспериментов: {len(args.models) * len(args.datasets)}")
    print(f"{'='*100}\n")
    
    all_results = []
    total_experiments = len(args.models) * len(args.datasets)
    current_experiment = 0
    
    start_time = time.time()
    
    # Запускаем все эксперименты
    for model_name in args.models:
        for dataset_name in args.datasets:
            current_experiment += 1
            
            print(f"\n{'='*100}")
            print(f"ЭКСПЕРИМЕНТ {current_experiment}/{total_experiments}")
            print(f"Модель: {model_name.upper()} | Датасет: {dataset_name.upper()}")
            print(f"{'='*100}\n")
            
            # Проверяем, нужно ли пропустить
            if args.skip_existing:
                results_dir = root_dir / "results" / "experiments"
                existing_file = results_dir / f"{model_name}_{dataset_name}_results.json"
                if existing_file.exists():
                    print(f"⏭️  Пропускаем (уже существует): {existing_file}")
                    continue
            
            # Обучаем и оцениваем
            result = train_and_evaluate(
                model_name=model_name,
                dataset_name=dataset_name,
                root_dir=root_dir,
                seed=args.seed
            )
            
            all_results.append(result)
            
            # Выводим результат
            if result['status'] == 'success':
                metrics = result.get('test_metrics', {})
                print(f"\n Успешно завершено!")
                print(f"   Recall@10: {metrics.get('recall@10', 0):.4f}")
                print(f"   NDCG@10: {metrics.get('ndcg@10', 0):.4f}")
                print(f"   Время: {result.get('training_time', 0):.2f} сек")
            else:
                print(f"\n Ошибка: {result.get('error', 'Unknown')}")
    
    total_time = time.time() - start_time
    
    # Сохраняем результаты
    save_results(all_results, root_dir, args.output_file)
    
    # Выводим сводку
    print_summary(all_results)
    
    print(f"\n{'='*100}")
    print(f"ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print(f"Общее время: {total_time / 60:.2f} минут ({total_time:.2f} секунд)")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()

