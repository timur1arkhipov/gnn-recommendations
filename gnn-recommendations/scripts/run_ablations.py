"""
Скрипт для ablation studies - проверка важности каждого компонента модели.

Для GroupShuffleGNN тестируем:
1. Full model (baseline)
2. Без residual connections (residual_alpha = 0)
3. Без shuffle (только ортогональная матрица, block_size = embedding_dim)
4. Разные block_size [4, 8, 16, 32]
"""

import sys
from pathlib import Path
import argparse
import json
import time
from typing import Dict
import torch

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import RecommendationDataset
from src.evaluation.evaluator import Evaluator
from src.training.trainer import Trainer
from src.models.group_shuffle.model import GroupShuffleGNN
from src.utils.visualization import plot_ablation_study
from scripts.run_all_experiments import load_config


def create_ablation_variant(
    variant_name: str,
    n_users: int,
    n_items: int,
    base_config: Dict
) -> GroupShuffleGNN:
    """
    Создаёт вариант модели для ablation study.
    
    Args:
        variant_name: название варианта
        n_users: количество пользователей
        n_items: количество айтемов
        base_config: базовая конфигурация
    
    Returns:
        Модель
    """
    embedding_dim = base_config.get('embedding_dim', 64)
    n_layers = base_config.get('n_layers', 3)
    init_scale = base_config.get('init_scale', 0.1)
    dropout = base_config.get('dropout', 0.0)
    
    if variant_name == 'full':
        # Полная модель
        model = GroupShuffleGNN(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            block_size=base_config.get('block_size', 8),
            residual_alpha=base_config.get('residual_alpha', 0.1),
            dropout=dropout,
            init_scale=init_scale
        )
    elif variant_name == 'no_residual':
        # Без residual connections
        model = GroupShuffleGNN(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            block_size=base_config.get('block_size', 8),
            residual_alpha=0.0,  # Отключаем residual
            dropout=dropout,
            init_scale=init_scale
        )
    elif variant_name == 'no_shuffle':
        # Без shuffle (block_size = embedding_dim)
        model = GroupShuffleGNN(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            block_size=embedding_dim,  # Нет shuffle
            residual_alpha=base_config.get('residual_alpha', 0.1),
            dropout=dropout,
            init_scale=init_scale
        )
    elif variant_name.startswith('block_'):
        # Разные block_size
        block_size = int(variant_name.split('_')[1])
        model = GroupShuffleGNN(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            block_size=block_size,
            residual_alpha=base_config.get('residual_alpha', 0.1),
            dropout=dropout,
            init_scale=init_scale
        )
    else:
        raise ValueError(f"Неизвестный вариант: {variant_name}")
    
    return model


def run_ablation_study(
    dataset_name: str,
    variants: Dict[str, str],
    root_dir: Path,
    output_dir: Path,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Запускает ablation study.
    
    Args:
        dataset_name: название датасета
        variants: словарь {variant_name: description}
        root_dir: корневая директория проекта
        output_dir: директория для сохранения результатов
        seed: random seed
    
    Returns:
        Словарь {variant_name: results}
    """
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY на {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    # Устанавливаем seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Загружаем датасет
    normalized_dataset_name = dataset_name.replace('-', '_')
    dataset = RecommendationDataset(name=normalized_dataset_name, root_dir=str(root_dir))
    dataset.load_processed_data()
    
    # Загружаем базовую конфигурацию
    config = load_config('groupshuffle_gnn', root_dir)
    base_model_config = config.get('model', {})
    
    results = {}
    
    for variant_name, description in variants.items():
        print(f"\n{'-'*80}")
        print(f"Вариант: {variant_name} - {description}")
        print(f"{'-'*80}")
        
        try:
            # Создаём модель
            model = create_ablation_variant(
                variant_name=variant_name,
                n_users=dataset.n_users,
                n_items=dataset.n_items,
                base_config=base_model_config
            )
            
            # Обновляем checkpoint_dir
            config['checkpoint_dir'] = str(
                output_dir / "checkpoints" / dataset_name / f"ablation_{variant_name}"
            )
            
            # Создаём Trainer
            trainer = Trainer(model, dataset, config)
            
            # Обучение
            training_results = trainer.train()
            
            # Оценка на test set
            evaluator = Evaluator()
            test_metrics = evaluator.evaluate(model, dataset, dataset.test_data)
            
            results[variant_name] = {
                'description': description,
                'training_time': training_results['training_time'],
                'best_epoch': training_results['best_epoch'],
                'test_metrics': test_metrics,
                'status': 'success'
            }
            
            print(f"\n✅ Успешно!")
            print(f"   Recall@10: {test_metrics.get('recall@10', 0):.4f}")
            print(f"   NDCG@10: {test_metrics.get('ndcg@10', 0):.4f}")
            
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")
            results[variant_name] = {
                'description': description,
                'status': 'failed',
                'error': str(e)
            }
    
    return results


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Ablation study для GroupShuffleGNN"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default='movie_lens',
        choices=['movie_lens', 'book_crossing', 'gowalla'],
        help="Датасет"
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
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Определяем root_dir
    if args.root_dir is None:
        root_dir = project_root
    else:
        root_dir = Path(args.root_dir)
    
    # Определяем output_dir
    if args.output_dir is None:
        output_dir = root_dir / "experiments" / "ablations"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Определяем варианты для ablation
    variants = {
        'full': 'Full model (baseline)',
        'no_residual': 'Without residual connections',
        'no_shuffle': 'Without shuffle (only orthogonal)',
        'block_4': 'Block size = 4',
        'block_8': 'Block size = 8 (default)',
        'block_16': 'Block size = 16',
        'block_32': 'Block size = 32'
    }
    
    print(f"\n{'='*80}")
    print("ABLATION STUDY")
    print(f"{'='*80}")
    print(f"Датасет: {args.dataset}")
    print(f"Варианты: {len(variants)}")
    for name, desc in variants.items():
        print(f"  - {name}: {desc}")
    print(f"Seed: {args.seed}")
    print(f"Результаты будут сохранены в: {output_dir}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Запускаем ablation study
    results = run_ablation_study(
        dataset_name=args.dataset,
        variants=variants,
        root_dir=root_dir,
        output_dir=output_dir,
        seed=args.seed
    )
    
    total_time = time.time() - start_time
    
    # Сохраняем результаты
    results_file = output_dir / f"{args.dataset}_ablation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nРезультаты сохранены: {results_file}")
    
    # Создаём графики
    print("\nСоздание графиков...")
    
    # Подготавливаем данные для графиков
    ablation_metrics = {}
    for variant_name, result in results.items():
        if result['status'] == 'success':
            ablation_metrics[variant_name] = result['test_metrics']
    
    plot_ablation_study(
        ablation_metrics,
        metrics=['recall@10', 'ndcg@10', 'precision@10'],
        output_file=str(output_dir / f"{args.dataset}_ablation_comparison.png"),
        title=f"Ablation Study on {args.dataset.upper()}"
    )
    
    print(f"Графики сохранены в: {output_dir}")
    
    # Выводим сводку
    print(f"\n{'='*80}")
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print(f"{'='*80}")
    print(f"{'Variant':<20} {'Recall@10':<12} {'NDCG@10':<12} {'Precision@10':<12} {'Status':<10}")
    print(f"{'-'*80}")
    
    for variant_name, result in results.items():
        if result['status'] == 'success':
            recall = result['test_metrics']['recall@10']
            ndcg = result['test_metrics']['ndcg@10']
            precision = result['test_metrics']['precision@10']
            status = '✅'
            print(f"{variant_name:<20} {recall:<12.4f} {ndcg:<12.4f} {precision:<12.4f} {status:<10}")
        else:
            print(f"{variant_name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'❌':<10}")
    
    print(f"{'-'*80}")
    print(f"Общее время: {total_time/60:.2f} минут")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

