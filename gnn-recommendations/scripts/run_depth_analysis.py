"""
Скрипт для анализа влияния глубины сети (количества слоёв) на производительность.

Обучает модели с разным количеством слоёв (2, 4, 8, 16) и анализирует:
- Recommendation quality (Recall@K, NDCG@K)
- Over-smoothing metrics (MCS, MAD)
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
from src.evaluation.oversmoothing import OversmoothingAnalyzer
from src.training.trainer import Trainer
from src.utils.visualization import plot_depth_analysis, plot_oversmoothing_by_layers
from scripts.run_all_experiments import load_config


def create_model_with_depth(
    model_name: str,
    n_users: int,
    n_items: int,
    n_layers: int,
    root_dir: Path
):
    """
    Создаёт модель с заданным количеством слоёв.
    
    Args:
        model_name: название модели
        n_users: количество пользователей
        n_items: количество айтемов
        n_layers: количество слоёв
        root_dir: корневая директория проекта
    
    Returns:
        Модель
    """
    # Загружаем конфигурацию модели
    config_file = root_dir / "config" / "models" / f"{model_name}.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Конфигурация не найдена: {config_file}")
    
    import yaml
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_config = config.get('model', {})
    
    # Переопределяем n_layers
    model_config['n_layers'] = n_layers
    
    # Создаём модель
    if model_name == 'lightgcn':
        from src.models.baselines.lightgcn import LightGCN
        model = LightGCN(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=model_config.get('embedding_dim', 64),
            n_layers=n_layers,
            init_scale=model_config.get('init_scale', 0.1)
        )
    elif model_name == 'gcnii':
        from src.models.baselines.gcnii import GCNII
        model = GCNII(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=model_config.get('embedding_dim', 64),
            n_layers=n_layers,
            alpha=model_config.get('alpha', 0.1),
            beta=model_config.get('beta', 0.5),
            dropout=model_config.get('dropout', 0.0),
            init_scale=model_config.get('init_scale', 0.1)
        )
    elif model_name == 'dgr':
        from src.models.baselines.dgr import DGR
        model = DGR(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=model_config.get('embedding_dim', 64),
            n_layers=n_layers,
            lambda_reg=model_config.get('lambda_reg', 0.1),
            dropout=model_config.get('dropout', 0.0),
            init_scale=model_config.get('init_scale', 0.1)
        )
    elif model_name == 'layergcn':
        from src.models.baselines.layergcn import LayerGCN
        model = LayerGCN(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=model_config.get('embedding_dim', 64),
            n_layers=n_layers,
            alpha=model_config.get('alpha', 0.5),
            dropout=model_config.get('dropout', 0.0),
            init_scale=model_config.get('init_scale', 0.1)
        )
    elif model_name == 'groupshuffle_gnn':
        from src.models.group_shuffle.model import GroupShuffleGNN
        model = GroupShuffleGNN(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=model_config.get('embedding_dim', 64),
            n_layers=n_layers,
            block_size=model_config.get('block_size', 8),
            residual_alpha=model_config.get('residual_alpha', 0.1),
            dropout=model_config.get('dropout', 0.0),
            init_scale=model_config.get('init_scale', 0.1)
        )
    else:
        raise ValueError(f"Модель {model_name} не поддерживается для depth analysis")
    
    return model


def run_depth_experiment(
    model_name: str,
    dataset_name: str,
    n_layers_list: List[int],
    root_dir: Path,
    output_dir: Path,
    seed: int = 42
) -> Dict[int, Dict]:
    """
    Запускает эксперимент с разными глубинами.
    
    Args:
        model_name: название модели
        dataset_name: название датасета
        n_layers_list: список количества слоёв
        root_dir: корневая директория проекта
        output_dir: директория для сохранения результатов
        seed: random seed
    
    Returns:
        Словарь {n_layers: results}
    """
    print(f"\n{'='*80}")
    print(f"DEPTH ANALYSIS: {model_name.upper()} на {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    # Устанавливаем seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Загружаем датасет
    normalized_dataset_name = dataset_name.replace('-', '_')
    dataset = RecommendationDataset(name=normalized_dataset_name, root_dir=str(root_dir))
    dataset.load_processed_data()
    
    # Загружаем adjacency matrix
    adj_matrix = dataset.get_torch_adjacency(normalized=True)
    
    results = {}
    
    for n_layers in n_layers_list:
        print(f"\n{'-'*80}")
        print(f"Обучение с n_layers = {n_layers}")
        print(f"{'-'*80}")
        
        try:
            # Создаём модель
            model = create_model_with_depth(
                model_name=model_name,
                n_users=dataset.n_users,
                n_items=dataset.n_items,
                n_layers=n_layers,
                root_dir=root_dir
            )
            
            # Загружаем конфигурацию обучения
            config = load_config(model_name, root_dir)
            config['checkpoint_dir'] = str(
                output_dir / "checkpoints" / model_name / dataset_name / f"layers_{n_layers}"
            )
            
            # Создаём Trainer
            trainer = Trainer(model, dataset, config)
            
            # Обучение
            training_results = trainer.train()
            
            # Оценка на test set
            evaluator = Evaluator()
            test_metrics = evaluator.evaluate(model, dataset, dataset.test_data)
            
            # Анализ over-smoothing
            analyzer = OversmoothingAnalyzer()
            adj_matrix_device = adj_matrix.to(trainer.device)
            oversmoothing_metrics = analyzer.analyze_model(model, adj_matrix_device, sample_size=1000)
            
            # Получаем MCS последнего слоя
            last_layer = list(oversmoothing_metrics.keys())[-1]
            final_mcs = oversmoothing_metrics[last_layer]['mcs']
            final_mad = oversmoothing_metrics[last_layer]['mad']
            
            results[n_layers] = {
                'training_time': training_results['training_time'],
                'best_epoch': training_results['best_epoch'],
                'test_metrics': test_metrics,
                'oversmoothing': oversmoothing_metrics,
                'final_mcs': final_mcs,
                'final_mad': final_mad,
                'status': 'success'
            }
            
            print(f"\n✅ Успешно!")
            print(f"   Recall@10: {test_metrics.get('recall@10', 0):.4f}")
            print(f"   MCS (final): {final_mcs:.4f}")
            print(f"   MAD (final): {final_mad:.4f}")
            
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")
            results[n_layers] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Анализ влияния глубины сети на производительность"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default='groupshuffle_gnn',
        choices=['lightgcn', 'gcnii', 'dgr', 'layergcn', 'groupshuffle_gnn'],
        help="Модель для анализа"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default='movie_lens',
        choices=['movie_lens', 'book_crossing', 'gowalla'],
        help="Датасет"
    )
    
    parser.add_argument(
        "--layers",
        type=int,
        nargs='+',
        default=[2, 4, 8, 16],
        help="Список количества слоёв"
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
        output_dir = root_dir / "experiments" / "depth_analysis"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("DEPTH ANALYSIS")
    print(f"{'='*80}")
    print(f"Модель: {args.model}")
    print(f"Датасет: {args.dataset}")
    print(f"Количество слоёв: {args.layers}")
    print(f"Seed: {args.seed}")
    print(f"Результаты будут сохранены в: {output_dir}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Запускаем эксперимент
    results = run_depth_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        n_layers_list=args.layers,
        root_dir=root_dir,
        output_dir=output_dir,
        seed=args.seed
    )
    
    total_time = time.time() - start_time
    
    # Сохраняем результаты
    results_file = output_dir / f"{args.model}_{args.dataset}_depth_analysis.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nРезультаты сохранены: {results_file}")
    
    # Создаём графики
    print("\nСоздание графиков...")
    
    # Подготавливаем данные для графиков
    depth_metrics = {}
    for n_layers, result in results.items():
        if result['status'] == 'success':
            depth_metrics[n_layers] = {
                'recall@10': result['test_metrics']['recall@10'],
                'ndcg@10': result['test_metrics']['ndcg@10'],
                'mcs': result['final_mcs'],
                'mad': result['final_mad']
            }
    
    # График: метрики vs глубина
    plot_depth_analysis(
        depth_metrics,
        metrics=['recall@10', 'ndcg@10'],
        output_file=str(output_dir / f"{args.model}_{args.dataset}_depth_performance.png"),
        title=f"{args.model.upper()} Performance vs Depth on {args.dataset.upper()}"
    )
    
    # График: over-smoothing vs глубина
    plot_depth_analysis(
        depth_metrics,
        metrics=['mcs', 'mad'],
        output_file=str(output_dir / f"{args.model}_{args.dataset}_depth_oversmoothing.png"),
        title=f"{args.model.upper()} Over-smoothing vs Depth on {args.dataset.upper()}"
    )
    
    # График: MCS по слоям для каждой глубины
    layer_results_for_plot = {}
    for n_layers, result in results.items():
        if result['status'] == 'success':
            layer_results_for_plot[f"{n_layers} layers"] = result['oversmoothing']
    
    plot_oversmoothing_by_layers(
        layer_results_for_plot,
        metric='mcs',
        output_file=str(output_dir / f"{args.model}_{args.dataset}_mcs_by_layers.png"),
        title=f"{args.model.upper()} MCS by Layer on {args.dataset.upper()}"
    )
    
    print(f"\nГрафики сохранены в: {output_dir}")
    
    # Выводим сводку
    print(f"\n{'='*80}")
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print(f"{'='*80}")
    print(f"{'Layers':<10} {'Recall@10':<12} {'NDCG@10':<12} {'MCS':<12} {'MAD':<12} {'Status':<10}")
    print(f"{'-'*80}")
    
    for n_layers in sorted(results.keys()):
        result = results[n_layers]
        if result['status'] == 'success':
            recall = result['test_metrics']['recall@10']
            ndcg = result['test_metrics']['ndcg@10']
            mcs = result['final_mcs']
            mad = result['final_mad']
            status = '✅'
            print(f"{n_layers:<10} {recall:<12.4f} {ndcg:<12.4f} {mcs:<12.4f} {mad:<12.4f} {status:<10}")
        else:
            print(f"{n_layers:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'❌':<10}")
    
    print(f"{'-'*80}")
    print(f"Общее время: {total_time/60:.2f} минут")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

