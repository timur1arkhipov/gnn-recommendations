"""
Скрипт для обучения одной модели на одном датасете.

Использование:
    python scripts/train_model.py --model lightgcn --dataset movie_lens
    python scripts/train_model.py --model groupshuffle_gnn --dataset gowalla
"""

import sys
from pathlib import Path
import yaml
import torch
import argparse

# Определяем корневую директорию проекта
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Добавляем путь к src в PYTHONPATH
sys.path.insert(0, str(project_root / "src"))

from models import (
    BPR_MF, LightGCN, GCNII, DGR, SVD_GCN, LayerGCN, GroupShuffleGNN
)
from data import RecommendationDataset
from training import Trainer


# Регистр моделей
MODEL_REGISTRY = {
    'bpr_mf': BPR_MF,
    'lightgcn': LightGCN,
    'gcnii': GCNII,
    'dgr': DGR,
    'svd_gcn': SVD_GCN,
    'layergcn': LayerGCN,
    'groupshuffle_gnn': GroupShuffleGNN,
    'groupshuffle': GroupShuffleGNN,
}


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
        print(f"⚠️  Конфигурация модели не найдена: {model_config_path}")
        print("   Используются значения по умолчанию")
        model_config = {'model': {}, 'training': {}}
    
    # Загружаем общую конфигурацию обучения
    training_config_path = root_dir / "config" / "training.yaml"
    if training_config_path.exists():
        with open(training_config_path, 'r', encoding='utf-8') as f:
            training_config = yaml.safe_load(f)
    else:
        print(f"⚠️  Конфигурация обучения не найдена: {training_config_path}")
        training_config = {}
    
    # Объединяем конфигурации (training из model_config имеет приоритет)
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
        raise ValueError(
            f"Неизвестная модель: {model_name}\n"
            f"Доступные модели: {list(MODEL_REGISTRY.keys())}"
        )
    
    # Загружаем конфигурацию модели
    model_config_path = root_dir / "config" / "models" / f"{model_name}.yaml"
    if model_config_path.exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        model_params = config.get('model', {})
    else:
        model_params = {}
    
    # Параметры по умолчанию
    default_params = {
        'embedding_dim': 64,
        'n_layers': 3,
    }
    default_params.update(model_params)
    
    # Создаем модель
    model = model_class(
        n_users=n_users,
        n_items=n_items,
        **default_params
    )
    
    return model


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Обучение модели рекомендательной системы")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Название модели"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["movie_lens", "book_crossing", "book-crossing", "gowalla"],
        help="Название датасета"
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
    
    args = parser.parse_args()
    
    # Определяем root_dir
    if args.root_dir is None:
        root_dir = project_root
    else:
        root_dir = Path(args.root_dir)
    
    # Устанавливаем seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"\n{'='*80}")
    print(f"ОБУЧЕНИЕ МОДЕЛИ: {args.model.upper()} на {args.dataset.upper()}")
    print(f"{'='*80}\n")
    
    try:
        # 1. Загружаем датасет
        print("ШАГ 1: Загрузка данных")
        print("-" * 80)
        dataset = RecommendationDataset(name=args.dataset, root_dir=str(root_dir))
        dataset.load_processed_data()
        print(f"✅ Данные загружены")
        print(f"   Пользователей: {dataset.n_users}")
        print(f"   Айтемов: {dataset.n_items}")
        print(f"   Train взаимодействий: {len(dataset.train_data)}")
        print(f"   Valid взаимодействий: {len(dataset.valid_data)}")
        print(f"   Test взаимодействий: {len(dataset.test_data)}")
        print()
        
        # 2. Создаем модель
        print("ШАГ 2: Создание модели")
        print("-" * 80)
        model = create_model(
            model_name=args.model,
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            root_dir=root_dir
        )
        print(f"✅ Модель создана: {model.__class__.__name__}")
        print(f"   Параметров: {model.get_parameters_count():,}")
        print()
        
        # 3. Загружаем конфигурацию обучения
        print("ШАГ 3: Загрузка конфигурации")
        print("-" * 80)
        config = load_config(args.model, root_dir)
        config['checkpoint_dir'] = str(root_dir / "results" / "checkpoints" / args.model / args.dataset)
        print(f"✅ Конфигурация загружена")
        print(f"   Learning rate: {config.get('learning_rate', 0.001)}")
        print(f"   Batch size: {config.get('batch_size', 2048)}")
        print(f"   Epochs: {config.get('epochs', 300)}")
        print()
        
        # 4. Создаем Trainer
        print("ШАГ 4: Инициализация Trainer")
        print("-" * 80)
        trainer = Trainer(model, dataset, config)
        print(f"✅ Trainer создан")
        print()
        
        # 5. Обучение
        print("ШАГ 5: Обучение модели")
        print("-" * 80)
        results = trainer.train()
        
        # 6. Финальная оценка на test set
        print("\nШАГ 6: Финальная оценка на test set")
        print("-" * 80)
        from evaluation import Evaluator
        evaluator = Evaluator()
        test_metrics = evaluator.evaluate(model, dataset, dataset.test_data)
        
        print("\nФИНАЛЬНЫЕ МЕТРИКИ НА TEST SET:")
        for metric_name, value in test_metrics.items():
            print(f"  {metric_name:20s}: {value:.4f}")
        
        # Сохраняем результаты
        results_file = root_dir / "results" / "metrics" / f"{args.model}_{args.dataset}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        final_results = {
            'model': args.model,
            'dataset': args.dataset,
            'seed': args.seed,
            'best_epoch': results['best_epoch'],
            'best_metric': results['best_metric'],
            'training_time': results['training_time'],
            'test_metrics': test_metrics
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Результаты сохранены: {results_file}")
        print(f"\n{'='*80}")
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

