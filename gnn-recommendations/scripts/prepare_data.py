"""
Скрипт для подготовки данных.

Запускает полный pipeline подготовки данных:
1. Загрузка сырых данных
2. Препроцессинг (фильтрация, бинаризация)
3. Разделение на train/valid/test
4. Построение графов
"""

import sys
from pathlib import Path

# Определяем корневую директорию проекта (gnn-recommendations)
# Скрипт находится в gnn-recommendations/scripts/, поэтому parent.parent = gnn-recommendations
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent  # gnn-recommendations/

# Добавляем путь к src в PYTHONPATH, чтобы можно было импортировать модули
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import yaml
# Теперь можно импортировать модули из src
from data import RecommendationDataset


def prepare_dataset(dataset_name: str, root_dir: str = "."):
    """
    Подготавливает датасет: загружает, обрабатывает и строит графы.
    
    Args:
        dataset_name: название датасета ('movie_lens', 'amazon_books', 'gowalla')
        root_dir: корневая директория проекта
    """
    print(f"\n{'='*80}")
    print(f"ПОДГОТОВКА ДАТАСЕТА: {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    # Создаем объект датасета
    dataset = RecommendationDataset(name=dataset_name, root_dir=root_dir)
    
    # 1. Загружаем сырые данные
    print("ШАГ 1: Загрузка сырых данных")
    dataset.load_raw_data()
    
    # 2. Препроцессинг
    print("\nШАГ 2: Препроцессинг")
    dataset.preprocess()
    
    # 3. Разделение на train/valid/test
    print("\nШАГ 3: Разделение данных")
    dataset.split()
    
    # 4. Построение графов
    print("\nШАГ 4: Построение графов")
    dataset.build_graph()
    
    print(f"\n{'='*80}")
    print("ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА!")
    print(f"{'='*80}\n")
    
    # Выводим статистику
    print("СТАТИСТИКА:")
    print(f"  Пользователей: {dataset.n_users}")
    print(f"  Айтемов: {dataset.n_items}")
    print(f"  Train взаимодействий: {len(dataset.train_data)}")
    print(f"  Valid взаимодействий: {len(dataset.valid_data)}")
    print(f"  Test взаимодействий: {len(dataset.test_data)}")
    print(f"  Разреженность: {dataset.stats['sparsity']:.4f}")
    print()
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Подготовка данных для рекомендательных систем")
    parser.add_argument(
        "--dataset",
        type=str,
        default="movie_lens",
        choices=["movie_lens", "book_crossing", "book-crossing", "gowalla"],
        help="Название датасета"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Корневая директория проекта (по умолчанию определяется автоматически)"
    )
    
    args = parser.parse_args()
    
    # Если root_dir не указан, используем автоматическое определение
    if args.root_dir is None:
        args.root_dir = str(project_root)
    else:
        # Преобразуем в абсолютный путь
        args.root_dir = str(Path(args.root_dir).resolve())
    
    try:
        dataset = prepare_dataset(args.dataset, args.root_dir)
        print(" Успешно!")
    except Exception as e:
        print(f" Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

