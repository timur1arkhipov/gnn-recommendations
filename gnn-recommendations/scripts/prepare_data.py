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

# Добавляем путь к src в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

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
        choices=["movie_lens", "amazon_books", "gowalla"],
        help="Название датасета"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Корневая директория проекта"
    )
    
    args = parser.parse_args()
    
    try:
        dataset = prepare_dataset(args.dataset, args.root_dir)
        print("✅ Успешно!")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

