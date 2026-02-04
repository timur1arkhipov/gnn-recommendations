"""
Тестовый скрипт для проверки обработки всех датасетов.

Проверяет:
1. Загрузку данных
2. Предобработку с правильной фильтрацией рейтингов
3. Глобальный 80-10-10 split
4. Построение графов
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в путь
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from data.dataset import RecommendationDataset


def test_dataset(dataset_name: str):
    """Тестирует обработку датасета."""
    print(f"\n{'#'*80}")
    print(f"# Тестирование датасета: {dataset_name}")
    print(f"{'#'*80}\n")

    try:
        # Создаем датасет
        dataset = RecommendationDataset(
            name=dataset_name,
            root_dir=str(root_dir)
        )

        # Загружаем и обрабатываем данные
        print("\n--- Этап 1: Загрузка данных ---")
        dataset.load_raw_data()

        print("\n--- Этап 2: Предобработка ---")
        dataset.preprocess()

        print("\n--- Этап 3: Разбиение на train/val/test ---")
        dataset.split()

        print("\n--- Этап 4: Построение графа ---")
        dataset.build_graph()

        # Выводим финальную статистику
        print(f"\n{'='*60}")
        print(f"ИТОГОВАЯ СТАТИСТИКА: {dataset_name}")
        print(f"{'='*60}")
        print(f"Пользователей: {dataset.n_users}")
        print(f"Айтемов: {dataset.n_items}")
        print(f"Train: {len(dataset.train_data)} ({len(dataset.train_data)/len(dataset.processed_data)*100:.1f}%)")
        print(f"Valid: {len(dataset.valid_data)} ({len(dataset.valid_data)/len(dataset.processed_data)*100:.1f}%)")
        print(f"Test: {len(dataset.test_data)} ({len(dataset.test_data)/len(dataset.processed_data)*100:.1f}%)")
        print(f"Разреженность: {dataset.stats['sparsity']:.6f}")
        print(f"{'='*60}\n")

        print(f"✓ Датасет {dataset_name} успешно обработан!")
        return True

    except Exception as e:
        print(f"\n✗ Ошибка при обработке датасета {dataset_name}:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Тестирует все датасеты."""
    datasets = [
        'ml-100k',      # MovieLens 100k
        'ml-1m',        # MovieLens 1M
        'facebook',     # Facebook
        # 'amazon_books', # Amazon Books (большой файл, можно закомментировать для быстрого теста)
    ]

    results = {}

    for dataset_name in datasets:
        success = test_dataset(dataset_name)
        results[dataset_name] = success

    # Итоговый отчет
    print(f"\n{'#'*80}")
    print("# ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'#'*80}\n")

    for dataset_name, success in results.items():
        status = "✓ OK" if success else "✗ FAIL"
        print(f"{status} - {dataset_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nВсего тестов: {total}")
    print(f"Пройдено: {passed}")
    print(f"Провалено: {total - passed}")

    if passed == total:
        print("\n🎉 Все датасеты успешно обработаны!")
    else:
        print("\n⚠️  Некоторые датасеты не прошли обработку.")


if __name__ == "__main__":
    main()
