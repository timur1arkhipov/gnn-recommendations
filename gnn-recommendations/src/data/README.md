# Модуль подготовки данных

Этот модуль содержит все необходимые компоненты для подготовки данных рекомендательных систем.

## Компоненты

### 1. `preprocessing.py`
Функции для препроцессинга данных:
- `filter_by_min_interactions()` - фильтрация по минимальному количеству взаимодействий
- `binarize_interactions()` - бинаризация взаимодействий (implicit feedback)
- `normalize_ids()` - нормализация ID пользователей и айтемов
- `remove_duplicates()` - удаление дубликатов
- `get_statistics()` - вычисление статистики

### 2. `graph_builder.py`
Функции для построения графов:
- `build_bipartite_graph()` - построение bipartite графа
- `normalize_adjacency_matrix()` - нормализация adjacency matrix
- `convert_to_torch_sparse()` - преобразование в PyTorch sparse tensor
- `save_adjacency_matrix()` / `load_adjacency_matrix()` - сохранение/загрузка

### 3. `dataset.py`
Основной класс `RecommendationDataset` для работы с датасетами.

## Использование

### Базовое использование

```python
from data import RecommendationDataset

# Создаем объект датасета
dataset = RecommendationDataset(name="movie_lens", root_dir=".")

# Загружаем и обрабатываем данные
dataset.load_raw_data()
dataset.preprocess()
dataset.split()
dataset.build_graph()

# Получаем граф для обучения
adj_matrix = dataset.get_torch_adjacency(normalized=True)
```

### Использование скрипта

```bash
# Подготовка MovieLens
python scripts/prepare_data.py --dataset movie_lens

# Подготовка Amazon Books
python scripts/prepare_data.py --dataset amazon_books

# Подготовка Gowalla
python scripts/prepare_data.py --dataset gowalla
```

## Поддерживаемые датасеты

### MovieLens
- Формат: `ratings.csv` с колонками: `userId`, `movieId`, `rating`, `timestamp`
- Путь: `data/raw/movie_lens/ratings.csv`

### Amazon Books
- **Внимание**: Требуется файл с рейтингами пользователей
- Текущий формат: `Books_df.csv` (каталог книг)
- Нужно добавить файл с рейтингами в формате: `userId`, `itemId`, `rating`

### Gowalla
- Формат: `Gowalla_cleanCheckins.csv` или `Gowalla_totalCheckins.txt`
- Ожидаемые колонки: `userId`, `itemId` (locationId), `timestamp`

## Структура обработанных данных

После обработки данные сохраняются в:
- `data/processed/{dataset_name}/`:
  - `train.txt` - обучающая выборка
  - `valid.txt` - валидационная выборка
  - `test.txt` - тестовая выборка
  - `stats.json` - статистика

- `data/graphs/{dataset_name}/`:
  - `adj_matrix.npz` - adjacency matrix
  - `norm_adj_matrix.npz` - нормализованная adjacency matrix

## Примеры

### Загрузка уже обработанных данных

```python
dataset = RecommendationDataset(name="movie_lens")
dataset.load_processed_data()  # Загружает из файлов
adj_matrix = dataset.get_torch_adjacency()
```

### Кастомные параметры

```python
dataset = RecommendationDataset(name="movie_lens")

# Кастомная фильтрация
dataset.preprocess(
    min_user_interactions=20,
    min_item_interactions=20,
    rating_threshold=3.0
)

# Кастомное разделение
dataset.split(
    strategy='random',
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1
)
```

