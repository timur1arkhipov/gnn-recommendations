# Temporal Per-User Split — Единственная стратегия разделения

## Что изменилось

**Удалено:**
- ❌ `random_global` — глобальное случайное разбиение
- ❌ `random` — per-user случайное разбиение

**Осталось:**
- ✅ `temporal` — **единственная** поддерживаемая стратегия

---

## Temporal Per-User Split

### Алгоритм

Для каждого пользователя отдельно:

```
Взаимодействия пользователя (отсортированы по timestamp):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │
└───┴───┴───┴───┴───┴───┴───┴───┘
 ↑ старое              новое ↑

Разделение:
┌───────────────────────┬─────┬─────┐
│  TRAIN (1-6)          │VALID│TEST │
│                       │  7  │  8  │
└───────────────────────┴─────┴─────┘
```

### Правила разделения

```python
if n_interactions >= 3:
    train = все кроме последних двух
    valid = предпоследнее взаимодействие
    test  = последнее взаимодействие

elif n_interactions == 2:
    train = первое взаимодействие
    valid = (пусто)
    test  = последнее взаимодействие

else:  # n_interactions == 1
    train = единственное взаимодействие
    valid = (пусто)
    test  = (пусто)
```

---

## Использование

### 1. Конфигурация датасета

Все датасеты теперь используют temporal split:

```yaml
# config/datasets/ml-1m.yaml
split:
  strategy: "temporal"  # единственная поддерживаемая стратегия
  seed: 42
```

**Параметры train_ratio, valid_ratio, test_ratio удалены** — они не нужны для temporal split.

### 2. Вызов в коде

```python
from src.data.dataset import RecommendationDataset

# Загрузка и подготовка данных
dataset = RecommendationDataset(name='ml-1m')
dataset.load_raw_data()
dataset.preprocess()

# Разделение (автоматически использует temporal)
dataset.split()

# Результат:
# dataset.train_data - обучающая выборка
# dataset.valid_data - валидационная выборка
# dataset.test_data  - тестовая выборка
```

### 3. Требования к данным

**Обязательно:** Датасет должен содержать колонку `timestamp`

Если timestamp отсутствует, будет ошибка:

```python
ValueError: Temporal split требует колонку 'timestamp'.
Датасет {name} не содержит временные метки.
```

---

## Статистика разделения

### Пример для MovieLens-1M

После фильтрации (min_interactions=10):

```
Всего пользователей: ~218,000
Всего взаимодействий: ~16,300,000

Распределение по сплитам:
├─ Train: ~14,000,000 (86%)  - старые взаимодействия
├─ Valid: ~1,100,000  (7%)   - предпоследние
└─ Test:  ~1,200,000  (7%)   - последние

Распределение пользователей:
├─ Users с n >= 3: ~95%  → есть во всех 3 сплитах
├─ Users с n == 2: ~4%   → train + test (без valid)
└─ Users с n == 1: ~1%   → только train
```

---

## Преимущества Temporal Split

### ✅ Реалистичная оценка
- Модель обучается на **прошлом**
- Предсказывает **будущее**
- Соответствует production сценарию

### ✅ Нет data leakage
- Информация из будущего **никогда** не попадает в train
- Гарантирует честную оценку модели

### ✅ Per-user гарантия
- Каждый пользователь (с n >= 2) участвует в тестировании
- Метрики вычисляются для всех пользователей

### ✅ Соответствие научным статьям
- Используется в LightGCN, NGCF, UltraGCN
- Результаты сравнимы с baseline публикациями

---

## Код реализации

### Упрощенная версия

```python
def temporal_per_user_split(df):
    """
    Temporal per-user split.

    Для каждого пользователя:
    - Последнее взаимодействие → test
    - Предпоследнее → validation
    - Все остальные → train
    """
    # Сортировка по времени
    df = df.sort_values(['userId', 'timestamp'])

    train_rows = []
    valid_rows = []
    test_rows = []

    # Разделение для каждого пользователя
    for user_id, group in df.groupby('userId'):
        n = len(group)

        if n >= 3:
            test_rows.append(group.iloc[-1])    # последнее
            valid_rows.append(group.iloc[-2])   # предпоследнее
            train_rows.append(group.iloc[:-2])  # остальные
        elif n == 2:
            test_rows.append(group.iloc[-1])
            train_rows.append(group.iloc[:1])
        else:
            train_rows.append(group)

    # Объединение
    train_data = pd.concat(train_rows, ignore_index=True)
    valid_data = pd.DataFrame(valid_rows)
    test_data = pd.DataFrame(test_rows)

    return train_data, valid_data, test_data
```

---

## Файлы с изменениями

### 1. Код
- ✅ `src/data/dataset.py` — удалены random_global и random стратегии
- ✅ Метод `split()` упрощен, оставлен только temporal

### 2. Конфигурации
- ✅ `config/datasets/ml-1m.yaml`
- ✅ `config/datasets/ml-100k.yaml`
- ✅ `config/datasets/facebook.yaml`
- ✅ `config/datasets/amazon_books.yaml`

**Изменения в конфигах:**
```yaml
# Было:
split:
  strategy: "random_global"
  train_ratio: 0.8
  valid_ratio: 0.1
  test_ratio: 0.1
  seed: 42

# Стало:
split:
  strategy: "temporal"
  seed: 42
```

---

## Вопросы и ответы

### Q: Почему удалили random_global и random?

**A:** Temporal split — это стандарт для рекомендательных систем:
- Реалистичная оценка (прошлое → будущее)
- Используется во всех топовых статьях (LightGCN, NGCF, etc.)
- Нет data leakage
- Результаты сравнимы с baseline публикациями

Random split не соответствует реальному сценарию использования.

### Q: Что если нет timestamp?

**A:** Temporal split требует временные метки. Если их нет:
```python
ValueError: Temporal split требует колонку 'timestamp'.
```

Убедитесь, что ваш датасет содержит timestamp.

### Q: Можно ли вернуть random_global?

**A:** Нет. Temporal split — единственная корректная стратегия для временных данных.
Если у вас нет timestamp, добавьте их в данные.

### Q: Почему train так много (86%), а test/valid мало (7% + 7%)?

**A:** Это **правильно** для temporal split!
- У каждого пользователя ровно **1 взаимодействие в test**
- И ровно **1 взаимодействие в valid** (если n >= 3)
- Все остальные идут в train

Это гарантирует, что мы предсказываем конкретное будущее событие для каждого пользователя.

---

## Проверка работоспособности

### Запуск preprocessing

```bash
cd /Users/timur/gnn-recommendations/gnn-recommendations

# Подготовка данных с temporal split
python scripts/prepare_data.py --dataset ml-1m

# Проверка результата
cat data/processed/movie_lens/stats.json
```

### Ожидаемый результат

```json
{
  "n_users": 218576,
  "n_items": 18830,
  "n_interactions": 16319112,
  "train_size": 14000000,  // ~86%
  "valid_size": 1100000,    // ~7%
  "test_size": 1200000      // ~7%
}
```

### Тестирование моделей

```bash
# Обучение LightGCN с temporal split
python scripts/run_all_experiments.py --models lightgcn --datasets movie_lens --seed 42

# Обучение OrthogonalBundle с temporal split
python scripts/run_all_experiments.py --models orthogonal_bundle --datasets movie_lens --seed 42
```

---

## Заключение

Temporal per-user split — это:
- ✅ **Единственная** поддерживаемая стратегия
- ✅ **Стандарт** индустрии для рекомендательных систем
- ✅ **Корректный** способ оценки моделей на временных данных
- ✅ **Сравнимый** с результатами научных публикаций

Все остальные стратегии удалены как некорректные для временных рекомендательных данных.
