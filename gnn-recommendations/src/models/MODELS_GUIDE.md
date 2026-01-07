# Руководство по моделям рекомендательных систем

## 📋 Обзор

В проекте реализовано **7 моделей** для рекомендательных систем:
- **6 baseline моделей** для сравнения
- **1 ваша модель** (GroupShuffleGNN)

Все модели наследуются от `BaseRecommender` и имеют **единый интерфейс**.

---

## 🎯 Быстрый старт

### Как использовать любую модель

```python
from models import LightGCN  # или любая другая модель
from data import RecommendationDataset

# 1. Загружаем данные
dataset = RecommendationDataset(name="movie_lens")
dataset.load_processed_data()
adj_matrix = dataset.get_torch_adjacency(normalized=True)

# 2. Создаем модель
model = LightGCN(
    n_users=dataset.n_users,
    n_items=dataset.n_items,
    embedding_dim=64,
    n_layers=3
)

# 3. Forward pass
user_emb, item_emb = model(adj_matrix)

# 4. Предсказание
scores = model.predict(users, items, adj_matrix)
```

**Все модели работают одинаково!** Различия только в параметрах инициализации.

---

## 🏗️ Архитектура моделей

### Общий принцип работы

```
┌─────────────────────────────────────────────────────────┐
│              ВХОДНЫЕ ДАННЫЕ                             │
├─────────────────────────────────────────────────────────┤
│  - n_users: количество пользователей                    │
│  - n_items: количество айтемов                          │
│  - adj_matrix: normalized adjacency matrix [N, N]       │
│    где N = n_users + n_items                            │
└─────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │   НАЧАЛЬНЫЕ EMBEDDINGS         │
        │   user_embedding: [n_users, d] │
        │   item_embedding: [n_items, d] │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │   МОДЕЛЬ-СПЕЦИФИЧНАЯ          │
        │   ОБРАБОТКА                    │
        │   (графовая свертка,           │
        │    трансформации, и т.д.)      │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │   ФИНАЛЬНЫЕ EMBEDDINGS        │
        │   user_emb: [n_users, d]      │
        │   item_emb: [n_items, d]      │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │   ПРЕДСКАЗАНИЕ SCORES         │
        │   score = user_emb @ item_emb │
        └───────────────────────────────┘
```

---

## 📊 Детальное описание моделей

### 1. BPR-MF (Bayesian Personalized Ranking - Matrix Factorization)

**Файл:** `baselines/bpr_mf.py`

#### Архитектура

```
Начальные embeddings
    ↓
[user_embedding] [item_embedding]
    ↓              ↓
    └──────┬───────┘
           ↓
    Скалярное произведение
           ↓
    scores = user_emb · item_emb
```

#### Особенности

- **Самая простая модель** - только embeddings, без графовой структуры
- **Не использует adj_matrix** - работает только с embeddings
- **Быстрая** - минимум вычислений
- **Базовый baseline** - для сравнения с более сложными методами

#### Формула

```
score(u, i) = user_emb[u] · item_emb[i]
```

#### Параметры

- `embedding_dim`: размерность embeddings (по умолчанию 64)
- `init_scale`: масштаб инициализации (по умолчанию 0.01)

#### Использование

```python
from models import BPR_MF

model = BPR_MF(n_users=1000, n_items=2000, embedding_dim=64)

# Forward (adj_matrix не используется)
user_emb, item_emb = model()  # или model(None)

# Предсказание
scores = model.predict(users, items)
```

---

### 2. LightGCN (Light Graph Convolutional Network)

**Файл:** `baselines/lightgcn.py`

#### Архитектура

```
Начальные embeddings
    ↓
x₀ = [user_emb, item_emb]
    ↓
Графовая свертка (слой 1)
    ↓
x₁ = A @ x₀
    ↓
Графовая свертка (слой 2)
    ↓
x₂ = A @ x₁
    ↓
... (n_layers раз)
    ↓
Layer Aggregation
    ↓
x_final = mean([x₀, x₁, x₂, ..., xₙ])
```

#### Особенности

- **Упрощенный GCN** - убраны нелинейности и веса
- **Только графовая свертка** - A @ x на каждом слое
- **Layer aggregation** - среднее всех слоев
- **Очень эффективный** - один из лучших baseline методов

#### Формула

```
x^(l+1) = A @ x^(l)
x_final = mean([x^(0), x^(1), ..., x^(L)])
```

где:
- `A` - normalized adjacency matrix
- `x^(l)` - embeddings на слое l
- `L` - количество слоев

#### Параметры

- `embedding_dim`: размерность embeddings (64)
- `n_layers`: количество слоев (3)
- `init_scale`: масштаб инициализации (0.01)

#### Использование

```python
from models import LightGCN

model = LightGCN(
    n_users=1000,
    n_items=2000,
    embedding_dim=64,
    n_layers=3
)

# Forward (требует adj_matrix)
user_emb, item_emb = model(adj_matrix)

# Предсказание
scores = model.predict(users, items, adj_matrix)
```

---

### 3. GCNII (Graph Convolutional Network with Initial residual and Identity mapping)

**Файл:** `baselines/gcnii.py`

#### Архитектура

```
Начальные embeddings
    ↓
x₀ = [user_emb, item_emb]
    ↓
Слой 1:
  x_conv = A @ x₀
  x_transformed = x_conv @ W₁
  x₁ = (1-α) · x_transformed + α · x₀  (identity mapping)
    ↓
Слой 2:
  x_conv = A @ x₁
  x_transformed = x_conv @ W₂
  x₂ = (1-α) · x_transformed + α · x₀  (identity mapping)
  x₂ = (1-β) · x₂ + β · x₁  (residual connection)
    ↓
... (n_layers раз)
```

#### Особенности

- **Residual connections** - связь между соседними слоями
- **Identity mapping** - связь с начальными embeddings
- **Борьба с over-smoothing** - сохранение информации из начальных слоев
- **Веса на каждом слое** - линейные преобразования

#### Формула

```
x^(l+1) = (1 - α) · (A @ x^(l) @ W^(l)) + α · x^(0)  (identity)
x^(l+1) = (1 - β) · x^(l+1) + β · x^(l)  (residual, если l > 0)
```

где:
- `α` - коэффициент identity mapping (0.1)
- `β` - коэффициент residual connection (0.5)
- `W^(l)` - веса слоя l

#### Параметры

- `embedding_dim`: размерность embeddings (64)
- `n_layers`: количество слоев (3)
- `alpha`: коэффициент identity mapping (0.1)
- `beta`: коэффициент residual connection (0.5)
- `dropout`: вероятность dropout (0.0)

#### Использование

```python
from models import GCNII

model = GCNII(
    n_users=1000,
    n_items=2000,
    embedding_dim=64,
    n_layers=3,
    alpha=0.1,
    beta=0.5
)

user_emb, item_emb = model(adj_matrix)
scores = model.predict(users, items, adj_matrix)
```

---

### 4. DGR (Desmoothing Graph Representation)

**Файл:** `baselines/dgr.py`

#### Архитектура

```
Начальные embeddings
    ↓
x₀ = [user_emb, item_emb]
    ↓
Слой 1:
  x_conv = A @ x₀
  x_transformed = x_conv @ W₁
  x₁ = (1-λ) · x_transformed + λ · x₀  (desmoothing)
    ↓
Слой 2:
  x_conv = A @ x₁
  x_transformed = x_conv @ W₂
  x₂ = (1-λ) · x_transformed + λ · x₁  (desmoothing)
    ↓
... (n_layers раз)
```

#### Особенности

- **Desmoothing framework** - предотвращение over-smoothing
- **Регуляризация** - комбинация текущего и предыдущего слоя
- **Современный метод** (2024)
- **Эффективен для глубоких сетей**

#### Формула

```
x^(l+1) = (1 - λ) · (A @ x^(l) @ W^(l)) + λ · x^(l)
```

где:
- `λ` - коэффициент регуляризации (0.1)

#### Параметры

- `embedding_dim`: размерность embeddings (64)
- `n_layers`: количество слоев (3)
- `lambda_reg`: коэффициент регуляризации (0.1)
- `dropout`: вероятность dropout (0.0)

#### Использование

```python
from models import DGR

model = DGR(
    n_users=1000,
    n_items=2000,
    embedding_dim=64,
    n_layers=3,
    lambda_reg=0.1
)

user_emb, item_emb = model(adj_matrix)
scores = model.predict(users, items, adj_matrix)
```

---

### 5. SVD-GCN (SVD Graph Convolutional Network)

**Файл:** `baselines/svd_gcn.py`

#### Архитектура

```
Начальные embeddings
    ↓
SVD декомпозиция A:
  A ≈ U @ S @ V^T
  (низкоранговое приближение)
    ↓
Слой 1:
  x_conv = U @ (S · (V^T @ x₀))  (эффективная свертка)
  x₁ = x_conv @ W₁
    ↓
Слой 2:
  x_conv = U @ (S · (V^T @ x₁))
  x₂ = x_conv @ W₂
    ↓
... (n_layers раз)
```

#### Особенности

- **SVD декомпозиция** - низкоранговое приближение adjacency matrix
- **Эффективность** - меньше вычислений для больших графов
- **Спектральный подход** - работа в пространстве собственных векторов
- **Ранг контролируется** - параметр `rank`

#### Формула

```
A ≈ U @ diag(S) @ V^T  (SVD, rank = k)
x_conv = U @ (S · (V^T @ x))
x^(l+1) = x_conv @ W^(l)
```

где:
- `U, S, V` - SVD компоненты
- `k` - ранг приближения (rank)

#### Параметры

- `embedding_dim`: размерность embeddings (64)
- `n_layers`: количество слоев (3)
- `rank`: ранг SVD (64, обычно = embedding_dim)
- `dropout`: вероятность dropout (0.0)

#### Использование

```python
from models import SVD_GCN

model = SVD_GCN(
    n_users=1000,
    n_items=2000,
    embedding_dim=64,
    n_layers=3,
    rank=32  # низкоранговое приближение
)

user_emb, item_emb = model(adj_matrix)
scores = model.predict(users, items, adj_matrix)
```

---

### 6. LayerGCN (Layer-wise Graph Convolutional Network)

**Файл:** `baselines/layergcn.py`

#### Архитектура

```
Начальные embeddings
    ↓
x₀ = [user_emb, item_emb]
    ↓
Слой 1:
  x_conv = A @ x₀
  x_transformed = x_conv @ W₁
  x₁ = α · x_transformed + (1-α) · x₀  (layer-wise refinement)
    ↓
Слой 2:
  x_conv = A @ x₁
  x_transformed = x_conv @ W₂
  prev_avg = mean([x₀, x₁])  (среднее предыдущих слоев)
  x₂ = α · x_transformed + (1-α) · prev_avg
    ↓
... (n_layers раз)
    ↓
Финальное: mean([x₀, x₁, ..., xₙ])
```

#### Особенности

- **Layer-wise refinement** - постепенное уточнение представлений
- **Использование предыдущих слоев** - комбинация с средним предыдущих
- **Финальная агрегация** - среднее всех слоев
- **Эффективен для глубоких сетей**

#### Формула

```
x^(l+1) = α · (A @ x^(l) @ W^(l)) + (1-α) · mean([x^(0), ..., x^(l)])
x_final = mean([x^(0), x^(1), ..., x^(L)])
```

где:
- `α` - коэффициент layer-wise refinement (0.5)

#### Параметры

- `embedding_dim`: размерность embeddings (64)
- `n_layers`: количество слоев (3)
- `alpha`: коэффициент layer-wise refinement (0.5)
- `dropout`: вероятность dropout (0.0)

#### Использование

```python
from models import LayerGCN

model = LayerGCN(
    n_users=1000,
    n_items=2000,
    embedding_dim=64,
    n_layers=3,
    alpha=0.5
)

user_emb, item_emb = model(adj_matrix)
scores = model.predict(users, items, adj_matrix)
```

---

### 7. GroupShuffleGNN ⭐ (Ваша модель)

**Файл:** `group_shuffle/model.py`

#### Архитектура

```
Начальные embeddings
    ↓
x₀ = [user_emb, item_emb]
    ↓
GroupShuffleLayer 1:
  1. Graph convolution: x_conv = A @ x₀
  2. Group (ортогональное преобразование):
     W_orth = block_diag([exp(A_skew_1), ..., exp(A_skew_k)])
     x_transformed = x_conv @ W_orth
  3. Shuffle: x_shuffled = x_transformed[:, perm]
  4. Residual: x₁ = (1-α) · x_shuffled + α · x₀
    ↓
GroupShuffleLayer 2:
  (аналогично)
    ↓
... (n_layers раз)
    ↓
Layer Aggregation: mean([x₀, x₁, ..., xₙ])
```

#### Особенности

- **Ортогональное преобразование** - через exponential map (Lie group)
- **Блочная структура** - эффективная параметризация O(p×n) вместо O(n²)
- **Shuffle** - перестановка признаков
- **Residual connections** - сохранение информации
- **Layer aggregation** - объединение всех слоев

#### Формула

```
x_conv = A @ x
W_orth = block_diag([exp(A_skew_1), ..., exp(A_skew_k)])  (ортогональная)
x_transformed = x_conv @ W_orth
x_shuffled = x_transformed[:, perm]
x^(l+1) = (1-α) · x_shuffled + α · x^(0)  (residual)
x_final = mean([x^(0), x^(1), ..., x^(L)])
```

где:
- `A_skew_i` - skew-symmetric матрицы для блока i
- `exp(A_skew)` - exponential map (гарантирует ортогональность)
- `perm` - перестановка для shuffle
- `α` - коэффициент residual (0.1)

#### Параметры

- `embedding_dim`: размерность embeddings (64, должна делиться на block_size)
- `n_layers`: количество слоев (3)
- `block_size`: размер блока для ортогональной матрицы (8)
- `residual_alpha`: коэффициент residual (0.1)
- `dropout`: вероятность dropout (0.0)

#### Использование

```python
from models import GroupShuffleGNN

model = GroupShuffleGNN(
    n_users=1000,
    n_items=2000,
    embedding_dim=64,
    n_layers=3,
    block_size=8,
    residual_alpha=0.1
)

user_emb, item_emb = model(adj_matrix)
scores = model.predict(users, items, adj_matrix)

# Проверка ортогональности
errors = model.get_orthogonality_errors()
```

---

## 🔄 Как все используется в системе

### Полный pipeline работы

```
┌─────────────────────────────────────────────────────────┐
│  ЭТАП 1: Подготовка данных                              │
├─────────────────────────────────────────────────────────┤
│  python scripts/prepare_data.py --dataset movie_lens   │
│    ↓                                                     │
│  - Загрузка данных (через loaders)                      │
│  - Препроцессинг (фильтрация, бинаризация)              │
│  - Разделение на train/valid/test                       │
│  - Построение графов (bipartite graph)                  │
│    ↓                                                     │
│  Результат:                                              │
│  - data/processed/{dataset}/train.txt, valid.txt, ...    │
│  - data/graphs/{dataset}/norm_adj_matrix.npz            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  ЭТАП 2: Инициализация модели                           │
├─────────────────────────────────────────────────────────┤
│  from models import LightGCN                             │
│  from data import RecommendationDataset                 │
│                                                          │
│  # Загружаем обработанные данные                        │
│  dataset = RecommendationDataset(name="movie_lens")     │
│  dataset.load_processed_data()                           │
│  adj_matrix = dataset.get_torch_adjacency()              │
│    # adj_matrix: [N, N] sparse tensor                   │
│    # N = n_users + n_items                              │
│                                                          │
│  # Создаем модель                                        │
│  model = LightGCN(                                       │
│      n_users=dataset.n_users,                            │
│      n_items=dataset.n_items,                             │
│      embedding_dim=64,                                   │
│      n_layers=3                                          │
│  )                                                       │
│    ↓                                                     │
│  # Инициализация embeddings:                            │
│  # user_embedding: [n_users, 64]                        │
│  # item_embedding: [n_items, 64]                        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  ЭТАП 3: Обучение модели                                │
├─────────────────────────────────────────────────────────┤
│  trainer = Trainer(model, dataset, config)              │
│  trainer.train()                                         │
│    ↓                                                     │
│  Для каждой эпохи:                                       │
│    1. Сэмплирование батчей:                              │
│       - user: [batch_size]                               │
│       - pos_item: [batch_size] (из train)                │
│       - neg_item: [batch_size] (случайные)               │
│                                                          │
│    2. Forward pass:                                      │
│       user_emb, item_emb = model(adj_matrix)            │
│       # user_emb: [n_users, 64]                         │
│       # item_emb: [n_items, 64]                         │
│                                                          │
│    3. Вычисление scores:                                │
│       pos_scores = (user_emb[users] * item_emb[pos_items]).sum(1)
│       neg_scores = (user_emb[users] * item_emb[neg_items]).sum(1)
│                                                          │
│    4. BPR Loss:                                         │
│       loss = -log(σ(pos_score - neg_score))             │
│                                                          │
│    5. Backward + оптимизация:                           │
│       loss.backward()                                    │
│       optimizer.step()                                   │
│                                                          │
│    6. Валидация (каждые N эпох):                        │
│       metrics = evaluator.evaluate(model, valid_data)   │
│                                                          │
│    7. Early stopping:                                   │
│       если метрика не улучшается → остановка            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  ЭТАП 4: Оценка модели                                  │
├─────────────────────────────────────────────────────────┤
│  evaluator = Evaluator()                                 │
│  metrics = evaluator.evaluate(model, dataset.test_data) │
│    ↓                                                     │
│  Процесс оценки:                                         │
│    1. Получить все embeddings:                          │
│       user_emb, item_emb = model.get_all_embeddings(adj)│
│                                                          │
│    2. Вычислить scores для всех пар:                    │
│       scores = user_emb @ item_emb.T  # [n_users, n_items]
│                                                          │
│    3. Для каждого пользователя:                         │
│       - Взять топ-K айтемов (по scores)                 │
│       - Сравнить с ground truth (test_data)              │
│                                                          │
│    4. Вычислить метрики:                                │
│       - Recall@K = hits / total_items                   │
│       - NDCG@K = normalized discounted cumulative gain  │
│       - Coverage = unique_items_recommended / n_items   │
└─────────────────────────────────────────────────────────┘
```

### Детальный поток работы модели

#### Forward pass (пример LightGCN)

```
Вход: adj_matrix [N, N], где N = n_users + n_items
      ↓
1. Начальные embeddings:
   x₀ = [user_embedding.weight, item_embedding.weight]
        [n_users, d] + [n_items, d] = [N, d]
      ↓
2. Графовая свертка (слой 1):
   x₁ = adj_matrix @ x₀
       [N, N] @ [N, d] = [N, d]
      ↓
3. Графовая свертка (слой 2):
   x₂ = adj_matrix @ x₁
       [N, N] @ [N, d] = [N, d]
      ↓
4. Графовая свертка (слой 3):
   x₃ = adj_matrix @ x₂
       [N, N] @ [N, d] = [N, d]
      ↓
5. Layer aggregation:
   x_final = mean([x₀, x₁, x₂, x₃])
            [N, d]
      ↓
6. Разделение:
   user_emb, item_emb = split(x_final, [n_users, n_items])
   user_emb: [n_users, d]
   item_emb: [n_items, d]
```

#### Предсказание scores

```
Вход: users [batch_size], items [batch_size]
      ↓
1. Получить embeddings:
   user_emb_all = model.get_all_embeddings(adj_matrix)[0]
                  [n_users, d]
   item_emb_all = model.get_all_embeddings(adj_matrix)[1]
                  [n_items, d]
      ↓
2. Выбрать нужные embeddings:
   user_emb_selected = user_emb_all[users]
                       [batch_size, d]
   item_emb_selected = item_emb_all[items]
                       [batch_size, d]
      ↓
3. Вычислить scores:
   scores = (user_emb_selected * item_emb_selected).sum(dim=1)
            [batch_size, d] * [batch_size, d] = [batch_size]
```

#### Обучение (BPR Loss)

```
Вход: user [batch_size], pos_item [batch_size], neg_item [batch_size]
      ↓
1. Forward pass:
   user_emb, item_emb = model(adj_matrix)
      ↓
2. Вычислить scores:
   pos_scores = (user_emb[user] * item_emb[pos_item]).sum(1)
                [batch_size]
   neg_scores = (user_emb[user] * item_emb[neg_item]).sum(1)
                [batch_size]
      ↓
3. BPR Loss:
   loss = -log(σ(pos_scores - neg_scores))
         где σ(x) = 1 / (1 + exp(-x))
      ↓
4. Backward:
   loss.backward()
   optimizer.step()
```

---

## 💻 Примеры использования

### Пример 1: Обучение одной модели

```python
import torch
from models import LightGCN
from data import RecommendationDataset

# 1. Загружаем данные
dataset = RecommendationDataset(name="movie_lens")
dataset.load_processed_data()
adj_matrix = dataset.get_torch_adjacency(normalized=True)

# 2. Создаем модель
model = LightGCN(
    n_users=dataset.n_users,
    n_items=dataset.n_items,
    embedding_dim=64,
    n_layers=3
)

# 3. Forward pass
user_emb, item_emb = model(adj_matrix)
print(f"User embeddings: {user_emb.shape}")
print(f"Item embeddings: {item_emb.shape}")

# 4. Предсказание
users = torch.tensor([0, 1, 2])
items = torch.tensor([0, 1, 2])
scores = model.predict(users, items, adj_matrix)
print(f"Scores: {scores}")
```

### Пример 2: Сравнение всех моделей

```python
from models import (
    BPR_MF, LightGCN, GCNII, DGR, SVD_GCN, LayerGCN, GroupShuffleGNN
)
from data import RecommendationDataset

# Загружаем данные
dataset = RecommendationDataset(name="movie_lens")
dataset.load_processed_data()
adj_matrix = dataset.get_torch_adjacency(normalized=True)

# Список всех моделей
models = {
    'BPR-MF': BPR_MF(dataset.n_users, dataset.n_items, 64),
    'LightGCN': LightGCN(dataset.n_users, dataset.n_items, 64, n_layers=3),
    'GCNII': GCNII(dataset.n_users, dataset.n_items, 64, n_layers=3),
    'DGR': DGR(dataset.n_users, dataset.n_items, 64, n_layers=3),
    'SVD-GCN': SVD_GCN(dataset.n_users, dataset.n_items, 64, n_layers=3),
    'LayerGCN': LayerGCN(dataset.n_users, dataset.n_items, 64, n_layers=3),
    'GroupShuffleGNN': GroupShuffleGNN(
        dataset.n_users, dataset.n_items, 64, n_layers=3, block_size=8
    ),
}

# Тестируем каждую модель
for name, model in models.items():
    print(f"\n{name}:")
    print(f"  Параметров: {model.get_parameters_count():,}")
    
    # Forward pass
    try:
        if name == 'BPR-MF':
            user_emb, item_emb = model()
        else:
            user_emb, item_emb = model(adj_matrix)
        print(f"  ✅ Forward pass успешен")
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
```

### Пример 3: Использование с конфигурацией

```python
import yaml
from models import LightGCN
from data import RecommendationDataset

# Загружаем конфигурацию
with open('config/models/lightgcn.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Загружаем данные
dataset = RecommendationDataset(name="movie_lens")
dataset.load_processed_data()
adj_matrix = dataset.get_torch_adjacency(normalized=True)

# Создаем модель из конфигурации
model_config = config['model']
model = LightGCN(
    n_users=dataset.n_users,
    n_items=dataset.n_items,
    **model_config
)

print(f"Модель создана с параметрами: {model_config}")
```

---

## 📊 Сравнительная таблица моделей

| Модель | Сложность | Граф | Residual | Layer Agg | Особенность |
|--------|-----------|------|----------|-----------|-------------|
| **BPR-MF** | ⭐ | ❌ | ❌ | ❌ | Только embeddings |
| **LightGCN** | ⭐⭐ | ✅ | ❌ | ✅ | Упрощенный GCN |
| **GCNII** | ⭐⭐ | ✅ | ✅ | ❌ | Identity + Residual |
| **DGR** | ⭐⭐⭐ | ✅ | ✅ | ❌ | Desmoothing |
| **SVD-GCN** | ⭐⭐⭐ | ✅ | ❌ | ❌ | SVD декомпозиция |
| **LayerGCN** | ⭐⭐⭐⭐ | ✅ | ✅ | ✅ | Layer-wise refinement |
| **GroupShuffleGNN** ⭐ | ⭐⭐⭐ | ✅ | ✅ | ✅ | Ортогональное преобразование |

---

## 🔍 Ключевые различия

### 1. Использование графа

- **BPR-MF**: Не использует граф (только embeddings)
- **Остальные**: Используют графовую структуру (adj_matrix)

### 2. Residual connections

- **BPR-MF, LightGCN, SVD-GCN**: Нет residual
- **GCNII, DGR, LayerGCN, GroupShuffleGNN**: Есть residual

### 3. Layer aggregation

- **BPR-MF, GCNII, DGR, SVD-GCN**: Нет layer aggregation
- **LightGCN, LayerGCN, GroupShuffleGNN**: Есть layer aggregation

### 4. Специальные техники

- **GCNII**: Identity mapping
- **DGR**: Desmoothing regularization
- **SVD-GCN**: SVD декомпозиция
- **LayerGCN**: Layer-wise refinement
- **GroupShuffleGNN**: Ортогональное преобразование + Shuffle

---

## 🎯 Когда использовать какую модель

### Для быстрого baseline
- **BPR-MF** - самая простая и быстрая

### Для лучшего качества
- **LightGCN** - один из лучших baseline методов

### Для глубоких сетей (много слоев)
- **GCNII** - residual connections
- **DGR** - desmoothing
- **LayerGCN** - layer-wise refinement
- **GroupShuffleGNN** - ортогональность

### Для больших графов
- **SVD-GCN** - эффективная декомпозиция

### Для вашего исследования
- **GroupShuffleGNN** - ваша модель с ортогональными преобразованиями

---

## 🚀 Полный цикл использования (от данных до результатов)

### Шаг 1: Подготовка данных (один раз)

```bash
# Подготовка датасета
python scripts/prepare_data.py --dataset movie_lens
python scripts/prepare_data.py --dataset book_crossing
python scripts/prepare_data.py --dataset gowalla
```

**Результат:**
- Обработанные данные в `data/processed/{dataset}/`
- Графы в `data/graphs/{dataset}/`

### Шаг 2: Обучение всех моделей

```python
# scripts/run_experiments.py (будет создан позже)

from models import (
    BPR_MF, LightGCN, GCNII, DGR, SVD_GCN, LayerGCN, GroupShuffleGNN
)
from data import RecommendationDataset
from training import Trainer
from evaluation import Evaluator

# Список всех моделей для тестирования
models_to_test = {
    'BPR-MF': BPR_MF,
    'LightGCN': LightGCN,
    'GCNII': GCNII,
    'DGR': DGR,
    'SVD-GCN': SVD_GCN,
    'LayerGCN': LayerGCN,
    'GroupShuffleGNN': GroupShuffleGNN,
}

# Для каждого датасета
for dataset_name in ['movie_lens', 'book_crossing', 'gowalla']:
    # Загружаем данные
    dataset = RecommendationDataset(name=dataset_name)
    dataset.load_processed_data()
    adj_matrix = dataset.get_torch_adjacency(normalized=True)
    
    # Для каждой модели
    for model_name, ModelClass in models_to_test.items():
        # Создаем модель
        model = ModelClass(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embedding_dim=64,
            n_layers=3
        )
        
        # Обучаем
        trainer = Trainer(model, dataset, config)
        trainer.train()
        
        # Оцениваем
        evaluator = Evaluator()
        metrics = evaluator.evaluate(model, dataset.test_data)
        
        # Сохраняем результаты
        save_results(model_name, dataset_name, metrics)
```

### Шаг 3: Сравнение результатов

```python
# Анализ результатов
results = load_all_results()

# Сравнение по метрикам
for metric in ['Recall@10', 'NDCG@10']:
    print(f"\n{metric}:")
    for model_name in models_to_test.keys():
        mean_score = results[model_name][metric]['mean']
        std_score = results[model_name][metric]['std']
        print(f"  {model_name:20s} {mean_score:.4f} ± {std_score:.4f}")

# Статистические тесты
perform_statistical_tests(results)
```

---

## 📊 Структура экспериментов

### Эксперимент 1: Сравнение всех моделей

```
Для каждого датасета:
  Для каждой модели:
    Для каждого seed (5 раз):
      1. Инициализация модели
      2. Обучение
      3. Оценка на test set
      4. Сохранение метрик
    
    Агрегация результатов (mean ± std)
```

### Эксперимент 2: Depth Analysis

```
Для GroupShuffleGNN:
  Для n_layers в [2, 4, 8, 16]:
    Обучение и оценка
    Анализ over-smoothing метрик
```

### Эксперимент 3: Ablation Studies

```
Для GroupShuffleGNN:
  - Без residual connections
  - Без shuffle
  - Без layer aggregation
  - Разные block_size
```

---

## 🔧 Технические детали

### Единый интерфейс всех моделей

Все модели наследуются от `BaseRecommender` и имеют одинаковые методы:

```python
class BaseRecommender:
    def forward(adj_matrix) -> (user_emb, item_emb)
    def predict(users, items, adj_matrix) -> scores
    def get_all_embeddings(adj_matrix) -> (user_emb, item_emb)
    def get_parameters_count() -> int
    def reset_parameters()
```

### Различия в использовании

| Модель | Требует adj_matrix? | Особенности |
|--------|---------------------|-------------|
| **BPR-MF** | ❌ Нет | Может работать без графа |
| **Остальные** | ✅ Да | Требуют граф для forward pass |

### Обработка sparse матриц

Все модели поддерживают как sparse, так и dense adjacency matrices:

```python
# Sparse (рекомендуется для больших графов)
adj_matrix = dataset.get_torch_adjacency(normalized=True)  # sparse

# Dense (для маленьких графов)
adj_matrix = adj_matrix.to_dense()  # если нужно
```

---

## 📈 Ожидаемые результаты

### По сложности моделей

1. **BPR-MF**: Базовый уровень (Recall@10 ≈ 0.045-0.052)
2. **LightGCN**: Сильный baseline (Recall@10 ≈ 0.058-0.065)
3. **GCNII, DGR, SVD-GCN**: Средний уровень (Recall@10 ≈ 0.060-0.068)
4. **LayerGCN**: Высокий уровень (Recall@10 ≈ 0.070-0.076)
5. **GroupShuffleGNN**: Ваша модель (ожидается лучше или на уровне LayerGCN)

### По борьбе с over-smoothing

- **BPR-MF, LightGCN**: Нет специальных техник
- **GCNII**: Residual connections
- **DGR**: Desmoothing regularization
- **LayerGCN**: Layer-wise refinement
- **GroupShuffleGNN**: Ортогональные преобразования (новая техника)

---

## ✅ Итог

Все модели имеют **единый интерфейс** и могут использоваться одинаково:

1. **Создание**: `model = ModelClass(n_users, n_items, ...)`
2. **Forward**: `user_emb, item_emb = model(adj_matrix)`
3. **Предсказание**: `scores = model.predict(users, items, adj_matrix)`
4. **Embeddings**: `user_emb, item_emb = model.get_all_embeddings(adj_matrix)`

**Различия только в внутренней архитектуре и параметрах**, но интерфейс одинаковый для всех!

Это позволяет:
- ✅ Легко сравнивать модели
- ✅ Использовать один и тот же код обучения
- ✅ Использовать один и тот же код оценки
- ✅ Легко добавлять новые модели

---

## 📚 Дополнительные ресурсы

- **Код моделей**: `src/models/`
- **Конфигурации**: `config/models/`
- **Тесты**: `scripts/test_all_models.py`
- **Документация по данным**: `src/data/README.md`

