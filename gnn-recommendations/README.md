# GNN-based Recommendation System

Система рекомендаций на основе графовых нейронных сетей с новым алгоритмом GroupShuffleGNN для борьбы с over-smoothing.

---

## Архитектура системы

```
┌─────────────────────────────────────────────────────────────┐
│                      Входные данные                         │
│    User-Item взаимодействия (ratings, check-ins)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Предобработка                             │
│  1. Фильтрация (min 10 взаимодействий)                     │
│  2. Train/Val/Test split (80/10/10)                         │
│  3. Построение bipartite графа User-Item                    │
│  4. Нормализация adjacency matrix: D^(-1/2) A D^(-1/2)     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    GNN модели                               │
│  - BPR-MF (baseline)                                        │
│  - LightGCN                                                 │
│  - GCNII                                                    │
│  - DGR                                                      │
│  - LayerGCN                                                 │
│  - GroupShuffleGNN (предложенный алгоритм)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Обучение                                  │
│  Loss: BPR (Bayesian Personalized Ranking)                 │
│  Optimizer: Adam (lr=0.0001)                               │
│  Techniques: gradient clipping, warmup, early stopping     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Оценка                                    │
│  Метрики качества + Over-smoothing метрики                 │
└─────────────────────────────────────────────────────────────┘
```

---

## GroupShuffleGNN: предложенный алгоритм

### Проблема

В глубоких GNN все node embeddings становятся слишком похожими (over-smoothing), что приводит к потере различимости и ухудшению качества рекомендаций.

### Решение: Group and Shuffle механизм

**Ключевая идея:** Разделить embeddings на группы, обрабатывать их независимо и перемешивать между слоями.

**Алгоритм:**

```
Input: E ∈ R^(N×D)  (embeddings для N узлов, D измерений)

Для каждого слоя l = 1...L:
    
    1. Split: Разделить E на G групп по размеру B
       [G₁, G₂, ..., G_G] = split(E, num_groups=G)
    
    2. Graph Convolution: Применить к каждой группе независимо
       G'ᵢ = σ(A_norm @ Gᵢ),  i = 1...G
    
    3. Shuffle: Перемешать группы случайной перестановкой π
       [G'_π(1), G'_π(2), ..., G'_π(G)] = shuffle([G'₁, G'₂, ..., G'_G])
    
    4. Residual: Добавить skip connection
       E_out = E + concat([G'_π(1), G'_π(2), ..., G'_π(G)])

Output: E_final
```

**Параметры:**
- `num_groups` (G): количество групп (по умолчанию 4)
- `block_size` (B): размер группы = D / G
- `n_layers` (L): количество слоёв (по умолчанию 4)

**Преимущества:**
- Разные группы развиваются независимо, предотвращая синхронизацию
- Shuffle нарушает симметрию между слоями
- Residual connections сохраняют начальную информацию
- Variance embeddings остаётся высокой даже в глубоких сетях

**Код реализации:** `src/models/group_shuffle/`

---

## Датасеты

### MovieLens-1M
- Пользователи: 6,040
- Items (фильмы): 3,706
- Рейтинги: 1,000,209
- Тип: Явные оценки (1-5 звёзд)
- Плотность: 4.47%

### Book-Crossing
- Пользователи: 92,107
- Items (книги): 270,170
- Рейтинги: 1,149,780
- Тип: Явные оценки (0-10)
- Плотность: 0.0046%

### Gowalla (опционально)
- Пользователи: 107,092
- Items (локации): 1,280,969
- Check-ins: 6,442,890
- Тип: Неявная обратная связь
- Предупреждение: Слишком большой для GPU с 8GB памяти

---

## Метрики

### Метрики качества рекомендаций

**Recall@K**
- Формула: hits / total_relevant
- Измеряет: Какую долю релевантных items нашли в топ-K

**NDCG@K** (Normalized Discounted Cumulative Gain)
- Формула: DCG / IDCG, где DCG = Σ(rel_i / log₂(i+1))
- Измеряет: Качество ранжирования с учётом позиций

**Precision@K**
- Формула: hits / K
- Измеряет: Точность в топ-K рекомендациях

**Coverage@K**
- Формула: |unique_items_in_recommendations| / |total_items|
- Измеряет: Разнообразие рекомендаций

**Gini Index@K**
- Формула: (2·Σ(i·xᵢ)) / (n·Σ(xᵢ)) - (n+1)/n
- Измеряет: Неравномерность распределения рекомендаций (0 = равномерно, 1 = неравномерно)

### Метрики Over-smoothing

**MCS** (Mean Cosine Similarity)
- Формула: mean(cos(emb_i, emb_j)) для всех пар i≠j
- Интерпретация: Высокое значение → сильный over-smoothing

**MAD** (Mean Average Distance)
- Формула: mean(||emb_i - emb_j||₂) для всех пар i≠j
- Интерпретация: Низкое значение → сильный over-smoothing

**Variance**
- Формула: mean(var(emb_dim)) по всем измерениям
- Интерпретация: Низкое значение → embeddings сжаты в узкую область

---

## Структура кода

```
gnn-recommendations/
│
├── src/
│   ├── data/              # Загрузка и предобработка данных
│   ├── models/            # Все модели (BPR-MF, GNN, GroupShuffleGNN)
│   ├── training/          # Trainer, losses, metrics
│   ├── evaluation/        # Evaluator, over-smoothing анализ
│   └── utils/             # Статистика, визуализация
│
├── config/                # YAML конфигурации
│   ├── training.yaml      # Глобальные параметры обучения
│   └── models/            # Конфиги для каждой модели
│
├── scripts/               # Скрипты для запуска
│   ├── check_gpu.py       # Проверка GPU
│   ├── prepare_data.py    # Подготовка датасетов
│   ├── run_all_experiments.py    # Один эксперимент
│   └── run_multiple_seeds.py     # Multiple seeds для статистики
│
├── data/                  # Данные
│   ├── raw/              # Исходные данные
│   ├── processed/        # Обработанные splits
│   └── graphs/           # Adjacency matrices
│
├── results/              # Результаты экспериментов
│   ├── checkpoints/      # Обученные модели
│   └── multiple_seeds/   # Агрегированные метрики
│
└── run_all.py           # Главный запускатор
```

---

## Обучение

### BPR Loss

```
L_BPR = -Σ log(σ(score_pos - score_neg)) + λ·||Θ||²

где:
  score_pos = <emb_user, emb_item_positive>
  score_neg = <emb_user, emb_item_negative>
  σ(x) = 1/(1+e^(-x))
```

### Гиперпараметры

- Learning rate: 0.0001
- Batch size: 2048
- Weight decay: 1e-5
- Epochs: 300
- Early stopping patience: 50
- Gradient clipping: max_norm = 1.0
- Warmup epochs: 5

---

## Запуск экспериментов

### Быстрый тест
```bash
python run_all.py --quick
```

Запустит 3 модели на MovieLens с 2 seeds.

### Полный цикл
```bash
python run_all.py
```

Запустит 6 моделей на 2 датасетах с 5 seeds.

### Результаты

```json
{
  "model_name": {
    "recall@10": {"mean": 0.152, "std": 0.003},
    "ndcg@10": {"mean": 0.071, "std": 0.002},
    "precision@10": {"mean": 0.030, "std": 0.001},
    "coverage@10": {"mean": 0.501, "std": 0.004},
    "gini@10": {"mean": 0.823, "std": 0.012}
  }
}
```

Сохраняются в `results/multiple_seeds/{dataset}_aggregated.json`

### Библиотеки
```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0
pyyaml>=6.0
tqdm>=4.65.0
```

Установка: `pip install -r requirements.txt`

## Лицензия

MIT License

