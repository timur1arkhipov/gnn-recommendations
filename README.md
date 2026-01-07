# Дипломная работа: Применение метода Group and Shuffle к GNN-рекомендациям

## 📋 Обзор проекта

### Метод из статьи Gorbunov and Yudin "Group and Shuffle"

- **Суть**: Структурированная ортогональная параметризация с блочной структурой
- **Отличие от expRNN**: Меньше параметров (было O(n²) стало O(p×n))
- **Применение**: Fine-tuning, ортогональные свертки, 1-Lipschitz сети
- **Для моего применения**: Борьба с over-smoothing через сохранение геометрии

---

## 📊 Датасеты

- ✅ **Movielens** (MovieLens-1M)
- ✅ **Amazon-Book**
- ⚠️ **Нужен еще один** (предложены: Yelp2018, Gowalla)

**Вопрос для уточнения**: Какой третий датасет будет выбран? (Yelp2018 или Gowalla?)

---

## 🏗️ Архитектура модели

### GroupShuffleGCNLayer

Архитектура слоя включает:

1. **Агрегация соседей (GCN)** - стандартная операция графовой свертки
2. **Group & Shuffle ортогональное преобразование** - ключевой компонент метода
3. **Residual connections** - для сохранения информации из предыдущих слоев
4. **Layer aggregation** - объединение представлений из всех слоев

---

## 🎯 Новизна работы

1. ✅ **Первое применение Group and Shuffle к GNN-рекомендациям**
2. ✅ **Новая стратегия борьбы с over-smoothing**
3. ✅ **Теоретическое обоснование через изометрию**
4. ✅ **Эффективная параметризация**

---

## 🎯 Цели исследования

1. Теоретически обосновать связь ортогональности и over-smoothing
2. Разработать эффективную архитектуру
3. Провести эксперименты на 4 датасетах с 6+ baseline методами
4. **Depth analysis** (2, 4, 8, 16 слоев)
5. **Ablation studies** (анализ влияния компонентов)

---

## 📐 Методология

### Экспериментальная установка

- **Количество запусков**: 5 запусков с разными seeds
- **Статистические тесты**: t-test, p-values
- **Метрики качества рекомендаций**:
  - Recall@K (K=10, 20, 50)
  - NDCG@K (K=10, 20, 50)
  - Coverage
- **Метрики over-smoothing**:
  - Cosine Similarity (по слоям)
  - MAD (Mean Average Distance)
  - Embedding Variance

### Типы анализов

1. **Depth analysis** - анализ влияния глубины сети
2. **Ablation studies** - анализ влияния компонентов
3. **Efficiency analysis** - анализ эффективности
4. **Visualization** - визуализация результатов

gnn-recommendations/
│
├── config/                           # ⚙️ КОНФИГУРАЦИИ
│   ├── datasets/                     # Настройки датасетов
│   │   ├── movielens1m.yaml
│   │   ├── yelp2018.yaml
│   │   ├── amazon_book.yaml
│   │   └── gowalla.yaml
│   ├── models/                       # Настройки моделей
│   │   ├── bpr_mf.yaml              # Baseline 1
│   │   ├── lightgcn.yaml            # Baseline 2
│   │   ├── gcnii.yaml               # Baseline 3
│   │   ├── dgr.yaml                 # Baseline 4
│   │   ├── svd_gcn.yaml             # Baseline 5
│   │   ├── layergcn.yaml            # Baseline 6
│   │   └── groupshuffle_gnn.yaml    # ⭐ ВАШ АЛГОРИТМ
│   └── training.yaml                # Общие настройки обучения
│
├── data/                            # 💾 ДАННЫЕ
│   ├── raw/                         # Сырые данные (скачанные)
│   │   ├── movielens1m/
│   │   ├── yelp2018/
│   │   ├── amazon_book/
│   │   └── gowalla/
│   ├── processed/                   # Обработанные данные
│   │   └── {dataset_name}/
│   │       ├── train.txt
│   │       ├── valid.txt
│   │       ├── test.txt
│   │       └── stats.json
│   └── graphs/                      # Графы (adjacency matrices)
│       └── {dataset_name}/
│           ├── adj_matrix.npz
│           └── norm_adj_matrix.npz
│
├── src/                             # 💻 ИСХОДНЫЙ КОД
│   │
│   ├── data/                        # 📊 DATA PIPELINE
│   │   ├── __init__.py
│   │   ├── dataset.py               # ⚡ Базовый класс датасета
│   │   ├── preprocessing.py         # Препроцессинг
│   │   └── graph_builder.py         # Построение графов
│   │
│   ├── models/                      # 🧠 МОДЕЛИ
│   │   ├── __init__.py
│   │   ├── base.py                  # Базовый класс для всех моделей
│   │   │
│   │   ├── baselines/               # 📌 6 BASELINE МЕТОДОВ
│   │   │   ├── __init__.py
│   │   │   ├── bpr_mf.py           # Baseline 1: BPR-MF
│   │   │   ├── lightgcn.py         # Baseline 2: LightGCN
│   │   │   ├── gcnii.py            # Baseline 3: GCNII
│   │   │   ├── dgr.py              # Baseline 4: DGR
│   │   │   ├── svd_gcn.py          # Baseline 5: SVD-GCN
│   │   │   └── layergcn.py         # Baseline 6: LayerGCN
│   │   │
│   │   └── group_shuffle/           # ⭐ ВАШ АЛГОРИТМ
│   │       ├── __init__.py
│   │       ├── layers.py            # GroupShuffleLayer
│   │       ├── model.py             # GroupShuffleGNN (основная модель)
│   │       └── utils.py             # Вспомогательные функции
│   │
│   ├── training/                    # 🏋️ ОБУЧЕНИЕ
│   │   ├── __init__.py
│   │   ├── trainer.py               # Главный класс Trainer
│   │   ├── losses.py                # BPR Loss и другие
│   │   └── metrics.py               # Recall@K, NDCG@K, etc.
│   │
│   ├── evaluation/                  # 📈 ОЦЕНКА
│   │   ├── __init__.py
│   │   ├── evaluator.py             # Evaluator для метрик
│   │   └── oversmoothing.py         # Анализ over-smoothing
│   │
│   └── utils/                       # 🛠️ УТИЛИТЫ
│       ├── __init__.py
│       ├── logger.py                # Логирование
│       ├── visualization.py         # Визуализация
│       └── statistics.py            # Статистические тесты
│
├── scripts/                         # 🚀 ЗАПУСК ЭКСПЕРИМЕНТОВ
│   ├── run_experiments.py           # ⚡ ГЛАВНЫЙ СКРИПТ
│   ├── train_single_model.py        # Обучение одной модели
│   ├── evaluate_model.py            # Оценка модели
│   └── analyze_results.py           # Анализ результатов
│
├── experiments/                     # 🔬 ЭКСПЕРИМЕНТЫ
│   ├── depth_analysis/              # Анализ глубины (2,4,8,16 слоёв)
│   ├── ablations/                   # Ablation studies
│   ├── efficiency/                  # Анализ эффективности
│   └── oversmoothing/               # Анализ over-smoothing
│
├── notebooks/                       # 📓 JUPYTER NOTEBOOKS
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
├── results/                         # 📊 РЕЗУЛЬТАТЫ
│   ├── checkpoints/                 # Сохранённые модели
│   ├── logs/                        # Логи обучения
│   ├── metrics/                     # CSV с метриками
│   └── figures/                     # Графики и визуализации
│
├── tests/                           # ✅ ТЕСТЫ
│   ├── test_data.py                 # Тесты data pipeline
│   ├── test_models.py               # Тесты моделей
│   ├── test_training.py             # Тесты обучения
│   └── test_evaluation.py           # Тесты оценки
│
├── requirements.txt                 # Зависимости
├── setup.py                         # Установка пакета
└── README.md                        # Документация

---

## 📊 ЧАСТЬ 1: 6 Оптимальных Baseline Методов

### Рекомендуемый набор baseline методов

#### 1. BPR-MF ⭐⭐⭐ (обязательно)

- **Сложность**: ⭐ ЛЕГКО
- **Тип**: Традиционный CF baseline
- **Ожидаемый Recall@10**: 0.045-0.052
- **Описание**: Базовый метод матричной факторизации с BPR loss

#### 2. LightGCN ⭐⭐⭐⭐⭐ (САМЫЙ ВАЖНЫЙ!)

- **Сложность**: ⭐⭐ СРЕДНЕ
- **Тип**: Сильнейший GNN baseline
- **Ожидаемый Recall@10**: 0.058-0.065
- **Код**: https://github.com/gusye1234/LightGCN-PyTorch
- **Описание**: Упрощенная версия GCN без нелинейностей и весов

#### 3. GCNII ⭐⭐⭐⭐

- **Сложность**: ⭐⭐ СРЕДНЕ
- **Тип**: Классический anti-oversmoothing (residual)
- **Ожидаемый Recall@10**: 0.062-0.068
- **Код**: https://github.com/chennnM/GCNII
- **Описание**: GCN с residual connections и identity mapping

#### 4. DGR ⭐⭐⭐⭐

- **Сложность**: ⭐⭐⭐ СРЕДНЕ-СЛОЖНО
- **Тип**: Desmoothing framework (2024, свежий!)
- **Ожидаемый Recall@10**: 0.068-0.074
- **Код**: https://github.com/YuanchenBei/DGR
- **Описание**: Современный метод борьбы с over-smoothing

#### 5. SVD-GCN ⭐⭐⭐

- **Сложность**: ⭐⭐⭐ СРЕДНЕ-СЛОЖНО
- **Тип**: Спектральный подход
- **Ожидаемый Recall@10**: 0.060-0.066
- **Описание**: GCN с использованием SVD декомпозиции

#### 6. LayerGCN ⭐⭐⭐⭐

- **Сложность**: ⭐⭐⭐⭐ СЛОЖНО
- **Тип**: Layer-wise refinement
- **Ожидаемый Recall@10**: 0.070-0.076
- **Код**: https://github.com/enoche/LayerGCN
- **Описание**: Постепенное уточнение представлений по слоям

### Альтернативы

- Если SVD-GCN сложен → **NGCF**
- Если LayerGCN сложен → **AFDGCF**

---

## 💻 ЧАСТЬ 3: Архитектура Программы

### Структура проекта

```
gnn-recommendations/
├── config/                    # Конфигурации YAML
├── data/                      # Датасеты
│   ├── raw/
│   ├── processed/
│   └── graphs/
├── src/                       # Исходный код
│   ├── data/                  # Dataset, preprocessing
│   ├── models/                # Модели
│   │   ├── baselines/         # 6 baseline методов
│   │   └── group_shuffle/     # Ваша модель
│   ├── training/              # Trainer, losses, metrics
│   ├── evaluation/            # Evaluator, over-smoothing
│   └── utils/                 # Logger, visualization
├── scripts/                   # Запуск экспериментов
├── experiments/               # Depth analysis, ablations
├── notebooks/                 # Jupyter для анализа
├── results/                   # Checkpoints, logs, figures
└── tests/                     # Unit tests
```

---

### Ключевые компоненты

#### 1. Data Pipeline

```python
class RecommendationDataset:
    - load_raw_data()
    - preprocess()
    - split(strategy='temporal')
    - build_graph()
```

#### 2. Models

```python
class GroupShuffleGNN(BaseRecommender):
    - GroupShuffleLayer (ваш слой)
    - forward()
    - predict()
    - compute_loss()
```

#### 3. Training

```python
class Trainer:
    - train_epoch()
    - validate()
    - early_stopping
    - checkpoint management
```

#### 4. Evaluation

```python
class Evaluator:
    - Recall@K, NDCG@K, Coverage
    - Statistical tests
    
class OversmoothingAnalyzer:
    - Cosine similarity по слоям
    - MAD metrics
```

---

### Технологический стек

- **PyTorch** 2.0+
- **PyTorch Geometric** - для работы с графами
- **NumPy, SciPy, Pandas** - обработка данных
- **Matplotlib, Seaborn** - визуализация
- **TensorBoard / Weights & Biases** - логирование экспериментов

**Вопрос для уточнения**: Есть ли требования к конкретным версиям библиотек?

---

### Timeline реализации

| Этап | Время | Сложность |
|------|-------|-----------|
| Data pipeline | 1 неделя | ⭐⭐ |
| BPR-MF | 1 день | ⭐ |
| LightGCN | 3-4 дня | ⭐⭐ |
| GCNII | 3-4 дня | ⭐⭐ |
| GroupShuffleGNN | 2 недели | ⭐⭐⭐ |
| DGR, LayerGCN | 1 неделя | ⭐⭐⭐ |
| SVD-GCN | 5 дней | ⭐⭐⭐ |
| Training/Eval | 1 неделя | ⭐⭐ |
| Эксперименты | 2 недели | ⭐⭐ |
| **ИТОГО** | **~4 месяца** | |




---

## 🎯 ИТОГОВЫЕ ВЫВОДЫ

### Ответы на ключевые вопросы

#### 1. Какие 6 методов?

**Baseline методы**: BPR-MF, LightGCN, GCNII, DGR, SVD-GCN, LayerGCN

**Покрытие подходов**:
- Традиционный CF (BPR-MF)
- Простой GNN (LightGCN)
- Residual connections (GCNII)
- Desmoothing framework (DGR)
- Спектральный подход (SVD-GCN)
- Layer-wise refinement (LayerGCN)

#### 2. На какой уровень потянет?

- **МАГИСТРАТУРА** (отличная работа) — если реализовать как в руководстве
- **АСПИРАНТУРА** (начальный) — если добавить теорию с доказательствами

#### 3. Архитектура программы?

- Модульная структура с 10+ компонентами
- ~5000-7000 строк кода
- Полная воспроизводимость
- 4 месяца разработки

---

### Что делать дальше

1. ✅ Изучить все материалы (3 документа готовы)
2. ✅ Настроить окружение и структуру проекта
3. ✅ Начать с data pipeline и BPR-MF
4. ✅ Постепенно добавлять baseline методы
5. ✅ Реализовать вашу модель
6. ✅ Провести эксперименты

---

## 📝 Ссылки на дополнительные материалы

- `complete_implementation_guide.md`
- `research_guide_gorbunov_yudin_gnn_recommendations.md`
- `critical_analysis_summary.md`
- `baseline_methods_analysis.md`

---

## 💬 Feedback и вопросы
ты мне скинула файл complete_implementation_guide.md, мне он понравился, но нужно некоторое пояснение, можешь рассказать про сам поток работы этой программы, где будет мой алгоритм, где 6 методов baseline, которые мы обсуждали, где тесты и какие тесты обязательно, мне важно именно понять о структуре будующей программы
Отлично! Давайте разберём структуру программы и поток работы пошагово. Я объясню, где что находится и как всё взаимодействует.

---

## 🔄 ПОТОК РАБОТЫ ПРОГРАММЫ

### 📁 СТРУКТУРА ПРОЕКТА (детально)
gnn-recommendations/
│
├── config/                           # ⚙️ КОНФИГУРАЦИИ
│   ├── datasets/                     # Настройки датасетов
│   │   ├── movielens1m.yaml
│   │   ├── yelp2018.yaml
│   │   ├── amazon_book.yaml
│   │   └── gowalla.yaml
│   ├── models/                       # Настройки моделей
│   │   ├── bpr_mf.yaml              # Baseline 1
│   │   ├── lightgcn.yaml            # Baseline 2
│   │   ├── gcnii.yaml               # Baseline 3
│   │   ├── dgr.yaml                 # Baseline 4
│   │   ├── svd_gcn.yaml             # Baseline 5
│   │   ├── layergcn.yaml            # Baseline 6
│   │   └── groupshuffle_gnn.yaml    # ⭐ ВАШ АЛГОРИТМ
│   └── training.yaml                # Общие настройки обучения
│
├── data/                            # 💾 ДАННЫЕ
│   ├── raw/                         # Сырые данные (скачанные)
│   │   ├── movielens1m/
│   │   ├── yelp2018/
│   │   ├── amazon_book/
│   │   └── gowalla/
│   ├── processed/                   # Обработанные данные
│   │   └── {dataset_name}/
│   │       ├── train.txt
│   │       ├── valid.txt
│   │       ├── test.txt
│   │       └── stats.json
│   └── graphs/                      # Графы (adjacency matrices)
│       └── {dataset_name}/
│           ├── adj_matrix.npz
│           └── norm_adj_matrix.npz
│
├── src/                             # 💻 ИСХОДНЫЙ КОД
│   │
│   ├── data/                        # 📊 DATA PIPELINE
│   │   ├── __init__.py
│   │   ├── dataset.py               # ⚡ Базовый класс датасета
│   │   ├── preprocessing.py         # Препроцессинг
│   │   └── graph_builder.py         # Построение графов
│   │
│   ├── models/                      # 🧠 МОДЕЛИ
│   │   ├── __init__.py
│   │   ├── base.py                  # Базовый класс для всех моделей
│   │   │
│   │   ├── baselines/               # 📌 6 BASELINE МЕТОДОВ
│   │   │   ├── __init__.py
│   │   │   ├── bpr_mf.py           # Baseline 1: BPR-MF
│   │   │   ├── lightgcn.py         # Baseline 2: LightGCN
│   │   │   ├── gcnii.py            # Baseline 3: GCNII
│   │   │   ├── dgr.py              # Baseline 4: DGR
│   │   │   ├── svd_gcn.py          # Baseline 5: SVD-GCN
│   │   │   └── layergcn.py         # Baseline 6: LayerGCN
│   │   │
│   │   └── group_shuffle/           # ⭐ ВАШ АЛГОРИТМ
│   │       ├── __init__.py
│   │       ├── layers.py            # GroupShuffleLayer
│   │       ├── model.py             # GroupShuffleGNN (основная модель)
│   │       └── utils.py             # Вспомогательные функции
│   │
│   ├── training/                    # 🏋️ ОБУЧЕНИЕ
│   │   ├── __init__.py
│   │   ├── trainer.py               # Главный класс Trainer
│   │   ├── losses.py                # BPR Loss и другие
│   │   └── metrics.py               # Recall@K, NDCG@K, etc.
│   │
│   ├── evaluation/                  # 📈 ОЦЕНКА
│   │   ├── __init__.py
│   │   ├── evaluator.py             # Evaluator для метрик
│   │   └── oversmoothing.py         # Анализ over-smoothing
│   │
│   └── utils/                       # 🛠️ УТИЛИТЫ
│       ├── __init__.py
│       ├── logger.py                # Логирование
│       ├── visualization.py         # Визуализация
│       └── statistics.py            # Статистические тесты
│
├── scripts/                         # 🚀 ЗАПУСК ЭКСПЕРИМЕНТОВ
│   ├── run_experiments.py           # ⚡ ГЛАВНЫЙ СКРИПТ
│   ├── train_single_model.py        # Обучение одной модели
│   ├── evaluate_model.py            # Оценка модели
│   └── analyze_results.py           # Анализ результатов
│
├── experiments/                     # 🔬 ЭКСПЕРИМЕНТЫ
│   ├── depth_analysis/              # Анализ глубины (2,4,8,16 слоёв)
│   ├── ablations/                   # Ablation studies
│   ├── efficiency/                  # Анализ эффективности
│   └── oversmoothing/               # Анализ over-smoothing
│
├── notebooks/                       # 📓 JUPYTER NOTEBOOKS
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
├── results/                         # 📊 РЕЗУЛЬТАТЫ
│   ├── checkpoints/                 # Сохранённые модели
│   ├── logs/                        # Логи обучения
│   ├── metrics/                     # CSV с метриками
│   └── figures/                     # Графики и визуализации
│
├── tests/                           # ✅ ТЕСТЫ
│   ├── test_data.py                 # Тесты data pipeline
│   ├── test_models.py               # Тесты моделей
│   ├── test_training.py             # Тесты обучения
│   └── test_evaluation.py           # Тесты оценки
│
├── requirements.txt                 # Зависимости
├── setup.py                         # Установка пакета
└── README.md                        # Документация



---

## 🔄 ПОТОК ВЫПОЛНЕНИЯ ПРОГРАММЫ

### ЭТАП 1: Подготовка данных 📊

**Файлы**: `src/data/dataset.py`, `src/data/preprocessing.py`, `src/data/graph_builder.py`

**Процесс**:
```
ВХОД: Сырые данные (ratings.csv)
  ↓
1. Загрузка (dataset.py)
  ↓
2. Фильтрация (min 10 interactions)
  ↓
3. Бинаризация (implicit feedback)
  ↓
4. Разделение (train/valid/test - temporal split)
  ↓
5. Построение графа (bipartite user-item graph)
  ↓
ВЫХОД: train.txt, valid.txt, test.txt, adj_matrix.npz
```

**Пример кода**:

```python
# src/data/dataset.py
class RecommendationDataset:
    def __init__(self, name, root_dir):
        self.name = name
        self.root_dir = root_dir
    
    def load_raw_data(self):
        """Загрузить сырые данные"""
        pass
    
    def preprocess(self):
        """Препроцессинг: фильтрация, бинаризация"""
        pass
    
    def split(self, strategy='temporal'):
        """Разделение на train/valid/test"""
        pass
    
    def build_graph(self):
        """Построить bipartite граф"""
        pass
```



---

### ЭТАП 2: Инициализация моделей 🧠

**Файлы**:
- `src/models/base.py` - базовый класс
- `src/models/baselines/*.py` - 6 baseline методов
- `src/models/group_shuffle/*.py` - ваш алгоритм

**СТРУКТУРА МОДЕЛЕЙ**:

```
BaseRecommender (base.py)
    ├── BPR_MF (baselines/bpr_mf.py)          ← Baseline 1
    ├── LightGCN (baselines/lightgcn.py)      ← Baseline 2
    ├── GCNII (baselines/gcnii.py)            ← Baseline 3
    ├── DGR (baselines/dgr.py)                ← Baseline 4
    ├── SVD_GCN (baselines/svd_gcn.py)        ← Baseline 5
    ├── LayerGCN (baselines/layergcn.py)      ← Baseline 6
    └── GroupShuffleGNN (group_shuffle/model.py)  ← ⭐ ВАШ АЛГОРИТМ
```


Базовый класс (все модели наследуют от него):
# src/models/base.py
class BaseRecommender(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
    
    def forward(self, users, items):
        """Forward pass - должен быть реализован в каждой модели"""
        raise NotImplementedError
    
    def predict(self, users, items):
        """Предсказание scores для user-item пар"""
        raise NotImplementedError
    
    def get_all_embeddings(self):
        """Получить все embeddings (для evaluation)"""
        raise NotImplementedError


Ваш алгоритм:
# src/models/group_shuffle/model.py
class GroupShuffleGNN(BaseRecommender):
    def __init__(self, n_users, n_items, embedding_dim, n_layers, 
                 block_size, residual_alpha):
        super().__init__(n_users, n_items, embedding_dim)
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # ⭐ ВАШИ СЛОИ
        self.layers = nn.ModuleList([
            GroupShuffleLayer(embedding_dim, block_size)
            for _ in range(n_layers)
        ])
        
        self.residual_alpha = residual_alpha
    
    def forward(self, adj_matrix):
        """
        adj_matrix: normalized adjacency matrix
        """
        # Начальные embeddings
        x_init = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        x = x_init
        all_embeddings = [x]
        
        # Прохождение через слои
        for layer in self.layers:
            x_transformed = layer(x, adj_matrix)
            
            # Residual connection
            x = (1 - self.residual_alpha) * x_transformed + \
                self.residual_alpha * x_init
            
            all_embeddings.append(x)
        
        # Layer aggregation
        x_final = torch.mean(torch.stack(all_embeddings), dim=0)
        
        # Разделить обратно на users и items
        user_emb, item_emb = torch.split(
            x_final, [self.n_users, self.n_items]
        )
        
        return user_emb, item_emb


GroupShuffleLayer (ваш ключевой компонент):
# src/models/group_shuffle/layers.py
class GroupShuffleLayer(nn.Module):
    def __init__(self, dim, block_size):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.n_blocks = dim // block_size
        
        # Параметры для skew-symmetric матриц
        self.skew_params = nn.ParameterList([
            nn.Parameter(torch.randn(block_size, block_size))
            for _ in range(self.n_blocks)
        ])
        
        # Shuffle permutation
        self.register_buffer('perm', self._create_shuffle_permutation())
    
    def _create_shuffle_permutation(self):
        """Создать перестановку для shuffle"""
        perm = torch.randperm(self.dim)
        return perm
    
    def forward(self, x, adj):
        """
        x: node features [N, dim]
        adj: adjacency matrix [N, N]
        """
        # 1. Graph convolution
        x_conv = torch.sparse.mm(adj, x)  # [N, dim]
        
        # 2. Построить ортогональную матрицу (Group)
        W_orth = self._build_orthogonal_matrix()  # [dim, dim]
        
        # 3. Применить ортогональное преобразование
        x_transformed = x_conv @ W_orth  # [N, dim]
        
        # 4. Shuffle
        x_shuffled = x_transformed[:, self.perm]
        
        return x_shuffled
    
    def _build_orthogonal_matrix(self):
        """Построить блочно-диагональную ортогональную матрицу"""
        blocks = []
        for param in self.skew_params:
            # Сделать skew-symmetric
            A_skew = param - param.T
            
            # Exponential map (Lie group)
            block_orth = torch.matrix_exp(A_skew)
            blocks.append(block_orth)
        
        # Собрать блочно-диагональную матрицу
        W_orth = torch.block_diag(*blocks)
        return W_orth



ЭТАП 3: Обучение 🏋️
Файлы: src/training/trainer.py, src/training/losses.py
ПРОЦЕСС ОБУЧЕНИЯ:

1. Инициализация модели
   ↓
2. Для каждой эпохи:
   ├── Сэмплирование батчей (user, positive_item, negative_item)
   ├── Forward pass
   ├── Вычисление BPR Loss
   ├── Backward pass + оптимизация
   └── Валидация (каждые N эпох)
   ↓
3. Early stopping (если validation не улучшается)
   ↓
4. Сохранение чекпоинта


Trainer:
# src/training/trainer.py
class Trainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['lr']
        )
        
        self.loss_fn = BPRLoss()
    
    def train_epoch(self):
        """Одна эпоха обучения"""
        self.model.train()
        total_loss = 0
        
        # Получить все embeddings один раз
        user_emb, item_emb = self.model(self.dataset.adj_matrix)
        
        # Сэмплирование батчей
        for batch in self.dataset.get_train_batches():
            users, pos_items, neg_items = batch
            
            # Scores
            pos_scores = (user_emb[users] * item_emb[pos_items]).sum(dim=1)
            neg_scores = (user_emb[users] * item_emb[neg_items]).sum(dim=1)
            
            # BPR Loss
            loss = self.loss_fn(pos_scores, neg_scores)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.dataset.train_batches)
    
    def validate(self):
        """Валидация"""
        self.model.eval()
        with torch.no_grad():
            metrics = self.evaluator.evaluate(
                self.model, 
                self.dataset.valid_data
            )
        return metrics
    
    def train(self):
        """Полный цикл обучения"""
        best_metric = 0
        patience_counter = 0
        
        for epoch in range(self.config['max_epochs']):
            # Обучение
            train_loss = self.train_epoch()
            
            # Валидация (каждые N эпох)
            if epoch % self.config['eval_every'] == 0:
                metrics = self.validate()
                
                # Early stopping
                if metrics['recall@10'] > best_metric:
                    best_metric = metrics['recall@10']
                    patience_counter = 0
                    self.save_checkpoint()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['patience']:
                    print("Early stopping!")
                    break
            
            # Логирование
            self.log(epoch, train_loss, metrics)



ЭТАП 4: Оценка 📈
Файлы: src/evaluation/evaluator.py, src/evaluation/oversmoothing.py
МЕТРИКИ:

1. Recommendation Quality:
   ├── Recall@K (K=10,20,50)
   ├── Precision@K
   ├── NDCG@K
   └── Coverage

2. Over-smoothing Analysis:
   ├── Cosine Similarity (по слоям)
   ├── MAD (Mean Average Distance)
   └── Embedding Variance


Evaluator:
# src/evaluation/evaluator.py
class Evaluator:
    def __init__(self, k_values=[10, 20, 50]):
        self.k_values = k_values
    
    def evaluate(self, model, test_data):
        """Оценка модели на тестовых данных"""
        model.eval()
        
        # Получить все embeddings
        user_emb, item_emb = model.get_all_embeddings()
        
        # Вычислить scores для всех пар
        scores = user_emb @ item_emb.T  # [n_users, n_items]
        
        metrics = {}
        for k in self.k_values:
            metrics[f'recall@{k}'] = self.recall_at_k(scores, test_data, k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(scores, test_data, k)
        
        return metrics
    
    def recall_at_k(self, scores, test_data, k):
        """Recall@K"""
        # Топ-K предсказания для каждого пользователя
        _, top_k_items = torch.topk(scores, k, dim=1)
        
        # Сравнить с ground truth
        hits = 0
        total = 0
        for user_id, true_items in test_data.items():
            pred_items = top_k_items[user_id].tolist()
            hits += len(set(pred_items) & set(true_items))
            total += len(true_items)
        
        return hits / total


Over-smoothing Analyzer:
# src/evaluation/oversmoothing.py
class OversmoothingAnalyzer:
    def analyze(self, model, dataset):
        """Анализ over-smoothing"""
        model.eval()
        
        # Получить embeddings по слоям
        layer_embeddings = model.get_layer_embeddings(dataset.adj_matrix)
        
        metrics = {}
        for i, emb in enumerate(layer_embeddings):
            # Cosine similarity между всеми парами нодов
            cos_sim = self.cosine_similarity_matrix(emb)
            metrics[f'layer_{i}_mean_cos_sim'] = cos_sim.mean().item()
            
            # MAD (Mean Average Distance)
            mad = self.mean_average_distance(emb)
            metrics[f'layer_{i}_mad'] = mad
        
        return metrics



ЭТАП 5: Запуск экспериментов 🚀
Главный скрипт: scripts/run_experiments.py
ЭКСПЕРИМЕНТЫ:

1. Depth Analysis (глубина 2, 4, 8, 16)
   ↓
2. Ablation Studies (без residual, без shuffle, и т.д.)
   ↓
3. Comparison с baseline методами
   ↓
4. Over-smoothing Analysis
   ↓
5. Statistical Significance Tests


Главный скрипт:
# scripts/run_experiments.py
def main():
    # 1. Загрузить конфигурации
    datasets = ['movielens1m', 'yelp2018', 'amazon_book', 'gowalla']
    models = [
        'bpr_mf',        # Baseline 1
        'lightgcn',      # Baseline 2
        'gcnii',         # Baseline 3
        'dgr',           # Baseline 4
        'svd_gcn',       # Baseline 5
        'layergcn',      # Baseline 6
        'groupshuffle'   # ⭐ ВАШ АЛГОРИТМ
    ]
    
    results = {}
    
    # 2. Для каждого датасета
    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        
        # 3. Для каждой модели
        for model_name in models:
            print(f"\n{'='*50}")
            print(f"Training {model_name} on {dataset_name}")
            print(f"{'='*50}\n")
            
            # Запустить 5 runs с разными seeds
            run_results = []
            for seed in range(5):
                set_seed(seed)
                
                # Инициализация
                model = create_model(model_name, dataset.config)
                trainer = Trainer(model, dataset, config)
                
                # Обучение
                trainer.train()
                
                # Оценка
                metrics = evaluator.evaluate(model, dataset.test_data)
                run_results.append(metrics)
            
            # Агрегировать результаты (mean ± std)
            results[f'{dataset_name}_{model_name}'] = aggregate_results(run_results)
    
    # 4. Сохранить результаты
    save_results(results, 'results/metrics/all_results.csv')
    
    # 5. Статистические тесты
    perform_statistical_tests(results)
    
    # 6. Визуализация
    plot_comparison(results)



✅ ТЕСТЫ (ОБЯЗАТЕЛЬНЫЕ)
Файлы: tests/test_*.py*_
1. Тесты Data Pipeline (tests/test_data.py)
import pytest

def test_data_loading():
    """Тест загрузки данных"""
    dataset = RecommendationDataset('movielens1m')
    dataset.load_raw_data()
    assert dataset.raw_data is not None
    assert len(dataset.raw_data) > 0

def test_preprocessing():
    """Тест препроцессинга"""
    dataset = RecommendationDataset('movielens1m')
    dataset.preprocess()
    
    # Проверить, что нет пользователей/айтемов с < 10 interactions
    assert all(dataset.user_counts >= 10)
    assert all(dataset.item_counts >= 10)

def test_train_test_split():
    """Тест разделения на train/test"""
    dataset = RecommendationDataset('movielens1m')
    dataset.split(strategy='temporal')
    
    # Проверить, что нет пересечений
    train_interactions = set(dataset.train_data)
    test_interactions = set(dataset.test_data)
    assert len(train_interactions & test_interactions) == 0

def test_graph_construction():
    """Тест построения графа"""
    dataset = RecommendationDataset('movielens1m')
    adj_matrix = dataset.build_graph()
    
    # Проверить размерность
    n_nodes = dataset.n_users + dataset.n_items
    assert adj_matrix.shape == (n_nodes, n_nodes)
    
    # Проверить симметричность (bipartite граф)
    assert torch.allclose(adj_matrix, adj_matrix.T)


2. Тесты Моделей (tests/test_models.py)
def test_groupshuffle_layer_orthogonality():
    """⭐ КРИТИЧЕСКИЙ ТЕСТ: Проверить, что матрица ортогональна"""
    layer = GroupShuffleLayer(dim=128, block_size=32)
    W_orth = layer._build_orthogonal_matrix()
    
    # W^T @ W должно быть близко к Identity
    identity = W_orth.T @ W_orth
    expected = torch.eye(128)
    
    assert torch.allclose(identity, expected, atol=1e-5)

def test_model_forward_pass():
    """Тест forward pass для всех моделей"""
    models = [
        BPR_MF, LightGCN, GCNII, DGR, SVD_GCN, LayerGCN, GroupShuffleGNN
    ]
    
    for ModelClass in models:
        model = ModelClass(n_users=100, n_items=200, embedding_dim=64)
        adj_matrix = create_dummy_adj_matrix(300, 300)
        
        # Forward pass
        user_emb, item_emb = model(adj_matrix)
        
        # Проверить размерности
        assert user_emb.shape == (100, 64)
        assert item_emb.shape == (200, 64)

def test_model_gradient_flow():
    """Тест, что градиенты проходят через модель"""
    model = GroupShuffleGNN(n_users=100, n_items=200, embedding_dim=64, n_layers=3)
    adj_matrix = create_dummy_adj_matrix(300, 300)
    
    # Forward + backward
    user_emb, item_emb = model(adj_matrix)
    loss = user_emb.sum() + item_emb.sum()
    loss.backward()
    
    # Проверить, что градиенты не None
    for param in model.parameters():
        assert param.grad is not None

def test_embedding_dimensions():
    """Тест размерностей embeddings"""
    model = GroupShuffleGNN(n_users=943, n_items=1682, embedding_dim=64, n_layers=3)
    
    assert model.user_embedding.weight.shape == (943, 64)
    assert model.item_embedding.weight.shape == (1682, 64)


3. Тесты Обучения (tests/test_training.py)
def test_bpr_loss():
    """Тест BPR Loss"""
    loss_fn = BPRLoss()
    
    pos_scores = torch.tensor([2.0, 3.0, 1.5])
    neg_scores = torch.tensor([1.0, 1.5, 0.5])
    
    loss = loss_fn(pos_scores, neg_scores)
    
    # Loss должен быть положительным
    assert loss.item() > 0

def test_trainer_one_epoch():
    """Тест одной эпохи обучения"""
    model = GroupShuffleGNN(n_users=100, n_items=200, embedding_dim=64, n_layers=2)
    dataset = create_dummy_dataset()
    trainer = Trainer(model, dataset, config={'lr': 0.001})
    
    initial_params = [p.clone() for p in model.parameters()]
    
    # Одна эпоха
    loss = trainer.train_epoch()
    
    # Проверить, что параметры изменились
    for p_init, p_current in zip(initial_params, model.parameters()):
        assert not torch.allclose(p_init, p_current)
    
    # Loss должен быть конечным
    assert not torch.isnan(torch.tensor(loss))

def test_early_stopping():
    """Тест early stopping"""
    model = GroupShuffleGNN(n_users=100, n_items=200, embedding_dim=64, n_layers=2)
    dataset = create_dummy_dataset()
    trainer = Trainer(model, dataset, config={'patience': 3})
    
    # Симулировать ухудшение метрики
    trainer.best_metric = 0.5
    for _ in range(5):
        trainer.validate()  # Метрика не улучшается
    
    # Должен остановиться
    assert trainer.should_stop == True


4. Тесты Evaluation (tests/test_evaluation.py)
def test_recall_at_k():
    """Тест Recall@K"""
    evaluator = Evaluator(k_values=[10])
    
    # Dummy данные
    scores = torch.randn(10, 20)  # 10 users, 20 items
    test_data = {0: [1, 5, 10], 1: [2, 7]}
    
    recall = evaluator.recall_at_k(scores, test_data, k=10)
    
    # Recall должен быть в [0, 1]
    assert 0 <= recall <= 1

def test_oversmoothing_analyzer():
    """Тест анализа over-smoothing"""
    analyzer = OversmoothingAnalyzer()
    
    # Создать embeddings с разной степенью over-smoothing
    emb_layer0 = torch.randn(100, 64)
    emb_layer10 = torch.ones(100, 64)  # Полностью collapsed
    
    cos_sim_0 = analyzer.mean_cosine_similarity(emb_layer0)
    cos_sim_10 = analyzer.mean_cosine_similarity(emb_layer10)
    
    # Layer 10 должен иметь более высокую similarity
    assert cos_sim_10 > cos_sim_0



---

## 📊 ИТОГОВАЯ КАРТИНА

### Где находится ваш алгоритм

```
src/models/group_shuffle/
├── layers.py        ← GroupShuffleLayer (ортогональная трансформация)
├── model.py         ← GroupShuffleGNN (основная модель)
└── utils.py         ← Вспомогательные функции
```

### Где находятся 6 baseline методов

```
src/models/baselines/
├── bpr_mf.py       ← Baseline 1
├── lightgcn.py     ← Baseline 2 (самый важный!)
├── gcnii.py        ← Baseline 3
├── dgr.py          ← Baseline 4
├── svd_gcn.py      ← Baseline 5
└── layergcn.py     ← Baseline 6
```

### Обязательные тесты

```
tests/
├── test_data.py           ← Тесты data pipeline
├── test_models.py         ← ⭐ Тест ортогональности (КРИТИЧЕСКИЙ!)
├── test_training.py       ← Тесты обучения
└── test_evaluation.py     ← Тесты метрик
```

### Главный скрипт запуска

`scripts/run_experiments.py` ← Запускает ВСЁ (все модели, все датасеты)



---

## 🎯 КАК ЭТО РАБОТАЕТ (ПОШАГОВО)

### Пользователь запускает:

```bash
python scripts/run_experiments.py
```

### Скрипт делает:

1. **Загружает 4 датасета** (MovieLens, Yelp, Amazon, Gowalla)
2. **Для каждого датасета**:
   - Обучает 7 моделей (6 baseline + ваш GroupShuffleGNN)
   - Каждую модель запускает 5 раз (разные seeds)
   - Сохраняет результаты
3. **Для каждой модели**:
   - Инициализация → Обучение → Валидация → Тестирование
   - Сохранение чекпоинтов
   - Логирование метрик
4. **После всех экспериментов**:
   - Агрегирование результатов (mean ± std)
   - Статистические тесты (t-test)
   - Визуализация (графики, таблицы)

---

# 📋 СТРУКТУРИРОВАННЫЙ АНАЛИЗ ПРОЕКТА

## 🔍 ЧТО ОПИСАНО В ФАЙЛЕ

### 1. Научная основа

- **Метод**: Group and Shuffle из статьи Gorbunov and Yudin
- **Применение**: Борьба с over-smoothing в GNN через ортогональные преобразования
- **Новизна**: Первое применение метода к задаче рекомендаций на графах

### 2. Экспериментальная установка

- **Датасеты**: 3-4 датасета (Movielens, Amazon-Book, + еще один)
- **Baseline методы**: 6 методов (BPR-MF, LightGCN, GCNII, DGR, SVD-GCN, LayerGCN)
- **Метрики**: Recall@K, NDCG@K, Coverage, метрики over-smoothing
- **Методология**: 5 запусков с разными seeds, статистические тесты

### 3. Архитектура системы

- **Модульная структура**: Разделение на data, models, training, evaluation
- **Базовый класс**: BaseRecommender для единообразия всех моделей
- **Ваш алгоритм**: GroupShuffleGNN с GroupShuffleLayer
- **Тестирование**: Unit-тесты для всех компонентов

### 4. Поток работы

1. Подготовка данных (загрузка, препроцессинг, построение графа)
2. Инициализация моделей (6 baseline + ваш алгоритм)
3. Обучение (BPR loss, early stopping, валидация)
4. Оценка (метрики качества и over-smoothing)
5. Эксперименты (depth analysis, ablation studies)

---

## ✅ ЧТО НУЖНО БУДЕТ СДЕЛАТЬ

### Этап 1: Подготовка инфраструктуры (1-2 недели)

#### 1.1 Настройка окружения
- [ ] Установить Python 3.8+
- [ ] Установить зависимости (PyTorch, PyG, NumPy, Pandas, etc.)
- [ ] Настроить структуру проекта
- [ ] Настроить систему версионирования (Git)

#### 1.2 Создание базовой структуры
- [ ] Создать директории проекта
- [ ] Создать конфигурационные файлы (YAML)
- [ ] Настроить логирование
- [ ] Создать базовые классы (BaseRecommender)

### Этап 2: Data Pipeline (1 неделя)

#### 2.1 Загрузка и обработка данных
- [ ] Реализовать `RecommendationDataset` класс
- [ ] Загрузить датасеты (Movielens, Amazon-Book, + третий)
- [ ] Реализовать фильтрацию (min 10 interactions)
- [ ] Реализовать бинаризацию (implicit feedback)
- [ ] Реализовать разделение (train/valid/test)

#### 2.2 Построение графов
- [ ] Реализовать построение bipartite графа
- [ ] Реализовать нормализацию adjacency matrix
- [ ] Сохранение графов в формате .npz

#### 2.3 Тестирование
- [ ] Тесты загрузки данных
- [ ] Тесты препроцессинга
- [ ] Тесты разделения данных
- [ ] Тесты построения графа

### Этап 3: Baseline методы (3-4 недели)

#### 3.1 Простые методы (1 неделя)
- [ ] **BPR-MF** (1 день) - базовый метод матричной факторизации
- [ ] **LightGCN** (3-4 дня) - самый важный baseline

#### 3.2 Средние методы (1-2 недели)
- [ ] **GCNII** (3-4 дня) - residual connections
- [ ] **SVD-GCN** (5 дней) - спектральный подход

#### 3.3 Сложные методы (1 неделя)
- [ ] **DGR** (3-4 дня) - desmoothing framework
- [ ] **LayerGCN** (3-4 дня) - layer-wise refinement

#### 3.4 Тестирование
- [ ] Тесты forward pass для всех моделей
- [ ] Тесты градиентного потока
- [ ] Тесты размерностей embeddings

### Этап 4: Ваш алгоритм GroupShuffleGNN (2 недели)

#### 4.1 Реализация GroupShuffleLayer
- [ ] Реализовать построение ортогональной матрицы (блочно-диагональная)
- [ ] Реализовать skew-symmetric параметризацию
- [ ] Реализовать exponential map (Lie group)
- [ ] Реализовать shuffle permutation
- [ ] **КРИТИЧЕСКИЙ ТЕСТ**: Проверка ортогональности матрицы

#### 4.2 Реализация GroupShuffleGNN
- [ ] Реализовать архитектуру модели
- [ ] Реализовать residual connections
- [ ] Реализовать layer aggregation
- [ ] Интеграция с BaseRecommender

#### 4.3 Тестирование
- [ ] Тест ортогональности (обязательно!)
- [ ] Тест forward pass
- [ ] Тест градиентного потока
- [ ] Тест размерностей

### Этап 5: Training и Evaluation (1 неделя)

#### 5.1 Training
- [ ] Реализовать BPR Loss
- [ ] Реализовать Trainer класс
- [ ] Реализовать early stopping
- [ ] Реализовать checkpoint management
- [ ] Реализовать батчинг и сэмплирование

#### 5.2 Evaluation
- [ ] Реализовать Recall@K
- [ ] Реализовать NDCG@K
- [ ] Реализовать Coverage
- [ ] Реализовать OversmoothingAnalyzer
- [ ] Реализовать метрики over-smoothing (cosine similarity, MAD)

#### 5.3 Тестирование
- [ ] Тесты BPR Loss
- [ ] Тесты одной эпохи обучения
- [ ] Тесты early stopping
- [ ] Тесты метрик оценки

### Этап 6: Эксперименты (2 недели)

#### 6.1 Основные эксперименты
- [ ] Реализовать `run_experiments.py`
- [ ] Запустить все модели на всех датасетах
- [ ] 5 запусков с разными seeds для каждой конфигурации
- [ ] Сохранение результатов

#### 6.2 Depth Analysis
- [ ] Эксперименты с глубиной 2, 4, 8, 16 слоев
- [ ] Анализ влияния глубины на качество и over-smoothing

#### 6.3 Ablation Studies
- [ ] Без residual connections
- [ ] Без shuffle
- [ ] Без layer aggregation
- [ ] Разные block_size

#### 6.4 Статистический анализ
- [ ] Вычисление mean ± std
- [ ] Статистические тесты (t-test)
- [ ] Вычисление p-values

### Этап 7: Визуализация и документация (1 неделя)

#### 7.1 Визуализация
- [ ] Графики сравнения методов
- [ ] Графики depth analysis
- [ ] Графики ablation studies
- [ ] Визуализация over-smoothing метрик

#### 7.2 Документация
- [ ] Обновить README
- [ ] Документация кода (docstrings)
- [ ] Инструкции по запуску
- [ ] Описание результатов

---

## ❓ ВОПРОСЫ ДЛЯ УТОЧНЕНИЯ

1. **Третий датасет**: Какой датасет будет выбран? (Yelp2018 или Gowalla?)

2. **Параметры Group & Shuffle**:
   - Какой `block_size` планируется использовать?
   - Какое значение `residual_alpha`?
   - Сколько слоев в базовой конфигурации?

3. **Технические требования**:
   - Есть ли доступ к GPU?
   - Какие ограничения по времени/ресурсам?
   - Требования к версиям библиотек?

4. **Метрики over-smoothing**:
   - Какие конкретные метрики будут использоваться?
   - Как будет измеряться степень over-smoothing?

5. **Теоретическая часть**:
   - Планируется ли добавлять теоретические доказательства?
   - Нужны ли теоремы о сохранении норм/углов?

---

## 📊 ОЦЕНКА СЛОЖНОСТИ И ВРЕМЕНИ

| Этап | Время | Сложность | Приоритет |
|------|-------|-----------|-----------|
| Инфраструктура | 1-2 недели | ⭐⭐ | Высокий |
| Data Pipeline | 1 неделя | ⭐⭐ | Высокий |
| Baseline методы | 3-4 недели | ⭐⭐⭐ | Высокий |
| GroupShuffleGNN | 2 недели | ⭐⭐⭐ | Критический |
| Training/Eval | 1 неделя | ⭐⭐ | Высокий |
| Эксперименты | 2 недели | ⭐⭐ | Высокий |
| Визуализация | 1 неделя | ⭐ | Средний |
| **ИТОГО** | **~4 месяца** | | |

---

## 🎯 КРИТИЧЕСКИЕ МОМЕНТЫ

1. **Тест ортогональности** - обязателен для валидации метода
2. **LightGCN** - самый важный baseline, должен быть реализован первым
3. **Статистические тесты** - необходимы для научной обоснованности
4. **5 запусков с разными seeds** - для надежности результатов
5. **Over-smoothing метрики** - ключевая часть исследования

---

## ✅ ЧЕКЛИСТ ГОТОВНОСТИ К РЕАЛИЗАЦИИ

- [ ] Все вопросы уточнены
- [ ] Структура проекта понятна
- [ ] Baseline методы выбраны и изучены
- [ ] Метод Group & Shuffle изучен
- [ ] Датасеты подготовлены или доступны
- [ ] Окружение настроено
- [ ] План реализации составлен


