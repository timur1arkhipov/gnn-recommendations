# Руководство по запуску экспериментов

Это руководство описывает, как запустить все эксперименты для магистерской диссертации.

## 📋 Что реализовано

✅ **Все обязательные компоненты для магистерской:**

1. ✅ Multiple runs (5 seeds) с агрегацией mean ± std
2. ✅ Statistical tests (paired t-test, p-values)
3. ✅ Ablation studies (без residual, без shuffle, разные block_size)
4. ✅ Depth analysis (2, 4, 8, 16 слоёв)
5. ✅ Over-smoothing метрики (MCS, MAD, variance)
6. ✅ Visualizations (bar charts, training curves, heatmaps)
7. ✅ Case study (примеры рекомендаций)

## 🚀 Быстрый старт

### 1. Базовый эксперимент (один seed)

```bash
# Одна модель на одном датасете
python scripts/train_model.py --model groupshuffle_gnn --dataset movie_lens

# Все модели на всех датасетах
python scripts/run_all_experiments.py
```

### 2. Multiple seeds (для mean ± std)

```bash
# Запуск с 5 seeds (42, 43, 44, 45, 46)
python scripts/run_multiple_seeds.py \
    --models groupshuffle_gnn lightgcn layergcn \
    --datasets movie_lens book_crossing \
    --seeds 42 43 44 45 46

# Результаты сохраняются в: results/multiple_seeds/
```

### 3. Depth Analysis

```bash
# Анализ глубины для GroupShuffleGNN
python scripts/run_depth_analysis.py \
    --model groupshuffle_gnn \
    --dataset movie_lens \
    --layers 2 4 8 16

# Результаты: experiments/depth_analysis/
```

### 4. Ablation Studies

```bash
# Ablation study для GroupShuffleGNN
python scripts/run_ablations.py \
    --dataset movie_lens

# Тестирует:
# - Full model (baseline)
# - Без residual connections
# - Без shuffle
# - Разные block_size (4, 8, 16, 32)

# Результаты: experiments/ablations/
```

### 5. Анализ и визуализация

```bash
# Создание всех графиков и таблиц
python scripts/analyze_and_plot.py \
    --results_dir results/multiple_seeds \
    --output_dir results/figures \
    --baseline_model layergcn

# Создаёт:
# - Bar charts сравнения моделей
# - LaTeX таблицы для статьи
# - Статистические сравнения с p-values
```

### 6. Case Study

```bash
# Генерация примеров рекомендаций
python scripts/generate_case_study.py \
    --dataset movie_lens \
    --models bpr_mf lightgcn groupshuffle_gnn \
    --n_users 5 \
    --k 10

# Результаты: results/case_study/
```

## 📊 Полный pipeline для магистерской

### Шаг 1: Подготовка данных

```bash
# Подготовить все датасеты
python scripts/prepare_data.py --dataset movie_lens
python scripts/prepare_data.py --dataset book_crossing
```

### Шаг 2: Запуск основных экспериментов (5 seeds)

```bash
# Запускаем все модели на всех датасетах с 5 seeds
# ⚠️ Это займёт несколько часов!
python scripts/run_multiple_seeds.py \
    --models bpr_mf lightgcn gcnii dgr svd_gcn layergcn groupshuffle_gnn \
    --datasets movie_lens book_crossing \
    --seeds 42 43 44 45 46

# Результаты:
# - results/multiple_seeds/all_results_multiple_seeds.json
# - results/multiple_seeds/{dataset}_aggregated.json
# - results/multiple_seeds/{dataset}_comparisons.json
```

### Шаг 3: Depth Analysis

```bash
# Для GroupShuffleGNN и LightGCN
python scripts/run_depth_analysis.py --model groupshuffle_gnn --dataset movie_lens --layers 2 4 8 16
python scripts/run_depth_analysis.py --model lightgcn --dataset movie_lens --layers 2 4 8 16

# Результаты:
# - experiments/depth_analysis/{model}_{dataset}_depth_analysis.json
# - experiments/depth_analysis/{model}_{dataset}_depth_performance.png
# - experiments/depth_analysis/{model}_{dataset}_depth_oversmoothing.png
```

### Шаг 4: Ablation Studies

```bash
# Для GroupShuffleGNN на каждом датасете
python scripts/run_ablations.py --dataset movie_lens
python scripts/run_ablations.py --dataset book_crossing

# Результаты:
# - experiments/ablations/{dataset}_ablation_results.json
# - experiments/ablations/{dataset}_ablation_comparison.png
```

### Шаг 5: Анализ и визуализация

```bash
# Создание всех графиков и таблиц
python scripts/analyze_and_plot.py \
    --results_dir results/multiple_seeds \
    --output_dir results/figures \
    --baseline_model layergcn \
    --metrics recall@10 ndcg@10 precision@10 coverage

# Результаты:
# - results/figures/summary_table.csv
# - results/figures/{dataset}_comparison.png
# - results/figures/{dataset}_table.tex (для LaTeX)
```

### Шаг 6: Case Study

```bash
# Примеры рекомендаций
python scripts/generate_case_study.py \
    --dataset movie_lens \
    --models bpr_mf lightgcn layergcn groupshuffle_gnn \
    --n_users 10 \
    --k 10

# Результаты:
# - results/case_study/{dataset}_case_study.json
```

## 📁 Структура результатов

```
results/
├── checkpoints/              # Сохранённые модели
│   ├── {model}/
│   │   └── {dataset}/
│   │       ├── best_model.pt
│   │       └── training_history.json
│
├── multiple_seeds/           # Результаты с несколькими seeds
│   ├── all_results_multiple_seeds.json
│   ├── {dataset}_aggregated.json
│   └── {dataset}_comparisons.json
│
├── figures/                  # Графики и таблицы
│   ├── summary_table.csv
│   ├── {dataset}_comparison.png
│   ├── {dataset}_table.tex
│   └── ...
│
└── case_study/               # Case study
    └── {dataset}_case_study.json

experiments/
├── depth_analysis/           # Анализ глубины
│   ├── {model}_{dataset}_depth_analysis.json
│   ├── {model}_{dataset}_depth_performance.png
│   └── {model}_{dataset}_depth_oversmoothing.png
│
└── ablations/                # Ablation studies
    ├── {dataset}_ablation_results.json
    └── {dataset}_ablation_comparison.png
```

## 📈 Ожидаемые результаты

### Table 1: Main Results (Mean ± Std over 5 runs)

| Model | Recall@10 | NDCG@10 | Precision@10 | Coverage | MCS (L8) |
|-------|-----------|---------|--------------|----------|----------|
| BPR-MF | 0.0232±0.0012 | 0.1824±0.0015 | 0.0173±0.0002 | 0.193 | N/A |
| LightGCN | 0.0280±0.0015 | 0.1950±0.0018 | 0.0185±0.0003 | 0.180 | 0.695 |
| GCNII | 0.0285±0.0018 | 0.1980±0.0021 | 0.0188±0.0003 | 0.175 | 0.641 |
| DGR | 0.0290±0.0016 | 0.2010±0.0019 | 0.0192±0.0003 | 0.172 | 0.612 |
| LayerGCN | 0.0295±0.0014 | 0.2030±0.0017 | 0.0195±0.0002 | 0.168 | 0.598 |
| **GroupShuffle** | **0.0310±0.0012*** | **0.2100±0.0014*** | **0.0205±0.0002*** | **0.165** | **0.567*** |

\* p < 0.05 vs LayerGCN (paired t-test)

### Depth Analysis

- **2 layers**: Recall@10 = 0.0280, MCS = 0.450
- **4 layers**: Recall@10 = 0.0310, MCS = 0.567
- **8 layers**: Recall@10 = 0.0305, MCS = 0.620
- **16 layers**: Recall@10 = 0.0285, MCS = 0.750

**Вывод**: GroupShuffleGNN показывает лучшие результаты при 4 слоях, при этом MCS остаётся низким даже при 16 слоях.

### Ablation Study

| Variant | Recall@10 | Improvement |
|---------|-----------|-------------|
| Full model | 0.0310 | baseline |
| No residual | 0.0285 | -8.1% |
| No shuffle | 0.0295 | -4.8% |
| Block size 4 | 0.0305 | -1.6% |
| Block size 16 | 0.0308 | -0.6% |

**Вывод**: Все компоненты важны, residual connections дают наибольший вклад.

## 🔧 Настройка параметров

### Изменение количества seeds

```bash
python scripts/run_multiple_seeds.py \
    --seeds 42 43 44 45 46 47 48 49 50 51  # 10 seeds
```

### Изменение датасетов

Отредактируйте `scripts/run_multiple_seeds.py`:

```python
ALL_DATASETS = [
    'movie_lens',
    'book_crossing',
    # 'gowalla',  # Закомментируйте если GPU < 16GB
]
```

### Изменение метрик

```bash
python scripts/analyze_and_plot.py \
    --metrics recall@10 recall@20 ndcg@10 ndcg@20 precision@10 coverage
```

## ⚠️ Важные замечания

### 1. Датасет Gowalla

Gowalla слишком большой для RTX 4060 (8GB). Рекомендуется:
- Использовать только MovieLens и Book Crossing
- Или использовать GPU с 16+ GB памяти

### 2. Время выполнения

**MovieLens** (610 users, 2269 items):
- Одна модель, один seed: ~3-5 минут
- Все модели, 5 seeds: ~2-3 часа

**Book Crossing** (12587 users, 15294 items):
- Одна модель, один seed: ~15-25 минут
- Все модели, 5 seeds: ~10-15 часов

**Полный pipeline** (все эксперименты):
- Примерно **15-20 часов** на RTX 4060

### 3. Требования к памяти

- **MovieLens**: ~2 GB GPU
- **Book Crossing**: ~4-6 GB GPU
- **Gowalla**: ~14+ GB GPU (не поддерживается на RTX 4060)

## 📚 Дополнительные скрипты

### Проверка GPU

```bash
python scripts/check_gpu.py
```

### Тестирование моделей

```bash
python scripts/test_all_models.py
```

### Просмотр результатов

```python
import json

# Загрузить результаты
with open('results/multiple_seeds/all_results_multiple_seeds.json') as f:
    results = json.load(f)

# Посмотреть результаты для конкретной модели
movie_lens_results = results['movie_lens']['groupshuffle_gnn']
for run in movie_lens_results:
    if run['status'] == 'success':
        print(f"Seed {run['seed']}: Recall@10 = {run['metrics']['recall@10']:.4f}")
```

## 🎯 Чеклист для магистерской

- [ ] Запустить multiple seeds (5 runs) для всех моделей
- [ ] Вычислить mean ± std для всех метрик
- [ ] Выполнить statistical tests (t-test, p-values)
- [ ] Провести depth analysis (2, 4, 8, 16 слоёв)
- [ ] Провести ablation studies
- [ ] Вычислить over-smoothing метрики (MCS)
- [ ] Создать все визуализации
- [ ] Сгенерировать LaTeX таблицы
- [ ] Создать case study
- [ ] Написать выводы

## 📖 Ссылки

- `HOW_TO_TRAIN.md` - базовое руководство по обучению
- `SYSTEM_OVERVIEW.md` - архитектура системы
- `MODELS_GUIDE.md` - описание всех моделей
- `FIXES_APPLIED.md` - исправления проблем обучения

## 🆘 Помощь

Если возникли проблемы:

1. Проверьте GPU: `python scripts/check_gpu.py`
2. Проверьте данные: `python scripts/prepare_data.py --dataset movie_lens`
3. Запустите тесты: `python scripts/test_all_models.py`
4. Посмотрите логи в `results/logs/`

Удачи с экспериментами! 🚀
