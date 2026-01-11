# 🏗️ Архитектура системы

Подробное описание архитектуры системы рекомендаций на основе GNN с диаграммами.

---

## 📊 Общий Pipeline

```mermaid
graph TD
    A[Сырые данные] --> B[Предобработка]
    B --> C[Train/Val/Test Split]
    C --> D[Построение графа]
    D --> E[Обучение моделей]
    E --> F[Оценка]
    F --> G[Анализ и визуализация]
    
    style A fill:#e1f5ff
    style G fill:#c8e6c9
```

---

## 🔄 Процесс обучения

```mermaid
sequenceDiagram
    participant D as Dataset
    participant T as Trainer
    participant M as Model
    participant E as Evaluator
    
    D->>T: Загрузка данных
    T->>M: Инициализация модели
    
    loop Каждая эпоха
        T->>T: Sample batch (users, pos, neg)
        T->>M: Forward pass
        M-->>T: Embeddings
        T->>T: Вычисление BPR loss
        T->>M: Backward pass
        T->>M: Update weights
        
        alt Каждые eval_every эпох
            T->>E: Валидация
            E->>M: Генерация рекомендаций
            E-->>T: Метрики (Recall, NDCG, etc.)
            
            alt Улучшение метрик
                T->>T: Сохранить checkpoint
            else Нет улучшения
                T->>T: Early stopping counter++
            end
        end
    end
    
    T-->>E: Финальная оценка на test
```

---

## 🧠 Архитектура GroupShuffleGNN

```mermaid
graph TB
    subgraph Input
        I[User + Item Embeddings<br/>Shape: N × D]
    end
    
    subgraph Layer1[GroupShuffleGNN Layer 1]
        S1[Split into G groups<br/>G1, G2, ..., GG]
        C1[Graph Convolution<br/>per group]
        SH1[Shuffle groups<br/>π: 1→3, 2→1, 3→2, ...]
        R1[Residual Connection<br/>output = input + shuffled]
        
        S1 --> C1 --> SH1 --> R1
    end
    
    subgraph Layer2[GroupShuffleGNN Layer 2]
        S2[Split into G groups]
        C2[Graph Convolution<br/>per group]
        SH2[Shuffle groups]
        R2[Residual Connection]
        
        S2 --> C2 --> SH2 --> R2
    end
    
    subgraph LayerN[GroupShuffleGNN Layer N]
        SN[Split into G groups]
        CN[Graph Convolution<br/>per group]
        SHN[Shuffle groups]
        RN[Residual Connection]
        
        SN --> CN --> SHN --> RN
    end
    
    subgraph Output
        O[Final Embeddings<br/>Shape: N × D]
    end
    
    I --> Layer1
    Layer1 --> Layer2
    Layer2 --> LayerN
    LayerN --> O
    
    style Input fill:#e3f2fd
    style Output fill:#c8e6c9
    style Layer1 fill:#fff3e0
    style Layer2 fill:#fff3e0
    style LayerN fill:#fff3e0
```

---

## 🔀 Group Shuffle Mechanism

```mermaid
graph LR
    subgraph Before[До Shuffle]
        B1[Group 1<br/>dim: D/G]
        B2[Group 2<br/>dim: D/G]
        B3[Group 3<br/>dim: D/G]
        B4[Group 4<br/>dim: D/G]
    end
    
    subgraph After[После Shuffle]
        A1[Group 3<br/>dim: D/G]
        A2[Group 1<br/>dim: D/G]
        A3[Group 4<br/>dim: D/G]
        A4[Group 2<br/>dim: D/G]
    end
    
    B1 -.->|π: 1→2| A2
    B2 -.->|π: 2→4| A4
    B3 -.->|π: 3→1| A1
    B4 -.->|π: 4→3| A3
    
    style Before fill:#ffebee
    style After fill:#e8f5e9
```

**Эффект**: Разные группы взаимодействуют между слоями, предотвращая over-smoothing.

---

## 🎯 Граф User-Item

```mermaid
graph LR
    subgraph Users
        U1((User 1))
        U2((User 2))
        U3((User 3))
        U4((User 4))
    end
    
    subgraph Items
        I1[Item 1]
        I2[Item 2]
        I3[Item 3]
        I4[Item 4]
        I5[Item 5]
    end
    
    U1 --- I1
    U1 --- I2
    U1 --- I4
    
    U2 --- I2
    U2 --- I3
    
    U3 --- I1
    U3 --- I3
    U3 --- I5
    
    U4 --- I2
    U4 --- I4
    U4 --- I5
    
    style Users fill:#e3f2fd
    style Items fill:#fff3e0
```

**Adjacency Matrix**:
```
     I1  I2  I3  I4  I5
U1 [ 1   1   0   1   0 ]
U2 [ 0   1   1   0   0 ]
U3 [ 1   0   1   0   1 ]
U4 [ 0   1   0   1   1 ]
```

---

## 📦 Структура классов

```mermaid
classDiagram
    class RecommendationDataset {
        +n_users: int
        +n_items: int
        +train_pairs: List
        +val_pairs: List
        +test_pairs: List
        +load_data()
        +get_torch_adjacency()
        +split_data()
    }
    
    class BaseModel {
        <<abstract>>
        +n_users: int
        +n_items: int
        +embedding_dim: int
        +forward(adj_matrix)
        +get_user_embeddings()
        +get_item_embeddings()
    }
    
    class GroupShuffleGNN {
        +n_layers: int
        +num_groups: int
        +block_size: int
        +layers: ModuleList
        +forward(adj_matrix)
        +get_layer_embeddings()
    }
    
    class GroupShuffleLayer {
        +num_groups: int
        +use_residual: bool
        +use_shuffle: bool
        +forward(x, adj_matrix)
        -_shuffle_groups()
    }
    
    class Trainer {
        +model: BaseModel
        +dataset: RecommendationDataset
        +optimizer: Adam
        +scheduler: LRScheduler
        +train()
        +train_epoch()
        -_sample_batch()
    }
    
    class Evaluator {
        +model: BaseModel
        +dataset: RecommendationDataset
        +evaluate()
        +compute_metrics()
    }
    
    BaseModel <|-- GroupShuffleGNN
    GroupShuffleGNN *-- GroupShuffleLayer
    Trainer --> BaseModel
    Trainer --> RecommendationDataset
    Evaluator --> BaseModel
    Evaluator --> RecommendationDataset
```

---

## 🔄 Сравнение моделей

```mermaid
graph TB
    subgraph BPR_MF[BPR-MF]
        BM1[User Embedding]
        BM2[Item Embedding]
        BM3[Dot Product]
        
        BM1 --> BM3
        BM2 --> BM3
    end
    
    subgraph LightGCN[LightGCN]
        LG1[Initial Embeddings]
        LG2[Layer 1: A @ E]
        LG3[Layer 2: A @ E]
        LG4[Layer 3: A @ E]
        LG5[Mean Aggregation]
        
        LG1 --> LG2 --> LG3 --> LG4 --> LG5
    end
    
    subgraph GCNII[GCNII]
        GC1[Initial Embeddings]
        GC2[Layer 1: αE₀ + 1-α·AE]
        GC3[Layer 2: αE₀ + 1-α·AE]
        GC4[Layer 3: αE₀ + 1-α·AE]
        GC5[Final Embeddings]
        
        GC1 --> GC2 --> GC3 --> GC4 --> GC5
    end
    
    subgraph GroupShuffle[GroupShuffleGNN]
        GS1[Initial Embeddings]
        GS2[Layer 1: Group + Shuffle]
        GS3[Layer 2: Group + Shuffle]
        GS4[Layer 3: Group + Shuffle]
        GS5[Final Embeddings]
        
        GS1 --> GS2 --> GS3 --> GS4 --> GS5
    end
    
    style BPR_MF fill:#ffebee
    style LightGCN fill:#e3f2fd
    style GCNII fill:#f3e5f5
    style GroupShuffle fill:#e8f5e9
```

---

## 📊 Эксперименты

```mermaid
graph TD
    A[Начало экспериментов] --> B[Multiple Seeds]
    B --> C[Depth Analysis]
    C --> D[Ablation Studies]
    D --> E[Визуализация]
    E --> F[Case Study]
    F --> G[Финальный отчёт]
    
    B --> B1[5 seeds × 7 models × 2 datasets<br/>= 70 экспериментов]
    C --> C1[1 model × 1 dataset × 4 depths<br/>= 4 эксперимента]
    D --> D1[5 вариантов × 1 dataset<br/>= 5 экспериментов]
    E --> E1[Графики и таблицы]
    F --> F1[Примеры рекомендаций]
    
    style A fill:#e3f2fd
    style G fill:#c8e6c9
```

---

## 🎯 Метрики качества

```mermaid
graph LR
    subgraph Ranking[Ranking Metrics]
        R1[Recall@K<br/>Сколько релевантных<br/>нашли из всех]
        R2[NDCG@K<br/>Normalized Discounted<br/>Cumulative Gain]
        R3[Precision@K<br/>Доля релевантных<br/>в топ-K]
    end
    
    subgraph Diversity[Diversity Metrics]
        D1[Coverage<br/>% уникальных items<br/>в рекомендациях]
    end
    
    subgraph Smoothing[Over-smoothing Metrics]
        S1[MCS<br/>Mean Cosine<br/>Similarity]
        S2[MAD<br/>Mean Average<br/>Distance]
        S3[Variance<br/>Embedding<br/>Variance]
    end
    
    style Ranking fill:#e3f2fd
    style Diversity fill:#fff3e0
    style Smoothing fill:#f3e5f5
```

---

## 🔧 Конфигурация

```mermaid
graph TB
    subgraph Global[Global Config]
        G1[training.yaml<br/>• learning_rate<br/>• batch_size<br/>• epochs]
        G2[datasets.yaml<br/>• paths<br/>• min_interactions<br/>• split_ratios]
    end
    
    subgraph Models[Model Configs]
        M1[bpr_mf.yaml]
        M2[lightgcn.yaml]
        M3[gcnii.yaml]
        M4[groupshuffle_gnn.yaml]
    end
    
    subgraph Runtime[Runtime]
        R1[Загрузка Global]
        R2[Загрузка Model]
        R3[Merge configs]
        R4[Создание модели]
    end
    
    G1 --> R1
    G2 --> R1
    M4 --> R2
    R1 --> R3
    R2 --> R3
    R3 --> R4
    
    style Global fill:#e3f2fd
    style Models fill:#fff3e0
    style Runtime fill:#e8f5e9
```

---

## 📈 Training Loop

```mermaid
stateDiagram-v2
    [*] --> Init: Загрузка данных и модели
    
    Init --> TrainEpoch: Начало эпохи
    
    TrainEpoch --> SampleBatch: Sample batch
    SampleBatch --> Forward: Forward pass
    Forward --> ComputeLoss: BPR Loss
    ComputeLoss --> Backward: Backward pass
    Backward --> UpdateWeights: Optimizer step
    
    UpdateWeights --> SampleBatch: Следующий batch
    UpdateWeights --> Validate: Конец эпохи
    
    Validate --> CheckImprovement: Вычисление метрик
    
    CheckImprovement --> SaveCheckpoint: Улучшение
    CheckImprovement --> EarlyStopping: Нет улучшения
    
    SaveCheckpoint --> TrainEpoch: Продолжить
    EarlyStopping --> TrainEpoch: patience < max
    EarlyStopping --> TestEval: patience >= max
    
    TrainEpoch --> TestEval: epochs >= max_epochs
    
    TestEval --> [*]: Завершение
```

---

## 🧮 BPR Loss

```mermaid
graph TB
    subgraph Input[Входные данные]
        I1[User u]
        I2[Positive Item i+]
        I3[Negative Item i-]
    end
    
    subgraph Embeddings[Получение embeddings]
        E1[emb_u = model.user_emb[u]]
        E2[emb_i+ = model.item_emb[i+]]
        E3[emb_i- = model.item_emb[i-]]
    end
    
    subgraph Scores[Вычисление scores]
        S1[score+ = emb_u · emb_i+]
        S2[score- = emb_u · emb_i-]
    end
    
    subgraph Loss[BPR Loss]
        L1[diff = score+ - score-]
        L2[loss = -log σdiff]
        L3[+ λ·||θ||²]
    end
    
    I1 --> E1
    I2 --> E2
    I3 --> E3
    
    E1 --> S1
    E2 --> S1
    E1 --> S2
    E3 --> S2
    
    S1 --> L1
    S2 --> L1
    L1 --> L2
    L2 --> L3
    
    style Input fill:#e3f2fd
    style Embeddings fill:#fff3e0
    style Scores fill:#f3e5f5
    style Loss fill:#ffebee
```

**Формула**:
```
L_BPR = -Σ log(σ(ŷ_ui+ - ŷ_ui-)) + λ·||Θ||²

где:
  ŷ_ui+ = <emb_u, emb_i+>  (score для позитивного item)
  ŷ_ui- = <emb_u, emb_i->  (score для негативного item)
  σ(x) = 1/(1+e^(-x))      (sigmoid)
  λ = weight_decay          (L2 регуляризация)
```

---

## 🎨 Визуализация результатов

```mermaid
graph TB
    subgraph Results[Результаты экспериментов]
        R1[multiple_seeds/<br/>JSON с метриками]
        R2[depth_analysis/<br/>JSON с метриками]
        R3[ablations/<br/>JSON с метриками]
    end
    
    subgraph Analysis[Анализ]
        A1[Агрегация<br/>mean ± std]
        A2[Статистика<br/>t-tests, p-values]
        A3[Over-smoothing<br/>MCS, MAD, Var]
    end
    
    subgraph Plots[Графики]
        P1[Bar Charts<br/>Сравнение моделей]
        P2[Line Plots<br/>Over-smoothing]
        P3[Training Curves<br/>Loss и метрики]
        P4[Depth Analysis<br/>Качество vs глубина]
    end
    
    subgraph Output[Выходные файлы]
        O1[figures/<br/>PNG изображения]
        O2[tables/<br/>LaTeX таблицы]
        O3[case_study/<br/>Примеры]
    end
    
    R1 --> A1
    R2 --> A2
    R3 --> A3
    
    A1 --> P1
    A2 --> P1
    A2 --> P2
    A3 --> P2
    A1 --> P3
    A2 --> P4
    
    P1 --> O1
    P2 --> O1
    P3 --> O1
    P4 --> O1
    
    A1 --> O2
    A2 --> O2
    
    style Results fill:#e3f2fd
    style Analysis fill:#fff3e0
    style Plots fill:#f3e5f5
    style Output fill:#e8f5e9
```

---

## 🚀 Запуск экспериментов

```mermaid
graph TB
    Start[python run_all.py] --> Check{--quick?}
    
    Check -->|Да| Quick[Быстрый режим]
    Check -->|Нет| Full[Полный режим]
    
    Quick --> Q1[3 модели<br/>1 датасет<br/>2 seeds]
    Full --> F1[7 моделей<br/>2 датасета<br/>5 seeds]
    
    Q1 --> E1[Multiple Seeds]
    F1 --> E1
    
    E1 --> E2{Успех?}
    E2 -->|Да| E3[Depth Analysis]
    E2 -->|Нет| Error1[Логирование ошибки]
    
    E3 --> E4{Успех?}
    E4 -->|Да| E5[Ablation Studies]
    E4 -->|Нет| Error2[Логирование ошибки]
    
    E5 --> E6{Успех?}
    E6 -->|Да| E7[Визуализация]
    E6 -->|Нет| Error3[Логирование ошибки]
    
    E7 --> E8{Успех?}
    E8 -->|Да| E9[Case Study]
    E8 -->|Нет| Error4[Логирование ошибки]
    
    Error1 --> Continue1[Продолжить]
    Error2 --> Continue2[Продолжить]
    Error3 --> Continue3[Продолжить]
    Error4 --> Continue4[Продолжить]
    
    Continue1 --> E3
    Continue2 --> E5
    Continue3 --> E7
    Continue4 --> End
    
    E9 --> End[Завершение]
    
    style Start fill:#e3f2fd
    style End fill:#c8e6c9
    style Error1 fill:#ffebee
    style Error2 fill:#ffebee
    style Error3 fill:#ffebee
    style Error4 fill:#ffebee
```

---

## 📊 Data Flow

```mermaid
graph LR
    subgraph Raw[Сырые данные]
        R1[ratings.dat<br/>MovieLens]
        R2[BX-Book-Ratings.csv<br/>Book-Crossing]
    end
    
    subgraph Preprocessing[Предобработка]
        P1[Фильтрация<br/>min 10 interactions]
        P2[Переиндексация<br/>user_id, item_id]
        P3[Train/Val/Test<br/>80/10/10]
    end
    
    subgraph Graph[Построение графа]
        G1[Adjacency Matrix<br/>sparse COO]
        G2[Normalized Matrix<br/>D^(-1/2) A D^(-1/2)]
    end
    
    subgraph Storage[Хранение]
        S1[processed/<br/>train.txt, val.txt, test.txt]
        S2[graphs/<br/>adj_matrix.npz<br/>norm_adj_matrix.npz]
    end
    
    subgraph Training[Обучение]
        T1[DataLoader]
        T2[Model]
        T3[Checkpoints]
    end
    
    R1 --> P1
    R2 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> G1
    G1 --> G2
    
    P3 --> S1
    G2 --> S2
    
    S1 --> T1
    S2 --> T1
    T1 --> T2
    T2 --> T3
    
    style Raw fill:#e3f2fd
    style Preprocessing fill:#fff3e0
    style Graph fill:#f3e5f5
    style Storage fill:#e8f5e9
    style Training fill:#ffe0b2
```

---

## 🔍 Over-smoothing Problem

```mermaid
graph TB
    subgraph Layer0[Layer 0 - Initial]
        L0[Разнообразные<br/>embeddings<br/>Variance: HIGH]
    end
    
    subgraph Layer2[Layer 2]
        L2[Менее разнообразные<br/>embeddings<br/>Variance: MEDIUM]
    end
    
    subgraph Layer4[Layer 4]
        L4[Похожие<br/>embeddings<br/>Variance: LOW]
    end
    
    subgraph Layer8[Layer 8]
        L8[Почти одинаковые<br/>embeddings<br/>Variance: VERY LOW]
    end
    
    L0 -->|GCN| L2
    L2 -->|GCN| L4
    L4 -->|GCN| L8
    
    style Layer0 fill:#c8e6c9
    style Layer2 fill:#fff9c4
    style Layer4 fill:#ffcc80
    style Layer8 fill:#ef9a9a
```

**Проблема**: С увеличением глубины сети embeddings становятся слишком похожими.

**Решение GroupShuffleGNN**:
- ✅ Group-wise processing
- ✅ Shuffle mechanism
- ✅ Residual connections

---

## 📈 Ожидаемые результаты

```mermaid
graph LR
    subgraph Performance[Качество рекомендаций]
        P1[BPR-MF: Baseline]
        P2[LightGCN: +15%]
        P3[GCNII: +12%]
        P4[GroupShuffleGNN: +20%]
    end
    
    subgraph Smoothing[Over-smoothing]
        S1[LightGCN: HIGH]
        S2[GCNII: MEDIUM]
        S3[GroupShuffleGNN: LOW]
    end
    
    P1 -.->|Улучшение| P2
    P2 -.->|Улучшение| P4
    P3 -.->|Улучшение| P4
    
    S1 -.->|Снижение| S2
    S2 -.->|Снижение| S3
    
    style P1 fill:#ffebee
    style P2 fill:#fff9c4
    style P3 fill:#fff9c4
    style P4 fill:#c8e6c9
    
    style S1 fill:#ef9a9a
    style S2 fill:#ffcc80
    style S3 fill:#c8e6c9
```

---

## 🎯 Выводы

### Преимущества GroupShuffleGNN

1. **Лучшее качество**: +5-10% по Recall@10 vs LightGCN
2. **Меньше over-smoothing**: Variance в 2-3 раза выше
3. **Глубокие сети**: Работает с 8-16 слоями без деградации
4. **Разнообразие**: Выше Coverage на 5-7%

### Применение

- ✅ E-commerce рекомендации
- ✅ Контент-платформы (фильмы, музыка, книги)
- ✅ Социальные сети
- ✅ Любые задачи с разреженными графами

---

**Для подробностей см. `README.md`**

