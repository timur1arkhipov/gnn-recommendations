ИТОГОВЫЙ ЧЕКЛИСТ: ЧТО НУЖНО ДОБАВИТЬ
ОБЯЗАТЕЛЬНО (для хорошей магистерской):
 Multiple runs (5 seeds) для каждой модели
 Statistical tests (paired t-test)
 Ablation studies (минимум 3 варианта)
 Depth analysis (2, 4, 8, 16 слоёв)
 Visualizations:
 Over-smoothing график (MCS по слоям)
 Performance comparison (bar chart)
 Training curves
 Case study (примеры рекомендаций)

Table 1: Main Results (Mean ± Std over 5 runs)

Dataset: MovieLens-1M и так для каждого датасета

| Model | Recall@10 | NDCG@10 | Precision@10 | Coverage | MCS (L8) | Time (min) |
|-------|-----------|---------|--------------|----------|----------|------------|
| BPR-MF | 0.0487±0.0012 | 0.0638±0.0015 | 0.0056±0.0002 | 0.231 | N/A | 3 |
| LightGCN | 0.0621±0.0015 | 0.0812±0.0018 | 0.0071±0.0003 | 0.342 | 0.695 | 7 |
| GCNII | 0.0658±0.0018 | 0.0857±0.0021 | 0.0076±0.0003 | 0.367 | 0.641 | 9 |
| DGR | 0.0689±0.0016 | 0.0894±0.0019 | 0.0080±0.0003 | 0.384 | 0.612 | 11 |
| SVD-GCN | 0.0647±0.0019 | 0.0843±0.0022 | 0.0075±0.0004 | 0.358 | 0.653 | 10 |
| LayerGCN | 0.0704±0.0014 | 0.0912±0.0017 | 0.0082±0.0002 | 0.396 | 0.598 | 12 |
| GroupShuffle | 0.0738±0.0012* | 0.0956±0.0014* | 0.0086±0.0002* | 0.418 | 0.567* | 11 |

* p < 0.05 vs best baseline (LayerGCN), paired t-test

Best results in bold. GroupShuffleGNN achieves +4.8% Recall@10 and -5.2% MCS improvement.