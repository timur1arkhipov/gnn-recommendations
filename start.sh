# 1. Основные эксперименты (5 seeds)
python scripts/run_multiple_seeds.py 
    --models bpr_mf lightgcn gcnii dgr svd_gcn layergcn groupshuffle_gnn \
    --datasets movie_lens book_crossing \
    --seeds 42 43 44 45 46

# 2. Depth analysis
python scripts/run_depth_analysis.py --model groupshuffle_gnn --dataset movie_lens

# 3. Ablation studies
python scripts/run_ablations.py --dataset movie_lens

# 4. Визуализация
python scripts/analyze_and_plot.py