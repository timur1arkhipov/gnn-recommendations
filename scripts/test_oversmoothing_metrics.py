"""
Тестовый скрипт для демонстрации новых метрик over-smoothing.

Метрики:
- Norm Retention (%): процент сохранения нормы эмбеддингов
- MAD (Mean Average Distance): средняя L2 дистанция между эмбеддингами
- Cosine Similarity: средняя косинусная схожесть
- Variance: дисперсия эмбеддингов

По умолчанию метрики вычисляются для layer 4.
"""

import sys
import torch
from pathlib import Path

# Добавляем путь к src
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Прямые импорты из файлов
from src.evaluation.oversmoothing import compute_oversmoothing_metrics, OversmoothingAnalyzer
from src.models.orthogonal_bundle.model import OrthogonalBundleGNN
from src.models.baselines.lightgcn import LightGCN


def test_oversmoothing_metrics():
    """Тестирование метрик over-smoothing."""

    print("=" * 80)
    print("ТЕСТ МЕТРИК OVER-SMOOTHING")
    print("=" * 80)

    # Параметры
    n_users = 100
    n_items = 200
    embedding_dim = 64
    n_layers = 5

    # Создаём тестовый граф
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")

    # Создаём случайную матрицу смежности (двудольный граф)
    adj_matrix = torch.rand(n_users + n_items, n_users + n_items) > 0.95
    adj_matrix = adj_matrix.float().to(device)

    # Нормализация (как в LightGCN)
    degree = adj_matrix.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
    degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
    adj_matrix_norm = degree_matrix_inv_sqrt @ adj_matrix @ degree_matrix_inv_sqrt

    print(f"Граф: {n_users} users, {n_items} items")
    print(f"Рёбер: {adj_matrix.sum().item():.0f}")

    # ============================================================================
    # ТЕСТ 1: OrthogonalBundleGNN
    # ============================================================================
    print("\n" + "=" * 80)
    print("ТЕСТ 1: OrthogonalBundleGNN")
    print("=" * 80)

    model_bundle = OrthogonalBundleGNN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        block_size=8,
        use_edge_index=False
    ).to(device)

    # Получаем метрики для layer 4 (по умолчанию)
    print("\n[Layer 4] Метрики over-smoothing:")
    metrics_layer4 = compute_oversmoothing_metrics(
        model=model_bundle,
        adj_matrix=adj_matrix_norm,
        layer_idx=4,  # по умолчанию
        sample_size=500
    )

    print(f"  • Norm Retention:     {metrics_layer4['norm_retention']:.2f}%")
    print(f"  • MAD:                {metrics_layer4['mad']:.4f}")
    print(f"  • Cosine Similarity:  {metrics_layer4['cosine_similarity']:.4f}")
    print(f"  • Variance:           {metrics_layer4['variance']:.4f}")

    # ============================================================================
    # ТЕСТ 2: LightGCN (для сравнения)
    # ============================================================================
    print("\n" + "=" * 80)
    print("ТЕСТ 2: LightGCN (Baseline)")
    print("=" * 80)

    model_lightgcn = LightGCN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers
    ).to(device)

    # Получаем метрики для layer 4
    print("\n[Layer 4] Метрики over-smoothing:")
    metrics_lightgcn = compute_oversmoothing_metrics(
        model=model_lightgcn,
        adj_matrix=adj_matrix_norm,
        layer_idx=4,
        sample_size=500
    )

    print(f"  • Norm Retention:     {metrics_lightgcn['norm_retention']:.2f}%")
    print(f"  • MAD:                {metrics_lightgcn['mad']:.4f}")
    print(f"  • Cosine Similarity:  {metrics_lightgcn['cosine_similarity']:.4f}")
    print(f"  • Variance:           {metrics_lightgcn['variance']:.4f}")

    # ============================================================================
    # ТЕСТ 3: Сравнение всех слоёв
    # ============================================================================
    print("\n" + "=" * 80)
    print("ТЕСТ 3: Анализ по всем слоям (OrthogonalBundle)")
    print("=" * 80)

    analyzer = OversmoothingAnalyzer()

    # Получаем embeddings для всех слоёв
    with torch.no_grad():
        layer_embeddings = model_bundle.get_layer_embeddings(adj_matrix=adj_matrix_norm)

    # Анализируем
    all_metrics = analyzer.analyze_layer_embeddings(layer_embeddings, sample_size=500)

    print("\n{:<10} {:<15} {:<12} {:<18} {:<12}".format(
        "Layer", "Norm Ret (%)", "MAD", "Cosine Sim", "Variance"
    ))
    print("-" * 70)

    for layer_name, metrics in all_metrics.items():
        layer_idx = int(layer_name.split('_')[1])
        print("{:<10} {:<15.2f} {:<12.4f} {:<18.4f} {:<12.4f}".format(
            f"Layer {layer_idx}",
            metrics['norm_retention'],
            metrics['mad'],
            metrics['mcs'],
            metrics['variance']
        ))

    # ============================================================================
    # ТЕСТ 4: Использование с edge_index
    # ============================================================================
    print("\n" + "=" * 80)
    print("ТЕСТ 4: Использование с edge_index")
    print("=" * 80)

    # Создаём модель с edge_index
    model_edge = OrthogonalBundleGNN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        block_size=8,
        use_edge_index=True
    ).to(device)

    # Преобразуем adj_matrix в edge_index
    edge_index = adj_matrix.nonzero().t()  # [2, num_edges]

    print(f"Edge index shape: {edge_index.shape}")
    print(f"Количество рёбер: {edge_index.shape[1]}")

    # Получаем метрики
    print("\n[Layer 4] Метрики over-smoothing:")
    metrics_edge = compute_oversmoothing_metrics(
        model=model_edge,
        edge_index=edge_index,
        layer_idx=4,
        sample_size=500
    )

    print(f"  • Norm Retention:     {metrics_edge['norm_retention']:.2f}%")
    print(f"  • MAD:                {metrics_edge['mad']:.4f}")
    print(f"  • Cosine Similarity:  {metrics_edge['cosine_similarity']:.4f}")
    print(f"  • Variance:           {metrics_edge['variance']:.4f}")

    # ============================================================================
    # Сводка
    # ============================================================================
    print("\n" + "=" * 80)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА (Layer 4)")
    print("=" * 80)

    print("\n{:<25} {:<15} {:<12} {:<18} {:<12}".format(
        "Model", "Norm Ret (%)", "MAD", "Cosine Sim", "Variance"
    ))
    print("-" * 85)

    print("{:<25} {:<15.2f} {:<12.4f} {:<18.4f} {:<12.4f}".format(
        "OrthogonalBundle",
        metrics_layer4['norm_retention'],
        metrics_layer4['mad'],
        metrics_layer4['cosine_similarity'],
        metrics_layer4['variance']
    ))

    print("{:<25} {:<15.2f} {:<12.4f} {:<18.4f} {:<12.4f}".format(
        "LightGCN",
        metrics_lightgcn['norm_retention'],
        metrics_lightgcn['mad'],
        metrics_lightgcn['cosine_similarity'],
        metrics_lightgcn['variance']
    ))

    # Вычисляем улучшения
    if metrics_lightgcn['norm_retention'] > 0:
        norm_improvement = ((metrics_layer4['norm_retention'] - metrics_lightgcn['norm_retention'])
                           / metrics_lightgcn['norm_retention'] * 100)
    else:
        norm_improvement = float('inf')

    mad_improvement = ((metrics_layer4['mad'] - metrics_lightgcn['mad'])
                      / metrics_lightgcn['mad'] * 100) if metrics_lightgcn['mad'] > 0 else 0

    print("\n📊 Улучшения OrthogonalBundle vs LightGCN:")
    print(f"  • Norm Retention: +{norm_improvement:.1f}%")
    print(f"  • MAD: +{mad_improvement:.1f}%")

    print("\n" + "=" * 80)
    print("✅ Тест завершён успешно!")
    print("=" * 80)


if __name__ == '__main__':
    test_oversmoothing_metrics()
