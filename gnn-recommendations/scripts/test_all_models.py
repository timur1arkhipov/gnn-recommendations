"""
Тестовый скрипт для проверки работы всех baseline моделей.
"""

import sys
from pathlib import Path
import torch

# Определяем корневую директорию проекта
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Добавляем путь к src в PYTHONPATH
sys.path.insert(0, str(project_root / "src"))

from models import (
    BPR_MF,
    LightGCN,
    GCNII,
    DGR,
    SVD_GCN,
    LayerGCN,
    GroupShuffleGNN
)


def test_model(model_class, model_name, **kwargs):
    """Тест создания и forward pass модели."""
    print(f"\n{'='*60}")
    print(f"ТЕСТ: {model_name}")
    print(f"{'='*60}")
    
    n_users = 100
    n_items = 200
    embedding_dim = 64
    
    # Базовые параметры
    model_kwargs = {
        'n_users': n_users,
        'n_items': n_items,
        'embedding_dim': embedding_dim,
        **kwargs
    }
    
    try:
        # Создание модели
        model = model_class(**model_kwargs)
        print(f"✅ Модель создана: {model_name}")
        print(f"   Параметров: {model.get_parameters_count():,}")
        
        # Создаем случайную adjacency matrix
        N = n_users + n_items
        adj_matrix = torch.rand(N, N)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        row_sum = adj_matrix.sum(dim=1, keepdim=True)
        adj_matrix = adj_matrix / (row_sum + 1e-8)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            try:
                user_emb, item_emb = model(adj_matrix)
                print(f"✅ Forward pass выполнен")
                print(f"   User embeddings: {user_emb.shape}")
                print(f"   Item embeddings: {item_emb.shape}")
                
                # Проверка размерностей
                assert user_emb.shape == (n_users, embedding_dim)
                assert item_emb.shape == (n_items, embedding_dim)
                
                # Тест predict
                users = torch.tensor([0, 1, 2])
                items = torch.tensor([0, 1, 2])
                scores = model.predict(users, items, adj_matrix)
                print(f"✅ Predict выполнен: scores shape = {scores.shape}")
                assert scores.shape == (3,)
                
            except Exception as e:
                # Некоторые модели могут не требовать adj_matrix
                if "adj_matrix" in str(e).lower():
                    user_emb, item_emb = model()
                    print(f"✅ Forward pass выполнен (без adj_matrix)")
                    print(f"   User embeddings: {user_emb.shape}")
                    print(f"   Item embeddings: {item_emb.shape}")
                else:
                    raise
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Запуск всех тестов."""
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ВСЕХ BASELINE МОДЕЛЕЙ")
    print("="*60)
    
    models_to_test = [
        (BPR_MF, "BPR-MF"),
        (LightGCN, "LightGCN", {'n_layers': 3}),
        (GCNII, "GCNII", {'n_layers': 3, 'alpha': 0.1, 'beta': 0.5}),
        (DGR, "DGR", {'n_layers': 3, 'lambda_reg': 0.1}),
        (SVD_GCN, "SVD-GCN", {'n_layers': 3, 'rank': 32}),
        (LayerGCN, "LayerGCN", {'n_layers': 3, 'alpha': 0.5}),
        (GroupShuffleGNN, "GroupShuffleGNN", {'n_layers': 3, 'block_size': 8, 'residual_alpha': 0.1}),
    ]
    
    results = []
    for model_info in models_to_test:
        model_class = model_info[0]
        model_name = model_info[1]
        kwargs = model_info[2] if len(model_info) > 2 else {}
        
        success = test_model(model_class, model_name, **kwargs)
        results.append((model_name, success))
    
    # Итоги
    print("\n" + "="*60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*60)
    
    for model_name, success in results:
        status = "✅ ПРОЙДЕН" if success else "❌ ПРОВАЛЕН"
        print(f"{model_name:20s} {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n✅ ВСЕ МОДЕЛИ РАБОТАЮТ КОРРЕКТНО!")
    else:
        print("\n❌ НЕКОТОРЫЕ МОДЕЛИ ИМЕЮТ ПРОБЛЕМЫ")
        sys.exit(1)


if __name__ == "__main__":
    main()

