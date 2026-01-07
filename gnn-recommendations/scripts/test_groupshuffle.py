"""
Тестовый скрипт для проверки работы GroupShuffleGNN модели.
"""

import sys
from pathlib import Path
import torch

# Определяем корневую директорию проекта
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Добавляем путь к src в PYTHONPATH
sys.path.insert(0, str(project_root / "src"))

from models import GroupShuffleGNN


def test_model_creation():
    """Тест создания модели."""
    print("="*60)
    print("ТЕСТ 1: Создание модели")
    print("="*60)
    
    n_users = 100
    n_items = 200
    embedding_dim = 64
    n_layers = 3
    block_size = 8
    
    model = GroupShuffleGNN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        block_size=block_size,
        residual_alpha=0.1
    )
    
    print(f"✅ Модель создана успешно!")
    print(f"   Параметров: {model.get_parameters_count():,}")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Layers: {n_layers}")
    print(f"   Block size: {block_size}")
    print()


def test_forward_pass():
    """Тест forward pass."""
    print("="*60)
    print("ТЕСТ 2: Forward pass")
    print("="*60)
    
    n_users = 100
    n_items = 200
    embedding_dim = 64
    n_layers = 2
    block_size = 8
    
    model = GroupShuffleGNN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        block_size=block_size,
        residual_alpha=0.1
    )
    
    # Создаем случайную adjacency matrix (dense для простоты)
    N = n_users + n_items
    adj_matrix = torch.rand(N, N)
    # Делаем симметричной и нормализуем
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    # Нормализация (простая)
    row_sum = adj_matrix.sum(dim=1, keepdim=True)
    adj_matrix = adj_matrix / (row_sum + 1e-8)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(adj_matrix)
    
    print(f"✅ Forward pass выполнен успешно!")
    print(f"   User embeddings shape: {user_emb.shape}")
    print(f"   Item embeddings shape: {item_emb.shape}")
    print(f"   Expected: user_emb [{n_users}, {embedding_dim}], item_emb [{n_items}, {embedding_dim}]")
    
    assert user_emb.shape == (n_users, embedding_dim), f"Неверная форма user_emb: {user_emb.shape}"
    assert item_emb.shape == (n_items, embedding_dim), f"Неверная форма item_emb: {item_emb.shape}"
    print()


def test_predict():
    """Тест предсказания."""
    print("="*60)
    print("ТЕСТ 3: Предсказание scores")
    print("="*60)
    
    n_users = 100
    n_items = 200
    embedding_dim = 64
    n_layers = 2
    block_size = 8
    
    model = GroupShuffleGNN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        block_size=block_size,
        residual_alpha=0.1
    )
    
    # Создаем случайную adjacency matrix
    N = n_users + n_items
    adj_matrix = torch.rand(N, N)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    row_sum = adj_matrix.sum(dim=1, keepdim=True)
    adj_matrix = adj_matrix / (row_sum + 1e-8)
    
    # Тестируем predict
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([0, 1, 2])
    
    model.eval()
    with torch.no_grad():
        scores = model.predict(users, items, adj_matrix)
    
    print(f"✅ Предсказание выполнено успешно!")
    print(f"   Scores shape: {scores.shape}")
    print(f"   Scores: {scores}")
    print(f"   Expected shape: [3]")
    
    assert scores.shape == (3,), f"Неверная форма scores: {scores.shape}"
    print()


def test_orthogonality():
    """Тест ортогональности матриц."""
    print("="*60)
    print("ТЕСТ 4: Ортогональность матриц")
    print("="*60)
    
    from models.group_shuffle import GroupShuffleLayer
    
    dim = 64
    block_size = 8
    
    layer = GroupShuffleLayer(dim=dim, block_size=block_size)
    
    # Получаем ошибку ортогональности
    error = layer.get_orthogonality_error()
    
    print(f"✅ Ошибка ортогональности: {error.item():.6f}")
    print(f"   (должна быть близка к 0)")
    
    # Ошибка должна быть маленькой (< 1e-5 для точных вычислений)
    assert error.item() < 1e-3, f"Ошибка ортогональности слишком большая: {error.item()}"
    print()


def test_gradient_flow():
    """Тест потока градиентов."""
    print("="*60)
    print("ТЕСТ 5: Поток градиентов")
    print("="*60)
    
    n_users = 50
    n_items = 100
    embedding_dim = 32
    n_layers = 2
    block_size = 8
    
    model = GroupShuffleGNN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        block_size=block_size,
        residual_alpha=0.1
    )
    
    # Создаем случайную adjacency matrix
    N = n_users + n_items
    adj_matrix = torch.rand(N, N)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    row_sum = adj_matrix.sum(dim=1, keepdim=True)
    adj_matrix = adj_matrix / (row_sum + 1e-8)
    
    # Forward + backward
    model.train()
    user_emb, item_emb = model(adj_matrix)
    loss = user_emb.sum() + item_emb.sum()
    loss.backward()
    
    # Проверяем, что градиенты не None
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    print(f"✅ Градиенты проходят через модель: {has_gradients}")
    
    if has_gradients:
        print("   Примеры параметров с градиентами:")
        count = 0
        for name, param in model.named_parameters():
            if param.grad is not None and count < 3:
                print(f"   - {name}: grad_norm = {param.grad.norm().item():.6f}")
                count += 1
    
    assert has_gradients, "Градиенты не проходят через модель!"
    print()


def main():
    """Запуск всех тестов."""
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ GroupShuffleGNN МОДЕЛИ")
    print("="*60 + "\n")
    
    try:
        test_model_creation()
        test_forward_pass()
        test_predict()
        test_orthogonality()
        test_gradient_flow()
        
        print("="*60)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("="*60)
        
    except Exception as e:
        print("="*60)
        print(f"❌ ОШИБКА: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

