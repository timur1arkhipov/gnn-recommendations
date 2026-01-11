"""
Скрипт для генерации case study - качественного анализа рекомендаций.

Показывает примеры рекомендаций для конкретных пользователей
и сравнивает результаты разных моделей.
"""

import sys
from pathlib import Path
import argparse
import json
import torch
import pandas as pd
from typing import Dict, List

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import RecommendationDataset
from scripts.run_all_experiments import create_model, load_config


def load_trained_model(
    model_name: str,
    dataset_name: str,
    root_dir: Path
):
    """
    Загружает обученную модель из чекпоинта.
    
    Args:
        model_name: название модели
        dataset_name: название датасета
        root_dir: корневая директория проекта
    
    Returns:
        Загруженная модель
    """
    # Загружаем датасет для получения n_users, n_items
    dataset = RecommendationDataset(name=dataset_name, root_dir=str(root_dir))
    dataset.load_processed_data()
    
    # Создаём модель
    model = create_model(
        model_name=model_name,
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        root_dir=root_dir
    )
    
    # Загружаем чекпоинт
    checkpoint_dir = root_dir / "results" / "checkpoints" / model_name / dataset_name
    checkpoint_file = checkpoint_dir / "best_model.pt"
    
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Модель загружена: {checkpoint_file}")
    else:
        print(f"⚠️  Чекпоинт не найден: {checkpoint_file}")
        print("   Используется необученная модель")
    
    return model, dataset


def get_top_k_recommendations(
    model,
    dataset: RecommendationDataset,
    user_id: int,
    k: int = 10,
    exclude_train: bool = True
) -> List[int]:
    """
    Получает топ-K рекомендаций для пользователя.
    
    Args:
        model: обученная модель
        dataset: датасет
        user_id: ID пользователя
        k: количество рекомендаций
        exclude_train: исключать ли train items
    
    Returns:
        Список ID айтемов
    """
    model.eval()
    
    with torch.no_grad():
        # Получаем embeddings
        adj_matrix = dataset.get_torch_adjacency(normalized=True)
        
        if hasattr(model, 'get_all_embeddings'):
            user_emb, item_emb = model.get_all_embeddings(adj_matrix)
        else:
            user_emb, item_emb = model(adj_matrix)
        
        # Вычисляем scores для пользователя
        user_embedding = user_emb[user_id]  # [embedding_dim]
        scores = user_embedding @ item_emb.T  # [n_items]
        
        # Исключаем train items
        if exclude_train:
            train_items = dataset.train_data[dataset.train_data['userId'] == user_id]['itemId'].values
            scores[train_items] = float('-inf')
        
        # Топ-K
        _, top_k_items = torch.topk(scores, k)
        
        return top_k_items.cpu().numpy().tolist()


def generate_case_study_for_user(
    user_id: int,
    models: Dict[str, any],
    dataset: RecommendationDataset,
    k: int = 10
) -> Dict:
    """
    Генерирует case study для одного пользователя.
    
    Args:
        user_id: ID пользователя
        models: словарь {model_name: model}
        dataset: датасет
        k: количество рекомендаций
    
    Returns:
        Словарь с рекомендациями
    """
    # Получаем историю пользователя
    train_items = dataset.train_data[dataset.train_data['userId'] == user_id]['itemId'].values.tolist()
    test_items = dataset.test_data[dataset.test_data['userId'] == user_id]['itemId'].values.tolist()
    
    # Получаем рекомендации от каждой модели
    recommendations = {}
    for model_name, model in models.items():
        top_k = get_top_k_recommendations(model, dataset, user_id, k)
        recommendations[model_name] = top_k
    
    return {
        'user_id': user_id,
        'train_items': train_items,
        'test_items': test_items,
        'recommendations': recommendations
    }


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Генерация case study - качественный анализ рекомендаций"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default='movie_lens',
        choices=['movie_lens', 'book_crossing', 'gowalla'],
        help="Датасет"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=['bpr_mf', 'lightgcn', 'groupshuffle_gnn'],
        help="Модели для сравнения"
    )
    
    parser.add_argument(
        "--user_ids",
        type=int,
        nargs='+',
        default=None,
        help="ID пользователей для анализа (если None, выбираются случайно)"
    )
    
    parser.add_argument(
        "--n_users",
        type=int,
        default=5,
        help="Количество пользователей для анализа (если user_ids не указаны)"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Количество рекомендаций"
    )
    
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Корневая директория проекта"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Файл для сохранения результатов"
    )
    
    args = parser.parse_args()
    
    # Определяем root_dir
    if args.root_dir is None:
        root_dir = project_root
    else:
        root_dir = Path(args.root_dir)
    
    # Определяем output_file
    if args.output_file is None:
        output_file = root_dir / "results" / "case_study" / f"{args.dataset}_case_study.json"
    else:
        output_file = Path(args.output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("ГЕНЕРАЦИЯ CASE STUDY")
    print(f"{'='*80}")
    print(f"Датасет: {args.dataset}")
    print(f"Модели: {', '.join(args.models)}")
    print(f"Количество рекомендаций: {args.k}")
    print(f"{'='*80}\n")
    
    # Загружаем модели
    print("Загрузка моделей...")
    models = {}
    dataset = None
    
    for model_name in args.models:
        try:
            model, ds = load_trained_model(model_name, args.dataset, root_dir)
            models[model_name] = model
            if dataset is None:
                dataset = ds
        except Exception as e:
            print(f"❌ Ошибка при загрузке {model_name}: {e}")
    
    if not models:
        print("❌ Не удалось загрузить ни одной модели!")
        return
    
    # Выбираем пользователей
    if args.user_ids is not None:
        user_ids = args.user_ids
    else:
        # Выбираем случайных пользователей с достаточным количеством взаимодействий
        user_counts = dataset.train_data['userId'].value_counts()
        active_users = user_counts[user_counts >= 10].index.tolist()
        import random
        random.seed(42)
        user_ids = random.sample(active_users, min(args.n_users, len(active_users)))
    
    print(f"\nВыбрано пользователей: {len(user_ids)}")
    print(f"User IDs: {user_ids}\n")
    
    # Генерируем case study
    case_studies = []
    
    for user_id in user_ids:
        print(f"\n{'-'*80}")
        print(f"Пользователь {user_id}")
        print(f"{'-'*80}")
        
        case_study = generate_case_study_for_user(
            user_id=user_id,
            models=models,
            dataset=dataset,
            k=args.k
        )
        
        case_studies.append(case_study)
        
        # Выводим результаты
        print(f"Train items ({len(case_study['train_items'])}): {case_study['train_items'][:5]}...")
        print(f"Test items ({len(case_study['test_items'])}): {case_study['test_items']}")
        print("\nРекомендации:")
        
        for model_name, recommendations in case_study['recommendations'].items():
            # Проверяем попадания
            hits = set(recommendations) & set(case_study['test_items'])
            print(f"  {model_name:20s}: {recommendations[:5]}... (hits: {len(hits)})")
    
    # Сохраняем результаты
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(case_studies, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Case study сохранён: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

