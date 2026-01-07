"""
Модуль для препроцессинга данных рекомендательных систем.

Основные функции:
- Фильтрация пользователей и айтемов по минимальному количеству взаимодействий
- Бинаризация взаимодействий (implicit feedback)
- Нормализация идентификаторов
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from collections import Counter


def filter_by_min_interactions(
    df: pd.DataFrame,
    user_col: str = 'userId',
    item_col: str = 'itemId',
    min_user_interactions: int = 10,
    min_item_interactions: int = 10
) -> pd.DataFrame:
    """
    Фильтрует данные, оставляя только пользователей и айтемы с минимальным количеством взаимодействий.
    
    Это важный шаг для удаления "холодных" пользователей и айтемов,
    которые могут ухудшить качество модели.
    
    Args:
        df: DataFrame с колонками userId и itemId
        user_col: название колонки с ID пользователей
        item_col: название колонки с ID айтемов
        min_user_interactions: минимальное количество взаимодействий для пользователя
        min_item_interactions: минимальное количество взаимодействий для айтема
    
    Returns:
        Отфильтрованный DataFrame
    """
    print(f"Начальная статистика: {len(df)} взаимодействий, "
          f"{df[user_col].nunique()} пользователей, {df[item_col].nunique()} айтемов")
    
    # Подсчитываем количество взаимодействий для каждого пользователя
    user_counts = df[user_col].value_counts()
    # Оставляем только пользователей с достаточным количеством взаимодействий
    valid_users = user_counts[user_counts >= min_user_interactions].index
    df_filtered = df[df[user_col].isin(valid_users)].copy()
    
    print(f"После фильтрации пользователей: {len(df_filtered)} взаимодействий, "
          f"{df_filtered[user_col].nunique()} пользователей")
    
    # Подсчитываем количество взаимодействий для каждого айтема
    item_counts = df_filtered[item_col].value_counts()
    # Оставляем только айтемы с достаточным количеством взаимодействий
    valid_items = item_counts[item_counts >= min_item_interactions].index
    df_filtered = df_filtered[df_filtered[item_col].isin(valid_items)].copy()
    
    print(f"После фильтрации айтемов: {len(df_filtered)} взаимодействий, "
          f"{df_filtered[user_col].nunique()} пользователей, {df_filtered[item_col].nunique()} айтемов")
    
    return df_filtered


def binarize_interactions(
    df: pd.DataFrame,
    rating_col: Optional[str] = None,
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Бинаризует взаимодействия (преобразует в implicit feedback).
    
    В рекомендательных системах часто используется implicit feedback:
    если пользователь взаимодействовал с айтемом (даже с низким рейтингом),
    это считается положительным сигналом.
    
    Args:
        df: DataFrame с взаимодействиями
        rating_col: название колонки с рейтингами (если None, все взаимодействия = 1)
        threshold: порог рейтинга для бинаризации (если рейтинг >= threshold, то 1, иначе 0)
    
    Returns:
        DataFrame с бинаризованными взаимодействиями (колонка rating заменена на 1/0)
    """
    df = df.copy()
    
    if rating_col and rating_col in df.columns:
        # Если есть рейтинги, бинаризуем по порогу
        # Обычно threshold = 0 означает, что любое взаимодействие = положительное
        df['rating'] = (df[rating_col] > threshold).astype(int)
        # Удаляем исходную колонку рейтинга
        df = df.drop(columns=[rating_col])
    else:
        # Если рейтингов нет, все взаимодействия = 1 (implicit feedback)
        df['rating'] = 1
    
    # Оставляем только положительные взаимодействия (rating = 1)
    df = df[df['rating'] == 1].drop(columns=['rating'])
    
    print(f"Бинаризация завершена: {len(df)} положительных взаимодействий")
    
    return df


def normalize_ids(
    df: pd.DataFrame,
    user_col: str = 'userId',
    item_col: str = 'itemId'
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Нормализует ID пользователей и айтемов, приводя их к последовательным числам от 0.
    
    Это необходимо, потому что исходные ID могут быть не последовательными
    (например, 1, 5, 100, 1000), а для работы с embeddings нужны индексы 0, 1, 2, 3...
    
    Args:
        df: DataFrame с колонками userId и itemId
        user_col: название колонки с ID пользователей
        item_col: название колонка с ID айтемов
    
    Returns:
        Tuple из:
        - DataFrame с нормализованными ID
        - Словарь маппинга: исходный_user_id -> новый_user_id
        - Словарь маппинга: исходный_item_id -> новый_item_id
    """
    df = df.copy()
    
    # Получаем уникальные ID
    unique_users = sorted(df[user_col].unique())
    unique_items = sorted(df[item_col].unique())
    
    # Создаем маппинги: старый_id -> новый_id (начиная с 0)
    user_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    
    # Применяем маппинги
    df[user_col] = df[user_col].map(user_mapping)
    df[item_col] = df[item_col].map(item_mapping)
    
    print(f"Нормализация ID: {len(unique_users)} пользователей, {len(unique_items)} айтемов")
    
    return df, user_mapping, item_mapping


def remove_duplicates(
    df: pd.DataFrame,
    user_col: str = 'userId',
    item_col: str = 'itemId',
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Удаляет дубликаты взаимодействий (один пользователь - один айтем).
    
    Args:
        df: DataFrame с взаимодействиями
        user_col: название колонки с ID пользователей
        item_col: название колонки с ID айтемов
        keep: какой дубликат оставить ('first', 'last', или False для удаления всех)
    
    Returns:
        DataFrame без дубликатов
    """
    initial_count = len(df)
    df = df.drop_duplicates(subset=[user_col, item_col], keep=keep)
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"Удалено {removed} дубликатов взаимодействий")
    
    return df


def get_statistics(df: pd.DataFrame, user_col: str = 'userId', item_col: str = 'itemId') -> Dict:
    """
    Вычисляет статистику по датасету.
    
    Args:
        df: DataFrame с взаимодействиями
        user_col: название колонки с ID пользователей
        item_col: название колонки с ID айтемов
    
    Returns:
        Словарь со статистикой
    """
    stats = {
        'num_interactions': len(df),
        'num_users': df[user_col].nunique(),
        'num_items': df[item_col].nunique(),
        'sparsity': 1 - len(df) / (df[user_col].nunique() * df[item_col].nunique()),
        'avg_interactions_per_user': len(df) / df[user_col].nunique(),
        'avg_interactions_per_item': len(df) / df[item_col].nunique(),
    }
    
    return stats

