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
    min_item_interactions: int = 10,
    iterative: bool = True,
    max_iters: int = 50
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
        iterative: использовать ли итеративную k-core фильтрацию
        max_iters: максимальное число итераций при iterative=True
    
    Returns:
        Отфильтрованный DataFrame
    """
    print(f"Начальная статистика: {len(df)} взаимодействий, "
          f"{df[user_col].nunique()} пользователей, {df[item_col].nunique()} айтемов")
    
    df_filtered = df.copy()
    iters = 0
    while True:
        iters += 1
        # Подсчитываем количество взаимодействий для каждого пользователя
        user_counts = df_filtered[user_col].value_counts()
        # Оставляем только пользователей с достаточным количеством взаимодействий
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df_filtered = df_filtered[df_filtered[user_col].isin(valid_users)].copy()

        # Подсчитываем количество взаимодействий для каждого айтема
        item_counts = df_filtered[item_col].value_counts()
        # Оставляем только айтемы с достаточным количеством взаимодействий
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df_filtered = df_filtered[df_filtered[item_col].isin(valid_items)].copy()

        print(f"Итерация {iters}: {len(df_filtered)} взаимодействий, "
              f"{df_filtered[user_col].nunique()} пользователей, {df_filtered[item_col].nunique()} айтемов")

        if not iterative:
            break
        if iters >= max_iters:
            print("Предупреждение: достигнут лимит итераций фильтрации.")
            break

        # Проверяем сходимость: если ни пользователи, ни айтемы не меняются
        new_user_count = df_filtered[user_col].nunique()
        new_item_count = df_filtered[item_col].nunique()
        if new_user_count == len(valid_users) and new_item_count == len(valid_items):
            break

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
    
    Особенности:
    - Для Book-Crossing: рейтинг 0 означает implicit feedback (пользователь прочитал, но не оценил)
    - Для других датасетов: рейтинг 0 обычно означает отсутствие взаимодействия
    
    Args:
        df: DataFrame с взаимодействиями
        rating_col: название колонки с рейтингами (если None, все взаимодействия = 1)
        threshold: порог рейтинга для бинаризации (если рейтинг >= threshold, то 1, иначе 0)
    
    Returns:
        DataFrame с бинаризованными взаимодействиями (колонка rating удалена, остаются только положительные)
    """
    df = df.copy()
    
    if rating_col and rating_col in df.columns:
        # Если есть рейтинги, фильтруем по порогу
        # Обычно threshold = 0 означает, что любое взаимодействие = положительное
        # Для Book-Crossing: рейтинг 0 тоже считается положительным (implicit feedback)
        
        # Проверяем наличие NaN значений в рейтингах
        nan_count = df[rating_col].isna().sum()
        if nan_count > 0:
            print(f"  Найдено {nan_count} взаимодействий без рейтинга (NaN)")
            print(f"  Они будут сохранены как implicit feedback (положительные взаимодействия)")
        
        # Оставляем взаимодействия с рейтингом >= threshold
        # Для NaN значений: если threshold = 0, сохраняем их как implicit feedback
        # Для threshold > 0: NaN значения будут исключены (так как NaN >= threshold = False)
        if threshold == 0.0:
            # При threshold = 0 сохраняем все, включая NaN (как implicit feedback)
            # NaN >= 0 возвращает False, поэтому используем fillna для сохранения NaN значений
            df = df[(df[rating_col] >= threshold) | (df[rating_col].isna())].copy()
        else:
            # При threshold > 0 исключаем NaN (они не проходят порог)
            df = df[df[rating_col] >= threshold].copy()
        
        # Удаляем колонку рейтинга, так как она больше не нужна
        df = df.drop(columns=[rating_col])
    else:
        # Если рейтингов нет, все взаимодействия считаются положительными (implicit feedback)
        # Это нормально для датасетов типа Gowalla, где есть только check-ins без рейтингов
        print(f"  Рейтинги отсутствуют - все взаимодействия считаются положительными (implicit feedback)")
        pass
    
    print(f"Бинаризация завершена: {len(df)} положительных взаимодействий")
    
    return df


def normalize_ids(
    df: pd.DataFrame,
    user_col: str = 'userId',
    item_col: str = 'itemId'
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Нормализует ID пользователей и айтемов, приводя их к последовательным числам от 0.
    
    Это необходимо, потому что исходные ID могут быть не последовательными
    (например, 1, 5, 100, 1000), а для работы с embeddings нужны индексы 0, 1, 2, 3...
    
    Поддерживает как числовые, так и строковые ID (например, ISBN в Book-Crossing).
    
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
    
    # Получаем уникальные ID (могут быть числами или строками)
    unique_users = sorted(df[user_col].unique())
    unique_items = sorted(df[item_col].unique())
    
    # Создаем маппинги: старый_id -> новый_id (начиная с 0)
    # Поддерживаем любые типы ID (int, str, и т.д.)
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
    timestamp_col: Optional[str] = None,
    keep: str = 'last'
) -> pd.DataFrame:
    """
    Удаляет дубликаты взаимодействий (один пользователь - один айтем).
    
    Args:
        df: DataFrame с взаимодействиями
        user_col: название колонки с ID пользователей
        item_col: название колонки с ID айтемов
        timestamp_col: название колонки с временной меткой (если есть)
        keep: какой дубликат оставить ('first', 'last', или False для удаления всех)
    
    Returns:
        DataFrame без дубликатов
    """
    initial_count = len(df)
    df = df.copy()

    # Если есть timestamp, считаем последнюю запись наиболее корректной
    if timestamp_col and timestamp_col in df.columns:
        df = df.sort_values(timestamp_col)

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
    num_interactions = len(df)
    num_users = df[user_col].nunique() if not df.empty else 0
    num_items = df[item_col].nunique() if not df.empty else 0
    
    # Вычисляем статистику с проверкой на деление на ноль
    if num_users > 0 and num_items > 0:
        sparsity = 1 - num_interactions / (num_users * num_items)
        avg_interactions_per_user = num_interactions / num_users
        avg_interactions_per_item = num_interactions / num_items
    else:
        # Если нет пользователей или айтемов, устанавливаем значения по умолчанию
        sparsity = 1.0
        avg_interactions_per_user = 0.0
        avg_interactions_per_item = 0.0
    
    stats = {
        'num_interactions': num_interactions,
        'num_users': num_users,
        'num_items': num_items,
        'sparsity': sparsity,
        'avg_interactions_per_user': avg_interactions_per_user,
        'avg_interactions_per_item': avg_interactions_per_item,
    }
    
    return stats

