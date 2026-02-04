"""
Модуль для построения графов из взаимодействий пользователей и айтемов.

Создает bipartite граф, где:
- Узлы: пользователи и айтемы
- Ребра: взаимодействия между пользователями и айтемами
"""

import numpy as np
import scipy.sparse as sp
import torch
from typing import Tuple, Optional
import pandas as pd


def build_bipartite_graph(
    interactions: pd.DataFrame,
    n_users: int,
    n_items: int,
    user_col: str = 'userId',
    item_col: str = 'itemId',
    self_loop: bool = False
) -> sp.coo_matrix:
    """
    Строит bipartite граф из взаимодействий пользователей и айтемов.
    
    Bipartite граф - это граф, где узлы разделены на два множества:
    пользователи и айтемы. Ребра могут быть только между узлами из разных множеств.
    
    Структура adjacency matrix:
    [0          R]
    [R^T        0]
    
    где R - матрица взаимодействий пользователей и айтемов (n_users x n_items)
    
    Args:
        interactions: DataFrame с колонками userId и itemId
        n_users: количество пользователей
        n_items: количество айтемов
        user_col: название колонки с ID пользователей
        item_col: название колонки с ID айтемов
        self_loop: добавлять ли self-loops (обычно False для bipartite графа)
    
    Returns:
        Sparse матрица смежности в формате COO (Coordinate format)
        Размерность: (n_users + n_items) x (n_users + n_items)
    """
    # Векторизованная сборка ребер для ускорения
    user_ids = interactions[user_col].to_numpy(dtype=np.int64, copy=False)
    item_ids = interactions[item_col].to_numpy(dtype=np.int64, copy=False)

    # Ребра от пользователей к айтемам
    rows_ui = user_ids
    cols_ui = n_users + item_ids

    # Ребра от айтемов к пользователям (для неориентированного графа)
    rows_iu = n_users + item_ids
    cols_iu = user_ids

    rows = np.concatenate([rows_ui, rows_iu])
    cols = np.concatenate([cols_ui, cols_iu])
    data = np.ones(len(rows), dtype=np.float32)
    
    # Создаем sparse матрицу в формате COO
    # COO (Coordinate) - самый простой формат для создания sparse матриц
    adj_matrix = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_users + n_items, n_users + n_items),
        dtype=np.float32
    )
    
    # Добавляем self-loops, если нужно
    if self_loop:
        adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0], dtype=np.float32)
    
    print(f"Построен bipartite граф: {adj_matrix.shape[0]} узлов, "
          f"{len(interactions)} взаимодействий, "
          f"{adj_matrix.nnz} ненулевых элементов")
    
    return adj_matrix


def normalize_adjacency_matrix(
    adj: sp.coo_matrix,
    normalization: str = 'symmetric'
) -> sp.coo_matrix:
    """
    Нормализует adjacency matrix для использования в GCN.
    
    Нормализация важна для стабильности обучения GCN.
    Основные методы:
    - 'symmetric': D^(-1/2) * A * D^(-1/2) - симметричная нормализация
    - 'row': D^(-1) * A - нормализация по строкам
    - 'none': без нормализации
    
    Args:
        adj: Sparse adjacency matrix в формате COO
        normalization: тип нормализации ('symmetric', 'row', 'none')
    
    Returns:
        Нормализованная sparse матрица в формате COO
    """
    if normalization == 'none':
        return adj
    
    # Преобразуем в CSR формат для эффективных операций
    adj = adj.tocsr()
    
    # Вычисляем степени узлов (degree)
    # Degree узла = количество его соседей
    degrees = np.array(adj.sum(axis=1)).flatten()
    
    # Избегаем деления на ноль (для изолированных узлов)
    degrees = np.maximum(degrees, 1.0)
    
    if normalization == 'symmetric':
        # Симметричная нормализация: D^(-1/2) * A * D^(-1/2)
        # Это стандартная нормализация для GCN
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        
        # Создаем диагональную матрицу D^(-1/2)
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # Применяем нормализацию
        adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
    elif normalization == 'row':
        # Нормализация по строкам: D^(-1) * A
        d_inv = np.power(degrees, -1.0)
        d_inv[np.isinf(d_inv)] = 0.0
        
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv @ adj
        
    else:
        raise ValueError(f"Неизвестный тип нормализации: {normalization}")
    
    # Преобразуем обратно в COO формат
    adj_normalized = adj_normalized.tocoo()
    
    print(f"Нормализация применена: {normalization}")
    
    return adj_normalized


def convert_to_torch_sparse(adj: sp.coo_matrix) -> torch.sparse.FloatTensor:
    """
    Преобразует scipy sparse матрицу в PyTorch sparse tensor.
    
    PyTorch требует специальный формат для sparse тензоров:
    - indices: координаты ненулевых элементов
    - values: значения ненулевых элементов
    - size: размерность матрицы
    
    Args:
        adj: Sparse матрица в формате COO
    
    Returns:
        PyTorch sparse tensor
    """
    # Получаем координаты и значения ненулевых элементов
    indices = np.vstack([adj.row, adj.col])
    values = adj.data
    
    # Создаем PyTorch sparse tensor
    # LongTensor для индексов, FloatTensor для значений
    indices_tensor = torch.LongTensor(indices)
    values_tensor = torch.FloatTensor(values)
    size = torch.Size(adj.shape)
    
    sparse_tensor = torch.sparse_coo_tensor(indices_tensor, values_tensor, size)
    
    return sparse_tensor


def save_adjacency_matrix(
    adj: sp.coo_matrix,
    filepath: str,
    format: str = 'npz'
):
    """
    Сохраняет adjacency matrix в файл.
    
    Args:
        adj: Sparse матрица
        filepath: путь к файлу
        format: формат сохранения ('npz' для scipy.sparse, 'pt' для PyTorch)
    """
    if format == 'npz':
        # Сохраняем в формате .npz (сжатый numpy формат)
        sp.save_npz(filepath, adj)
        print(f"Adjacency matrix сохранена: {filepath}")
    elif format == 'pt':
        # Сохраняем как PyTorch tensor
        torch_adj = convert_to_torch_sparse(adj)
        torch.save(torch_adj, filepath)
        print(f"Adjacency matrix сохранена как PyTorch tensor: {filepath}")
    else:
        raise ValueError(f"Неизвестный формат: {format}")


def load_adjacency_matrix(
    filepath: str,
    format: str = 'npz'
) -> sp.coo_matrix:
    """
    Загружает adjacency matrix из файла.
    
    Args:
        filepath: путь к файлу
        format: формат файла ('npz' или 'pt')
    
    Returns:
        Sparse матрица в формате COO
    """
    if format == 'npz':
        adj = sp.load_npz(filepath)
        print(f"Adjacency matrix загружена: {filepath}")
        return adj
    elif format == 'pt':
        torch_adj = torch.load(filepath)
        # Преобразуем PyTorch sparse tensor обратно в scipy sparse
        indices = torch_adj.indices().numpy()
        values = torch_adj.values().numpy()
        shape = torch_adj.shape
        adj = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
        print(f"Adjacency matrix загружена из PyTorch tensor: {filepath}")
        return adj
    else:
        raise ValueError(f"Неизвестный формат: {format}")

