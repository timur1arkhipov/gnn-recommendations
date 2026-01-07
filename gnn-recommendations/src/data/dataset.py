"""
Основной класс для работы с датасетами рекомендательных систем.

Поддерживает:
- Загрузку данных из разных форматов (MovieLens, Book-Crossing, Gowalla)
- Препроцессинг (фильтрация, бинаризация, нормализация ID)
- Разделение на train/valid/test (temporal или random split)
- Построение bipartite графов
- Сохранение и загрузку обработанных данных
"""

import os
import pandas as pd
import numpy as np
import yaml
import json
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
import scipy.sparse as sp

from .preprocessing import (
    filter_by_min_interactions,
    binarize_interactions,
    normalize_ids,
    remove_duplicates,
    get_statistics
)
from .graph_builder import (
    build_bipartite_graph,
    normalize_adjacency_matrix,
    save_adjacency_matrix,
    load_adjacency_matrix,
    convert_to_torch_sparse
)
from .loaders import get_loader


class RecommendationDataset:
    """
    Класс для работы с датасетами рекомендательных систем.
    
    Обрабатывает весь pipeline от загрузки сырых данных до построения графов.
    """
    
    def __init__(
        self,
        name: str,
        root_dir: str = ".",
        config_path: Optional[str] = None
    ):
        """
        Инициализация датасета.
        
        Args:
            name: название датасета ('movielens1m', 'amazon_book', 'gowalla')
            root_dir: корневая директория проекта
            config_path: путь к конфигурационному файлу (если None, используется по умолчанию)
        """
        self.name = name
        self.root_dir = Path(root_dir)
        
        # Загружаем конфигурацию
        if config_path is None:
            # Маппинг имен датасетов на имена конфигурационных файлов
            config_name_mapping = {
                'movie_lens': 'movielens1m',
                'movielens1m': 'movielens1m',
                'book_crossing': 'book_crossing',
                'book-crossing': 'book_crossing',
                'gowalla': 'gowalla',
                'yelp2018': 'yelp2018',
            }
            
            # Получаем имя конфига из маппинга или используем исходное имя
            config_name = config_name_mapping.get(name, name)
            config_path = self.root_dir / "config" / "datasets" / f"{config_name}.yaml"
        
        # Проверяем существование файла
        if not config_path.exists():
            raise FileNotFoundError(
                f"Конфигурационный файл не найден: {config_path}\n"
                f"Доступные файлы в {self.root_dir / 'config' / 'datasets'}: "
                f"{list((self.root_dir / 'config' / 'datasets').glob('*.yaml'))}"
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Пути к данным
        self.raw_data_path = self.root_dir / self.config['data_path']
        self.processed_data_path = self.root_dir / "data" / "processed" / name
        self.graphs_path = self.root_dir / "data" / "graphs" / name
        
        # Создаем директории, если их нет
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.graphs_path.mkdir(parents=True, exist_ok=True)
        
        # Данные (будут заполнены в процессе обработки)
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        
        # Статистика
        self.n_users = 0
        self.n_items = 0
        self.user_mapping = {}
        self.item_mapping = {}
        self.stats = {}
        
        # Графы
        self.adj_matrix = None
        self.norm_adj_matrix = None
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Загружает сырые данные из файлов используя соответствующий загрузчик.
        
        Каждый датасет имеет свой загрузчик, который знает формат данных
        и преобразует их в единый формат: userId, itemId, rating (опционально), timestamp (опционально)
        
        Returns:
            DataFrame с колонками: userId, itemId, rating (опционально), timestamp (опционально)
        """
        print(f"\n{'='*60}")
        print(f"Загрузка данных: {self.name}")
        print(f"{'='*60}")
        
        # Получаем соответствующий загрузчик для датасета
        try:
            loader = get_loader(self.name)
        except ValueError as e:
            raise ValueError(
                f"Не удалось найти загрузчик для датасета '{self.name}'\n"
                f"{e}"
            )
        
        # Загружаем данные через загрузчик
        # Загрузчик знает формат конкретного датасета и преобразует его в единый формат
        df = loader.load(self.raw_data_path)
        
        # Проверяем, что данные загружены и имеют нужные колонки
        if df.empty:
            raise ValueError(
                f"Данные не загружены или пусты для датасета {self.name}!\n"
                f"Проверьте наличие файлов в: {self.raw_data_path}"
            )
        
        if 'userId' not in df.columns or 'itemId' not in df.columns:
            raise ValueError(
                f"Загруженные данные не содержат нужных колонок!\n"
                f"Ожидаются: userId, itemId\n"
                f"Найдены: {list(df.columns)}"
            )
        
        # Сохраняем сырые данные
        self.raw_data = df
        
        print(f"Загружено: {len(df)} строк")
        print(f"Колонки: {list(df.columns)}")
        print(f"Пользователей: {df['userId'].nunique()}")
        print(f"Айтемов: {df['itemId'].nunique()}")
        
        return df
    
    def preprocess(
        self,
        min_user_interactions: Optional[int] = None,
        min_item_interactions: Optional[int] = None,
        rating_threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Препроцессинг данных: фильтрация, бинаризация, нормализация ID.
        
        Args:
            min_user_interactions: минимальное количество взаимодействий для пользователя
            min_item_interactions: минимальное количество взаимодействий для айтема
            rating_threshold: порог рейтинга для бинаризации
        
        Returns:
            Обработанный DataFrame
        """
        print(f"\n{'='*60}")
        print(f"Препроцессинг данных: {self.name}")
        print(f"{'='*60}")
        
        if self.raw_data is None:
            self.load_raw_data()
        
        df = self.raw_data.copy()
        
        # Проверяем, что данные не пустые
        if df.empty:
            raise ValueError(
                f"Данные пусты! Проверьте, что файлы с данными существуют и содержат информацию.\n"
                f"Для {self.name} проверьте наличие файлов в: {self.raw_data_path}"
            )
        
        # Проверяем наличие необходимых колонок
        if 'userId' not in df.columns or 'itemId' not in df.columns:
            raise ValueError(
                f"DataFrame должен содержать колонки 'userId' и 'itemId'.\n"
                f"Найденные колонки: {list(df.columns)}"
            )
        
        # 1. Удаляем дубликаты
        print("\n1. Удаление дубликатов...")
        df = remove_duplicates(df)
        
        # 2. Фильтрация по минимальному количеству взаимодействий
        print("\n2. Фильтрация...")
        min_user = min_user_interactions or self.config['filtering']['min_user_interactions']
        min_item = min_item_interactions or self.config['filtering']['min_item_interactions']
        
        df = filter_by_min_interactions(
            df,
            min_user_interactions=min_user,
            min_item_interactions=min_item
        )
        
        # Проверяем, что после фильтрации остались данные
        if df.empty:
            raise ValueError(
                f"После фильтрации данных не осталось!\n"
                f"Попробуйте уменьшить min_user_interactions или min_item_interactions.\n"
                f"Текущие значения: min_user={min_user}, min_item={min_item}"
            )
        
        # 3. Бинаризация (implicit feedback)
        print("\n3. Бинаризация взаимодействий...")
        rating_col = 'rating' if 'rating' in df.columns else None
        df = binarize_interactions(df, rating_col=rating_col, threshold=rating_threshold)
        
        # Проверяем, что после бинаризации остались данные
        if df.empty:
            raise ValueError(
                f"После бинаризации данных не осталось!\n"
                f"Возможно, все рейтинги были ниже порога threshold={rating_threshold}."
            )
        
        # 4. Нормализация ID (приведение к последовательным числам от 0)
        print("\n4. Нормализация ID...")
        df, self.user_mapping, self.item_mapping = normalize_ids(df)
        
        # Сохраняем обработанные данные
        self.processed_data = df
        self.n_users = df['userId'].nunique()
        self.n_items = df['itemId'].nunique()
        
        # Вычисляем статистику
        self.stats = get_statistics(df)
        
        print(f"\n{'='*60}")
        print("Препроцессинг завершен!")
        print(f"Пользователей: {self.n_users}")
        print(f"Айтемов: {self.n_items}")
        print(f"Взаимодействий: {len(df)}")
        print(f"Разреженность: {self.stats['sparsity']:.4f}")
        print(f"{'='*60}\n")
        
        return df
    
    def split(
        self,
        strategy: str = 'temporal',
        train_ratio: Optional[float] = None,
        valid_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None
    ):
        """
        Разделяет данные на train/valid/test.
        
        Стратегии:
        - 'temporal': разделение по времени (для временных данных)
        - 'random': случайное разделение
        
        Args:
            strategy: стратегия разделения
            train_ratio: доля обучающей выборки
            valid_ratio: доля валидационной выборки
            test_ratio: доля тестовой выборки
        """
        print(f"\n{'='*60}")
        print(f"Разделение данных: {strategy}")
        print(f"{'='*60}")
        
        if self.processed_data is None:
            self.preprocess()
        
        df = self.processed_data.copy()
        
        # Получаем параметры из конфига, если не указаны
        split_config = self.config['split']
        train_ratio = train_ratio or split_config['train_ratio']
        valid_ratio = valid_ratio or split_config['valid_ratio']
        test_ratio = test_ratio or split_config['test_ratio']
        
        # Проверяем, что сумма = 1.0
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
            "Сумма долей должна быть равна 1.0"
        
        if strategy == 'temporal':
            # Разделение по времени (если есть timestamp)
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
                n = len(df)
                train_end = int(n * train_ratio)
                valid_end = int(n * (train_ratio + valid_ratio))
                
                self.train_data = df.iloc[:train_end].copy()
                self.valid_data = df.iloc[train_end:valid_end].copy()
                self.test_data = df.iloc[valid_end:].copy()
            else:
                # Если нет timestamp, используем случайное разделение
                print("Внимание: timestamp не найден, используется случайное разделение")
                strategy = 'random'
        
        if strategy == 'random':
            # Случайное разделение
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            n = len(df)
            train_end = int(n * train_ratio)
            valid_end = int(n * (train_ratio + valid_ratio))
            
            self.train_data = df.iloc[:train_end].copy()
            self.valid_data = df.iloc[train_end:valid_end].copy()
            self.test_data = df.iloc[valid_end:].copy()
        
        print(f"Train: {len(self.train_data)} взаимодействий")
        print(f"Valid: {len(self.valid_data)} взаимодействий")
        print(f"Test: {len(self.test_data)} взаимодействий")
        
        # Сохраняем разделенные данные
        self._save_split_data()
    
    def _save_split_data(self):
        """Сохраняет разделенные данные в файлы."""
        # Формат: userId itemId (по одному взаимодействию на строку)
        train_file = self.processed_data_path / "train.txt"
        valid_file = self.processed_data_path / "valid.txt"
        test_file = self.processed_data_path / "test.txt"
        
        self.train_data[['userId', 'itemId']].to_csv(
            train_file, sep='\t', index=False, header=False
        )
        self.valid_data[['userId', 'itemId']].to_csv(
            valid_file, sep='\t', index=False, header=False
        )
        self.test_data[['userId', 'itemId']].to_csv(
            test_file, sep='\t', index=False, header=False
        )
        
        # Сохраняем статистику
        stats_file = self.processed_data_path / "stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                'n_users': self.n_users,
                'n_items': self.n_items,
                'n_interactions': len(self.processed_data),
                'train_size': len(self.train_data),
                'valid_size': len(self.valid_data),
                'test_size': len(self.test_data),
                **self.stats
            }, f, indent=2)
        
        print(f"\nДанные сохранены в: {self.processed_data_path}")
    
    def build_graph(
        self,
        normalize: Optional[bool] = None,
        self_loop: Optional[bool] = None,
        normalization_type: str = 'symmetric'
    ) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
        """
        Строит bipartite граф из обучающих данных.
        
        Args:
            normalize: нормализовать ли граф
            self_loop: добавлять ли self-loops
            normalization_type: тип нормализации ('symmetric', 'row', 'none')
        
        Returns:
            Tuple из (adj_matrix, norm_adj_matrix)
        """
        print(f"\n{'='*60}")
        print(f"Построение графа: {self.name}")
        print(f"{'='*60}")
        
        if self.train_data is None:
            self.split()
        
        # Получаем параметры из конфига
        graph_config = self.config['graph']
        normalize = normalize if normalize is not None else graph_config['normalize']
        self_loop = self_loop if self_loop is not None else graph_config['self_loop']
        
        # Строим граф из обучающих данных
        self.adj_matrix = build_bipartite_graph(
            self.train_data,
            n_users=self.n_users,
            n_items=self.n_items,
            self_loop=self_loop
        )
        
        # Нормализуем, если нужно
        if normalize:
            self.norm_adj_matrix = normalize_adjacency_matrix(
                self.adj_matrix,
                normalization=normalization_type
            )
        else:
            self.norm_adj_matrix = self.adj_matrix
        
        # Сохраняем графы
        adj_file = self.graphs_path / "adj_matrix.npz"
        norm_adj_file = self.graphs_path / "norm_adj_matrix.npz"
        
        save_adjacency_matrix(self.adj_matrix, str(adj_file))
        save_adjacency_matrix(self.norm_adj_matrix, str(norm_adj_file))
        
        print(f"Графы сохранены в: {self.graphs_path}")
        
        return self.adj_matrix, self.norm_adj_matrix
    
    def get_torch_adjacency(self, normalized: bool = True) -> torch.sparse.FloatTensor:
        """
        Возвращает adjacency matrix как PyTorch sparse tensor.
        
        Args:
            normalized: использовать нормализованную матрицу
        
        Returns:
            PyTorch sparse tensor
        """
        if normalized:
            if self.norm_adj_matrix is None:
                self.build_graph()
            adj = self.norm_adj_matrix
        else:
            if self.adj_matrix is None:
                self.build_graph()
            adj = self.adj_matrix
        
        return convert_to_torch_sparse(adj)
    
    def load_processed_data(self):
        """Загружает уже обработанные данные из файлов."""
        train_file = self.processed_data_path / "train.txt"
        valid_file = self.processed_data_path / "valid.txt"
        test_file = self.processed_data_path / "test.txt"
        stats_file = self.processed_data_path / "stats.json"
        
        if not all(f.exists() for f in [train_file, valid_file, test_file, stats_file]):
            raise FileNotFoundError("Обработанные данные не найдены. Запустите preprocess() и split()")
        
        # Загружаем данные
        self.train_data = pd.read_csv(train_file, sep='\t', header=None, names=['userId', 'itemId'])
        self.valid_data = pd.read_csv(valid_file, sep='\t', header=None, names=['userId', 'itemId'])
        self.test_data = pd.read_csv(test_file, sep='\t', header=None, names=['userId', 'itemId'])
        
        # Загружаем статистику
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
            self.n_users = stats['n_users']
            self.n_items = stats['n_items']
            self.stats = stats
        
        # Загружаем графы
        adj_file = self.graphs_path / "adj_matrix.npz"
        norm_adj_file = self.graphs_path / "norm_adj_matrix.npz"
        
        if adj_file.exists():
            self.adj_matrix = load_adjacency_matrix(str(adj_file))
        if norm_adj_file.exists():
            self.norm_adj_matrix = load_adjacency_matrix(str(norm_adj_file))
        
        print(f"Данные загружены из: {self.processed_data_path}")

