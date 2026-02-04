"""
Модуль с адаптерами для загрузки разных датасетов.

Каждый адаптер знает, как загрузить конкретный датасет и преобразовать его
в единый формат: userId, itemId, rating (опционально), timestamp (опционально).

После загрузки все данные имеют одинаковую структуру, поэтому остальной
функционал (препроцессинг, построение графов) работает одинаково для всех датасетов.
"""

import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod


class BaseDatasetLoader(ABC):
    """
    Базовый класс для загрузчиков датасетов.
    
    Все загрузчики должны преобразовывать данные в единый формат:
    - userId: ID пользователя
    - itemId: ID айтема
    - rating: рейтинг (опционально)
    - timestamp: временная метка (опционально)
    """
    
    @abstractmethod
    def load(self, data_path: Path) -> pd.DataFrame:
        """
        Загружает данные из файлов и преобразует в единый формат.
        
        Args:
            data_path: путь к директории с данными
        
        Returns:
            DataFrame с колонками: userId, itemId, rating (опционально), timestamp (опционально)
        """
        pass
    
    def _normalize_columns(self, df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
        """
        Нормализует колонки DataFrame согласно маппингу.
        
        Args:
            df: исходный DataFrame
            column_mapping: словарь {новое_имя: старое_имя}
        
        Returns:
            DataFrame с нормализованными колонками
        """
        df = df.copy()
        
        # Переименовываем колонки
        for new_name, old_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Оставляем только нужные колонки
        required_cols = ['userId', 'itemId']
        optional_cols = ['rating', 'timestamp']
        
        cols_to_keep = required_cols + [col for col in optional_cols if col in df.columns]
        df = df[cols_to_keep]
        
        return df


class MovieLensLoader(BaseDatasetLoader):
    """Загрузчик для датасета MovieLens (1M, 100k и др.)."""

    def load(self, data_path: Path) -> pd.DataFrame:
        """
        Загружает данные MovieLens.

        Поддерживаемые форматы:
        - ratings.csv с колонками userId, movieId, rating, timestamp
        - ratings.dat (MovieLens-1M) с разделителем "::"
        - u.data (MovieLens-100k) с разделителем табуляция
        """
        # Ищем файл ratings.csv
        ratings_file = data_path / "ratings.csv"
        if not ratings_file.exists():
            # Пробуем альтернативные пути
            ratings_file = data_path.parent / "movie_lens" / "ratings.csv"
        if not ratings_file.exists():
            ratings_file = data_path.parent.parent / "data" / "raw" / "movie_lens" / "ratings.csv"

        df = pd.DataFrame()
        if ratings_file.exists():
            # Загружаем данные
            df = pd.read_csv(ratings_file)
        else:
            # Пробуем MovieLens-1M формат (ratings.dat)
            ratings_dat = data_path / "ratings.dat"
            if not ratings_dat.exists():
                ratings_dat = data_path.parent / "movie_lens" / "ratings.dat"
            if not ratings_dat.exists():
                ratings_dat = data_path.parent.parent / "data" / "raw" / "movie_lens" / "ratings.dat"

            if ratings_dat.exists():
                df = pd.read_csv(
                    ratings_dat,
                    sep="::",
                    engine="python",
                    names=["userId", "movieId", "rating", "timestamp"]
                )
            else:
                # Пробуем MovieLens-100k формат (u.data)
                udata_file = data_path / "u.data"
                if not udata_file.exists():
                    udata_file = data_path.parent / "ml-100k" / "u.data"
                if not udata_file.exists():
                    udata_file = data_path.parent.parent / "data" / "raw" / "ml-100k" / "u.data"

                if udata_file.exists():
                    df = pd.read_csv(
                        udata_file,
                        sep="\t",
                        names=["userId", "movieId", "rating", "timestamp"]
                    )
                else:
                    raise FileNotFoundError(
                        f"Файл с рейтингами не найден в {data_path}\n"
                        f"Искали: ratings.csv, ratings.dat, u.data"
                    )

        # Нормализуем колонки: movieId -> itemId
        df = self._normalize_columns(df, {'itemId': 'movieId'})

        print(f"Загружено {len(df)} взаимодействий из MovieLens")

        return df


class FacebookLoader(BaseDatasetLoader):
    """Загрузчик для датасета Facebook."""

    def load(self, data_path: Path) -> pd.DataFrame:
        """
        Загружает данные Facebook.

        Формат: dataset_facebook.tsv
        - userId itemId rating (разделитель - табуляция, без timestamp)
        """
        # Ищем файл dataset_facebook.tsv
        facebook_file = data_path / "dataset_facebook.tsv"
        if not facebook_file.exists():
            facebook_file = data_path.parent / "facebook" / "dataset_facebook.tsv"
        if not facebook_file.exists():
            facebook_file = data_path.parent.parent / "data" / "raw" / "dataset_facebook.tsv"

        if not facebook_file.exists():
            raise FileNotFoundError(
                f"Файл dataset_facebook.tsv не найден в {data_path}"
            )

        print(f"Загрузка данных Facebook из {facebook_file.name}...")

        # Загружаем TSV файл
        df = pd.read_csv(
            facebook_file,
            sep='\t',
            header=None,
            names=['userId', 'itemId', 'rating']
        )

        print(f"Загружено {len(df)} взаимодействий из Facebook")
        print(f"Уникальных пользователей: {df['userId'].nunique()}")
        print(f"Уникальных айтемов: {df['itemId'].nunique()}")
        if 'rating' in df.columns:
            print(f"Диапазон весов: {df['rating'].min()} - {df['rating'].max()}")

        return df


class YahooLoader(BaseDatasetLoader):
    """Загрузчик для датасетов Yahoo (Music, Movies)."""

    def load(self, data_path: Path) -> pd.DataFrame:
        """
        Загружает данные Yahoo.
        """
        # Список возможных файлов Yahoo для поиска
        possible_files = [
            "ydata-ymovies-user-movie-ratings-train-v1_0.txt",
            "ydata-ymovies-user-movie-ratings-test-v1_0.txt",
        ]

        # Ищем файлы и собираем найденные
        found_files = []
        for filename in possible_files:
            # Проверяем основной путь
            file_path = data_path / filename
            if file_path.exists():
                found_files.append(file_path)
                continue

            file_path = data_path.parent / "yahoo" / filename
            if file_path.exists():
                found_files.append(file_path)
                continue

            file_path = data_path.parent.parent / "data" / "raw" / "yahoo" / filename
            if file_path.exists():
                found_files.append(file_path)

        if not found_files:
            raise FileNotFoundError(
                f"Файлы Yahoo не найдены в {data_path}\n"
                f"Искали: {possible_files}\n"
                f"Поместите файлы датасета Yahoo в директорию data/raw/yahoo/"
            )

        print(f"Загрузка данных Yahoo...")
        print(f"Найдено файлов: {len(found_files)}")

        # Загружаем все найденные файлы
        dataframes = []
        for file_path in found_files:
            df = self._load_single_file(file_path)
            dataframes.append(df)

        # Объединяем все DataFrames
        if len(dataframes) == 1:
            df = dataframes[0]
        else:
            print(f"\nОбъединение {len(dataframes)} файлов...")
            df = pd.concat(dataframes, ignore_index=True)

        print(f"\nВсего загружено {len(df)} взаимодействий из Yahoo")
        print(f"Уникальных пользователей: {df['userId'].nunique()}")
        print(f"Уникальных айтемов: {df['itemId'].nunique()}")
        print(f"Диапазон рейтингов: {df['rating'].min()} - {df['rating'].max()}")

        return df

    def _load_single_file(self, file_path: Path) -> pd.DataFrame:
        """
        Загружает один файл Yahoo dataset.

        Args:
            file_path: путь к файлу

        Returns:
            DataFrame с данными
        """
        print(f"  Загрузка файла: {file_path.name}...")

        # Определяем разделитель (табуляция или запятая)
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if '\t' in first_line:
                separator = '\t'
            elif ',' in first_line:
                separator = ','
            else:
                separator = None

            # Проверяем количество колонок
            parts = first_line.strip().split(separator)
            num_cols = len(parts)

        # Загружаем данные в зависимости от количества колонок
        if num_cols == 4:
            # Формат: userId, itemId, timestamp, rating
            # (Movies format)
            df = pd.read_csv(
                file_path,
                sep=separator,
                header=None,
                names=['userId', 'itemId', 'timestamp', 'rating'],
                engine='python'
            )
        elif num_cols == 3:
            # Формат: userId, itemId, rating
            # (Music format или другие)
            df = pd.read_csv(
                file_path,
                sep=separator,
                header=None,
                names=['userId', 'itemId', 'rating'],
                engine='python'
            )
        else:
            raise ValueError(f"Неожиданное количество колонок: {num_cols}")

        # Проверяем первую строку - может быть заголовок
        if str(df.iloc[0]['userId']).lower() in ['userid', 'user']:
            df = df.iloc[1:]  # Пропускаем заголовок

        # Преобразуем типы
        df['userId'] = pd.to_numeric(df['userId'], errors='coerce')
        df['itemId'] = pd.to_numeric(df['itemId'], errors='coerce')
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

        # Удаляем строки с некорректными данными
        initial_count = len(df)
        df = df.dropna(subset=['userId', 'itemId', 'rating'])
        if len(df) < initial_count:
            print(f"  Удалено {initial_count - len(df)} строк с некорректными данными")

        print(f"  Загружено {len(df)} взаимодействий")

        return df


# Регистр загрузчиков
LOADER_REGISTRY = {
    'movie_lens': MovieLensLoader,
    'movielens1m': MovieLensLoader,
    'movielens100k': MovieLensLoader,
    'ml-1m': MovieLensLoader,
    'ml-100k': MovieLensLoader,
    'facebook': FacebookLoader,
    'yahoo': YahooLoader,
}


def get_loader(dataset_name: str) -> BaseDatasetLoader:
    """
    Возвращает загрузчик для указанного датасета.
    
    Args:
        dataset_name: название датасета
    
    Returns:
        Экземпляр загрузчика
    """
    loader_class = LOADER_REGISTRY.get(dataset_name)
    if loader_class is None:
        raise ValueError(
            f"Неизвестный датасет: {dataset_name}\n"
            f"Доступные датасеты: {list(LOADER_REGISTRY.keys())}"
        )
    
    return loader_class()
