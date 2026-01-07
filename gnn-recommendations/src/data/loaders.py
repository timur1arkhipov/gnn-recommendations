"""
Модуль с адаптерами для загрузки разных датасетов.

Каждый адаптер знает, как загрузить конкретный датасет и преобразовать его
в единый формат: userId, itemId, rating (опционально), timestamp (опционально).

После загрузки все данные имеют одинаковую структуру, поэтому остальной
функционал (препроцессинг, построение графов) работает одинаково для всех датасетов.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
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
    """Загрузчик для датасета MovieLens."""
    
    def load(self, data_path: Path) -> pd.DataFrame:
        """
        Загружает данные MovieLens.
        
        Формат: ratings.csv с колонками userId, movieId, rating, timestamp
        """
        # Ищем файл ratings.csv
        ratings_file = data_path / "ratings.csv"
        if not ratings_file.exists():
            # Пробуем альтернативные пути
            ratings_file = data_path.parent / "movie_lens" / "ratings.csv"
        if not ratings_file.exists():
            ratings_file = data_path.parent.parent / "data" / "raw" / "movie_lens" / "ratings.csv"
        
        if not ratings_file.exists():
            raise FileNotFoundError(f"Файл ratings.csv не найден в {data_path}")
        
        # Загружаем данные
        df = pd.read_csv(ratings_file)
        
        # Нормализуем колонки: movieId -> itemId
        df = self._normalize_columns(df, {'itemId': 'movieId'})
        
        print(f"Загружено {len(df)} взаимодействий из MovieLens")
        
        return df


class GowallaLoader(BaseDatasetLoader):
    """Загрузчик для датасета Gowalla."""
    
    def load(self, data_path: Path) -> pd.DataFrame:
        """
        Загружает данные Gowalla.
        
        Поддерживает форматы:
        - Gowalla_cleanCheckins.csv (CSV с колонками)
        - Gowalla_totalCheckins.txt (текстовый файл: userId itemId timestamp)
        - Gowalla_edges.txt (текстовый файл: userId itemId, без timestamp)
        """
        # Пробуем разные файлы в порядке приоритета
        files_to_try = [
            ("Gowalla_cleanCheckins.csv", self._load_csv),
            ("Gowalla_totalCheckins.txt", self._load_txt_checkins),
            ("Gowalla_edges.txt", self._load_txt_edges),
        ]
        
        for filename, loader_func in files_to_try:
            file_path = data_path / filename
            if not file_path.exists():
                # Пробуем альтернативные пути
                file_path = data_path.parent / "gowalla" / filename
            if not file_path.exists():
                file_path = data_path.parent.parent / "data" / "raw" / "gowalla" / filename
            
            if file_path.exists():
                print(f"Найден файл: {filename}")
                try:
                    df = loader_func(file_path)
                    if not df.empty:
                        print(f"Загружено {len(df)} check-ins из Gowalla")
                        return df
                except Exception as e:
                    print(f"Ошибка при загрузке {filename}: {e}")
                    continue
        
        raise FileNotFoundError(
            f"Не найден файл с данными Gowalla в {data_path}\n"
            f"Искали: {[f[0] for f in files_to_try]}"
        )
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Загружает CSV файл Gowalla."""
        # Пробуем загрузить с разными разделителями
        try:
            df = pd.read_csv(file_path, nrows=1000)  # Читаем первые 1000 строк для определения формата
        except Exception:
            return pd.DataFrame()
        
        # Определяем колонки автоматически
        column_mapping = {}
        
        # Маппинг для userId
        for col in df.columns:
            col_lower = col.lower()
            if 'user' in col_lower and 'id' in col_lower:
                column_mapping['userId'] = col
                break
            elif col_lower == 'user':
                column_mapping['userId'] = col
                break
        
        # Маппинг для itemId
        for col in df.columns:
            col_lower = col.lower()
            if 'location' in col_lower and 'id' in col_lower:
                column_mapping['itemId'] = col
                break
            elif 'item' in col_lower and 'id' in col_lower:
                column_mapping['itemId'] = col
                break
            elif col_lower == 'location':
                column_mapping['itemId'] = col
                break
        
        # Маппинг для timestamp
        for col in df.columns:
            col_lower = col.lower()
            if 'time' in col_lower or 'date' in col_lower:
                column_mapping['timestamp'] = col
                break
        
        if 'userId' not in column_mapping or 'itemId' not in column_mapping:
            print(f"Не удалось определить колонки. Найденные колонки: {list(df.columns)}")
            return pd.DataFrame()
        
        # Загружаем весь файл
        df = pd.read_csv(file_path)
        df = self._normalize_columns(df, column_mapping)
        
        return df
    
    def _load_txt_checkins(self, file_path: Path) -> pd.DataFrame:
        """Загружает текстовый файл с check-ins (userId itemId timestamp)."""
        print(f"Чтение большого файла {file_path.name}...")
        print("Это может занять некоторое время...")
        
        data = []
        batch_size = 100000
        line_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    if line_count % 1000000 == 0:
                        print(f"  Обработано {line_count} строк...")
                    
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            data.append({
                                'userId': int(parts[0]),
                                'itemId': int(parts[1]),
                                'timestamp': int(parts[2]) if len(parts) > 2 else 0
                            })
                        except (ValueError, IndexError):
                            continue
                    
                    # Обрабатываем батчами
                    if len(data) >= batch_size:
                        if 'df' not in locals():
                            df = pd.DataFrame(data)
                        else:
                            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
                        data = []
            
            # Добавляем оставшиеся данные
            if data:
                if 'df' not in locals():
                    df = pd.DataFrame(data)
                else:
                    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
            
            if 'df' not in locals():
                df = pd.DataFrame(columns=['userId', 'itemId', 'timestamp'])
            
            print(f"  Всего обработано {line_count} строк, загружено {len(df)} записей")
            return df
            
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            return pd.DataFrame()
    
    def _load_txt_edges(self, file_path: Path) -> pd.DataFrame:
        """Загружает текстовый файл с edges (userId itemId, разделитель - табуляция или пробел)."""
        print(f"Чтение файла {file_path.name}...")
        print("Это может занять некоторое время...")
        
        # Читаем построчно (файл может быть большим)
        data = []
        batch_size = 100000
        line_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    if line_count % 100000 == 0:
                        print(f"  Обработано {line_count} строк...")
                    
                    # Пробуем табуляцию, затем пробел
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        parts = line.strip().split()
                    
                    if len(parts) >= 2:
                        try:
                            data.append({
                                'userId': int(parts[0]),
                                'itemId': int(parts[1]),
                            })
                        except (ValueError, IndexError):
                            continue
                    
                    # Обрабатываем батчами для экономии памяти
                    if len(data) >= batch_size:
                        if 'df' not in locals():
                            df = pd.DataFrame(data)
                        else:
                            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
                        data = []
            
            # Добавляем оставшиеся данные
            if data:
                if 'df' not in locals():
                    df = pd.DataFrame(data)
                else:
                    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
            
            if 'df' not in locals():
                df = pd.DataFrame(columns=['userId', 'itemId'])
            
            print(f"  Всего обработано {line_count} строк, загружено {len(df)} записей")
            return df
            
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()


class BookCrossingLoader(BaseDatasetLoader):
    """Загрузчик для датасета Book-Crossing."""
    
    def load(self, data_path: Path) -> pd.DataFrame:
        """
        Загружает данные Book-Crossing.
        
        Формат: BX-Book-Ratings.csv с колонками:
        - User-ID: ID пользователя
        - ISBN: ID книги
        - Book-Rating: рейтинг (0-10, где 0 означает implicit feedback)
        
        Примечание: Рейтинг 0 означает, что пользователь прочитал книгу, но не оценил её.
        Для implicit feedback это считается положительным взаимодействием.
        """
        # Ищем файл с рейтингами
        ratings_file = data_path / "BX-Book-Ratings.csv"
        if not ratings_file.exists():
            # Пробуем альтернативные пути
            ratings_file = data_path.parent / "book-crossing" / "BX-Book-Ratings.csv"
        if not ratings_file.exists():
            ratings_file = data_path.parent.parent / "data" / "raw" / "book-crossing" / "BX-Book-Ratings.csv"
        
        if not ratings_file.exists():
            raise FileNotFoundError(
                f"Файл BX-Book-Ratings.csv не найден в {data_path}\n"
                f"Проверьте наличие файла с рейтингами Book-Crossing."
            )
        
        print(f"Загрузка данных Book-Crossing из {ratings_file.name}...")
        
        # Book-Crossing использует точку с запятой как разделитель и кавычки
        # Формат: "User-ID";"ISBN";"Book-Rating"
        try:
            # Пробуем загрузить с разными разделителями
            df = pd.read_csv(ratings_file, sep=';', quotechar='"', encoding='latin-1')
        except Exception as e:
            print(f"Ошибка при чтении с разделителем ';': {e}")
            # Пробуем с запятой
            try:
                df = pd.read_csv(ratings_file, sep=',', quotechar='"', encoding='latin-1')
            except Exception as e2:
                raise ValueError(f"Не удалось загрузить файл: {e2}")
        
        print(f"Загружено {len(df)} строк")
        print(f"Колонки: {list(df.columns)}")
        
        # Нормализуем колонки
        # Book-Crossing использует: User-ID, ISBN, Book-Rating
        column_mapping = {
            'userId': 'User-ID',
            'itemId': 'ISBN',
            'rating': 'Book-Rating'
        }
        
        df = self._normalize_columns(df, column_mapping)
        
        # В Book-Crossing рейтинг 0 означает implicit feedback (пользователь прочитал, но не оценил)
        # Для рекомендательных систем это считается положительным взаимодействием
        # Преобразуем: rating 0 -> оставляем как есть (будет обработано в бинаризации)
        # rating > 0 -> положительное взаимодействие
        
        # Преобразуем userId и itemId в числовой формат (если они строковые)
        try:
            df['userId'] = pd.to_numeric(df['userId'], errors='coerce')
            df['itemId'] = df['itemId'].astype(str)  # ISBN может содержать буквы
        except Exception as e:
            print(f"Предупреждение при преобразовании типов: {e}")
        
        # Удаляем строки с NaN в userId или itemId (это обязательные поля)
        initial_count = len(df)
        df = df.dropna(subset=['userId', 'itemId'])
        if len(df) < initial_count:
            print(f"Удалено {initial_count - len(df)} строк с некорректными userId или itemId")
        
        # Проверяем наличие NaN в рейтингах (это нормально - будет обработано как implicit feedback)
        nan_ratings = df['rating'].isna().sum()
        if nan_ratings > 0:
            print(f"Найдено {nan_ratings} взаимодействий без рейтинга (будут обработаны как implicit feedback)")
        
        print(f"Загружено {len(df)} взаимодействий из Book-Crossing")
        print(f"Уникальных пользователей: {df['userId'].nunique()}")
        print(f"Уникальных книг: {df['itemId'].nunique()}")
        if not df['rating'].isna().all():
            print(f"Диапазон рейтингов: {df['rating'].min()} - {df['rating'].max()}")
            print(f"Рейтингов со значением 0 (implicit): {(df['rating'] == 0).sum()}")
        else:
            print("Все взаимодействия без рейтинга (implicit feedback)")
        
        return df


# Регистр загрузчиков
LOADER_REGISTRY = {
    'movie_lens': MovieLensLoader,
    'movielens1m': MovieLensLoader,
    'gowalla': GowallaLoader,
    'book_crossing': BookCrossingLoader,
    'book-crossing': BookCrossingLoader,
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

