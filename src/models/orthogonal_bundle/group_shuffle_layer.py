"""
GroupShuffleLayer - локальные преобразования внутри пространства волокон.

Это механизм Group и Shuffle для ортогональных преобразований.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class GroupShuffleLayer(nn.Module):
    """
    Слой Group & Shuffle для локальных преобразований внутри пространства волокон.

    Компоненты:
    1. Group (ортогональное преобразование) - блочно-диагональная ортогональная матрица
    2. Shuffle - перестановка признаков

    Этот слой применяет ортогональные преобразования для сохранения нормы, при этом
    позволяя выразительное смешивание признаков.
    """

    def __init__(
        self,
        dim: int,
        block_size: int,
        init_scale: float = 0.01
    ):
        """
        Инициализирует GroupShuffleLayer.

        Args:
            dim: размерность признаков (embedding_dim)
            block_size: размер блоков для ортогональной матрицы
            init_scale: масштаб инициализации для параметров
        """
        super().__init__()
        
        self.dim = dim
        self.block_size = block_size

        # Проверяем, что dim делится на block_size
        if dim % block_size != 0:
            raise ValueError(
                f"dim ({dim}) должна делиться на block_size ({block_size})"
            )

        self.n_blocks = dim // block_size

        # Параметры для кососимметричных матриц
        # Каждый блок строится из кососимметричной матрицы через экспоненциальное отображение
        self.skew_params = nn.ParameterList([
            nn.Parameter(torch.randn(block_size, block_size) * init_scale)
            for _ in range(self.n_blocks)
        ])

        # Перестановка для Shuffle - фиксированная перестановка
        # Регистрируем как буфер (не обучаемый параметр)
        self.register_buffer('perm', self._create_shuffle_permutation())

    def _create_shuffle_permutation(self) -> torch.Tensor:
        """
        Создаёт перестановку для shuffle.

        Returns:
            Тензор с индексами перестановки [dim]
        """
        # Создаём случайную перестановку
        perm = torch.randperm(self.dim)
        return perm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через слой.

        Процесс:
        1. Group: ортогональное преобразование (блочно-диагональная матрица)
        2. Shuffle: перестановка признаков

        Args:
            x: признаки вершин [N, dim] или [batch, dim]

        Returns:
            Преобразованные признаки [N, dim]
        """
        # 1. Строим ортогональную матрицу (Group)
        W_orth = self._build_orthogonal_matrix()  # [dim, dim]

        # 2. Применяем ортогональное преобразование
        x_transformed = x @ W_orth  # [N, dim]

        # 3. Shuffle - перестановка признаков
        x_shuffled = x_transformed[:, self.perm]  # [N, dim]

        return x_shuffled
    
    def _build_orthogonal_matrix(self) -> torch.Tensor:
        """
        Строит блочно-диагональную ортогональную матрицу.

        Метод:
        1. Для каждого блока создаём кососимметричную матрицу A_skew
        2. Применяем экспоненциальное отображение: exp(A_skew) → ортогональная матрица
        3. Собираем блочно-диагональную матрицу

        Returns:
            Ортогональная матрица [dim, dim]
        """
        blocks = []

        for param in self.skew_params:
            # Делаем кососимметричной: A_skew = A - A^T
            A_skew = param - param.T  # [block_size, block_size]

            # Экспоненциальное отображение: exp(A_skew) → ортогональная матрица
            # Это гарантирует ортогональность (группа Ли SO(n))
            try:
                block_orth = torch.matrix_exp(A_skew)  # [block_size, block_size]
            except RuntimeError:
                # Если matrix_exp не работает (старые версии PyTorch), используем альтернативу
                block_orth = self._matrix_exp_alternative(A_skew)

            blocks.append(block_orth)

        # Собираем блочно-диагональную матрицу
        W_orth = torch.block_diag(*blocks)  # [dim, dim]

        return W_orth
    
    def _matrix_exp_alternative(self, A: torch.Tensor, n_terms: int = 10) -> torch.Tensor:
        """
        Альтернативная реализация матричной экспоненты через ряд Тейлора.

        Используется, если torch.matrix_exp недоступна.

        Args:
            A: матрица [block_size, block_size]
            n_terms: количество членов ряда Тейлора

        Returns:
            exp(A) [block_size, block_size]
        """
        # Ряд Тейлора: exp(A) = I + A + A^2/2! + A^3/3! + ...
        result = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        A_power = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        factorial = 1.0

        for i in range(1, n_terms + 1):
            A_power = A_power @ A
            factorial *= i
            result = result + A_power / factorial

        return result

    def get_orthogonality_error(self) -> torch.Tensor:
        """
        Вычисляет ошибку ортогональности матрицы.

        Полезно для мониторинга во время обучения.

        Returns:
            Ошибка ортогональности (должна быть близка к 0)
        """
        W_orth = self._build_orthogonal_matrix()
        # W^T @ W должна быть близка к единичной матрице
        identity = torch.eye(self.dim, device=W_orth.device, dtype=W_orth.dtype)
        WtW = W_orth.T @ W_orth
        error = torch.norm(WtW - identity, p='fro')
        return error

    def get_orthogonality_metrics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисляет метрики ортогональности для мониторинга.

        Returns:
            Tuple (frobenius_error, max_deviation)
        """
        W_orth = self._build_orthogonal_matrix()
        identity = torch.eye(self.dim, device=W_orth.device, dtype=W_orth.dtype)
        diff = W_orth.T @ W_orth - identity
        fro_error = torch.norm(diff, p='fro')
        max_deviation = diff.abs().max()
        return fro_error, max_deviation
    
    def reset_parameters(self):
        """Сбрасывает параметры к начальным значениям."""
        for param in self.skew_params:
            nn.init.normal_(param, mean=0.0, std=0.01)

