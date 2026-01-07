"""
GroupShuffleLayer - ключевой компонент метода Group and Shuffle.

Реализует ортогональное преобразование с блочной структурой для борьбы
с over-smoothing в GNN рекомендательных системах.
"""

import torch
import torch.nn as nn
from typing import Optional


class GroupShuffleLayer(nn.Module):
    """
    Слой Group and Shuffle для GNN рекомендательных систем.
    
    Компоненты:
    1. Graph convolution - агрегация соседей
    2. Group (ортогональное преобразование) - блочно-диагональная ортогональная матрица
    3. Shuffle - перестановка признаков
    
    Метод из статьи Gorbunov and Yudin "Group and Shuffle".
    """
    
    def __init__(
        self,
        dim: int,
        block_size: int,
        init_scale: float = 0.01
    ):
        """
        Инициализация GroupShuffleLayer.
        
        Args:
            dim: размерность признаков (embedding_dim)
            block_size: размер блока для ортогональной матрицы
            init_scale: масштаб инициализации параметров
        """
        super().__init__()
        
        self.dim = dim
        self.block_size = block_size
        
        # Проверяем, что dim делится на block_size
        if dim % block_size != 0:
            raise ValueError(
                f"dim ({dim}) должно делиться на block_size ({block_size})"
            )
        
        self.n_blocks = dim // block_size
        
        # Параметры для skew-symmetric матриц
        # Каждый блок будет построен из skew-symmetric матрицы через exponential map
        self.skew_params = nn.ParameterList([
            nn.Parameter(torch.randn(block_size, block_size) * init_scale)
            for _ in range(self.n_blocks)
        ])
        
        # Shuffle permutation - фиксированная перестановка
        # Регистрируем как buffer (не обучаемый параметр)
        self.register_buffer('perm', self._create_shuffle_permutation())
        self.register_buffer('inv_perm', self._create_inverse_permutation())
    
    def _create_shuffle_permutation(self) -> torch.Tensor:
        """
        Создать перестановку для shuffle.
        
        Returns:
            Тензор с перестановкой индексов [dim]
        """
        # Создаем случайную перестановку
        perm = torch.randperm(self.dim)
        return perm
    
    def _create_inverse_permutation(self) -> torch.Tensor:
        """
        Создать обратную перестановку для shuffle.
        
        Нужна для правильной работы с градиентами.
        
        Returns:
            Тензор с обратной перестановкой индексов [dim]
        """
        # Создаем обратную перестановку
        inv_perm = torch.zeros_like(self.perm)
        inv_perm[self.perm] = torch.arange(self.dim, device=self.perm.device)
        return inv_perm
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass через слой.
        
        Процесс:
        1. Graph convolution: агрегация соседей
        2. Group: ортогональное преобразование (блочно-диагональная матрица)
        3. Shuffle: перестановка признаков
        
        Args:
            x: node features [N, dim], где N = n_users + n_items
            adj: normalized adjacency matrix [N, N] (sparse или dense)
        
        Returns:
            Преобразованные features [N, dim]
        """
        # 1. Graph convolution - агрегация соседей
        # Если adj sparse, используем sparse matrix multiplication
        if adj.is_sparse:
            x_conv = torch.sparse.mm(adj, x)  # [N, dim]
        else:
            x_conv = torch.mm(adj, x)  # [N, dim]
        
        # 2. Построить ортогональную матрицу (Group)
        W_orth = self._build_orthogonal_matrix()  # [dim, dim]
        
        # 3. Применить ортогональное преобразование
        x_transformed = x_conv @ W_orth  # [N, dim]
        
        # 4. Shuffle - перестановка признаков
        x_shuffled = x_transformed[:, self.perm]  # [N, dim]
        
        return x_shuffled
    
    def _build_orthogonal_matrix(self) -> torch.Tensor:
        """
        Построить блочно-диагональную ортогональную матрицу.
        
        Метод:
        1. Для каждого блока создаем skew-symmetric матрицу A_skew
        2. Применяем exponential map: exp(A_skew) → ортогональная матрица
        3. Собираем блочно-диагональную матрицу
        
        Returns:
            Ортогональная матрица [dim, dim]
        """
        blocks = []
        
        for param in self.skew_params:
            # Делаем skew-symmetric: A_skew = A - A^T
            A_skew = param - param.T  # [block_size, block_size]
            
            # Exponential map: exp(A_skew) → ортогональная матрица
            # Это гарантирует ортогональность (Lie group SO(n))
            try:
                block_orth = torch.matrix_exp(A_skew)  # [block_size, block_size]
            except RuntimeError as e:
                # Если matrix_exp не работает (старые версии PyTorch), используем альтернативу
                # Для малых матриц можно использовать ряд Тейлора
                block_orth = self._matrix_exp_alternative(A_skew)
            
            blocks.append(block_orth)
        
        # Собираем блочно-диагональную матрицу
        W_orth = torch.block_diag(*blocks)  # [dim, dim]
        
        return W_orth
    
    def _matrix_exp_alternative(self, A: torch.Tensor, n_terms: int = 10) -> torch.Tensor:
        """
        Альтернативная реализация matrix exponential через ряд Тейлора.
        
        Используется, если torch.matrix_exp недоступен.
        
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
        Вычислить ошибку ортогональности матрицы.
        
        Полезно для мониторинга во время обучения.
        
        Returns:
            Ошибка ортогональности (должна быть близка к 0)
        """
        W_orth = self._build_orthogonal_matrix()
        # W^T @ W должно быть близко к Identity
        identity = torch.eye(self.dim, device=W_orth.device, dtype=W_orth.dtype)
        WtW = W_orth.T @ W_orth
        error = torch.norm(WtW - identity, p='fro')
        return error

