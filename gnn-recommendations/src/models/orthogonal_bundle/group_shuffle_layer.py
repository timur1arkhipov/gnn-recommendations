"""
GroupShuffleLayer - local transformations within fiber space.

This is the Group and Shuffle mechanism for orthogonal transformations.
"""

import torch
import torch.nn as nn
from typing import Optional


class GroupShuffleLayer(nn.Module):
    """
    Group & Shuffle Layer for local transformations within fiber space.
    
    Components:
    1. Group (orthogonal transformation) - block-diagonal orthogonal matrix
    2. Shuffle - feature permutation
    
    This layer applies orthogonal transformations to preserve norm while
    allowing expressive feature mixing.
    """
    
    def __init__(
        self,
        dim: int,
        block_size: int,
        init_scale: float = 0.01
    ):
        """
        Initialize GroupShuffleLayer.
        
        Args:
            dim: feature dimension (embedding_dim)
            block_size: size of blocks for orthogonal matrix
            init_scale: initialization scale for parameters
        """
        super().__init__()
        
        self.dim = dim
        self.block_size = block_size
        
        # Check that dim is divisible by block_size
        if dim % block_size != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by block_size ({block_size})"
            )
        
        self.n_blocks = dim // block_size
        
        # Parameters for skew-symmetric matrices
        # Each block is built from skew-symmetric matrix via exponential map
        self.skew_params = nn.ParameterList([
            nn.Parameter(torch.randn(block_size, block_size) * init_scale)
            for _ in range(self.n_blocks)
        ])
        
        # Shuffle permutation - fixed permutation
        # Register as buffer (not a trainable parameter)
        self.register_buffer('perm', self._create_shuffle_permutation())
    
    def _create_shuffle_permutation(self) -> torch.Tensor:
        """
        Create permutation for shuffle.
        
        Returns:
            Tensor with permutation indices [dim]
        """
        # Create random permutation
        perm = torch.randperm(self.dim)
        return perm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer.
        
        Process:
        1. Group: orthogonal transformation (block-diagonal matrix)
        2. Shuffle: feature permutation
        
        Args:
            x: node features [N, dim] or [batch, dim]
        
        Returns:
            Transformed features [N, dim]
        """
        # 1. Build orthogonal matrix (Group)
        W_orth = self._build_orthogonal_matrix()  # [dim, dim]
        
        # 2. Apply orthogonal transformation
        x_transformed = x @ W_orth  # [N, dim]
        
        # 3. Shuffle - feature permutation
        x_shuffled = x_transformed[:, self.perm]  # [N, dim]
        
        return x_shuffled
    
    def _build_orthogonal_matrix(self) -> torch.Tensor:
        """
        Build block-diagonal orthogonal matrix.
        
        Method:
        1. For each block, create skew-symmetric matrix A_skew
        2. Apply exponential map: exp(A_skew) → orthogonal matrix
        3. Assemble block-diagonal matrix
        
        Returns:
            Orthogonal matrix [dim, dim]
        """
        blocks = []
        
        for param in self.skew_params:
            # Make skew-symmetric: A_skew = A - A^T
            A_skew = param - param.T  # [block_size, block_size]
            
            # Exponential map: exp(A_skew) → orthogonal matrix
            # This guarantees orthogonality (Lie group SO(n))
            try:
                block_orth = torch.matrix_exp(A_skew)  # [block_size, block_size]
            except RuntimeError:
                # If matrix_exp doesn't work (older PyTorch versions), use alternative
                block_orth = self._matrix_exp_alternative(A_skew)
            
            blocks.append(block_orth)
        
        # Assemble block-diagonal matrix
        W_orth = torch.block_diag(*blocks)  # [dim, dim]
        
        return W_orth
    
    def _matrix_exp_alternative(self, A: torch.Tensor, n_terms: int = 10) -> torch.Tensor:
        """
        Alternative implementation of matrix exponential via Taylor series.
        
        Used if torch.matrix_exp is unavailable.
        
        Args:
            A: matrix [block_size, block_size]
            n_terms: number of Taylor series terms
        
        Returns:
            exp(A) [block_size, block_size]
        """
        # Taylor series: exp(A) = I + A + A^2/2! + A^3/3! + ...
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
        Compute orthogonality error of the matrix.
        
        Useful for monitoring during training.
        
        Returns:
            Orthogonality error (should be close to 0)
        """
        W_orth = self._build_orthogonal_matrix()
        # W^T @ W should be close to Identity
        identity = torch.eye(self.dim, device=W_orth.device, dtype=W_orth.dtype)
        WtW = W_orth.T @ W_orth
        error = torch.norm(WtW - identity, p='fro')
        return error
    
    def reset_parameters(self):
        """Reset parameters to initial values."""
        for param in self.skew_params:
            nn.init.normal_(param, mean=0.0, std=0.01)

