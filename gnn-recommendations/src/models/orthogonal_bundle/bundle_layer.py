# src/models/orthogonal_bundle/bundle_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BundleConnectionLayer(nn.Module):
    """
    Bundle Connection Layer: computes orthogonal connection matrices W_{ij}
    for parallel transport along graph edges.
    
    Key idea:
    - Each edge (i,j) has its own connection matrix W_{ij}
    - W_{ij} is orthogonal → preserves norm during transport
    - Built through Group & Shuffle mechanism
    """
    
    def __init__(self, embedding_dim, block_size, n_blocks=None):
        """
        Args:
            embedding_dim: dimension of embeddings
            block_size: size of blocks for Group mechanism
            n_blocks: number of blocks (if None, computed automatically)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.n_blocks = n_blocks or (embedding_dim // block_size)
        
        # Parameters for skew-symmetric matrices (for each block)
        self.skew_params = nn.ParameterList([
            nn.Parameter(torch.randn(block_size, block_size) * 0.01)
            for _ in range(self.n_blocks)
        ])
        
        # Shuffle permutation (fixed for this layer)
        self.register_buffer('shuffle_perm', self._create_shuffle_permutation())
    
    def _create_shuffle_permutation(self):
        """Create shuffle permutation"""
        return torch.randperm(self.embedding_dim)
    
    def forward(self, edge_index=None):
        """
        Build orthogonal connection matrix
        
        Args:
            edge_index: [2, num_edges] - optional, for edge-specific matrices
        
        Returns:
            W_connection: [embedding_dim, embedding_dim] orthogonal matrix
        """
        # Build block-diagonal orthogonal matrix
        blocks = []
        
        for skew_param in self.skew_params:
            # Make skew-symmetric: A = A - A^T
            A_skew = skew_param - skew_param.transpose(-2, -1)
            
            # Exponential map: W = exp(A_skew)
            # This guarantees orthogonality!
            W_block = torch.matrix_exp(A_skew)
            
            blocks.append(W_block)
        
        # Assemble block-diagonal matrix
        W_orth = torch.block_diag(*blocks)
        
        # Apply shuffle permutation
        W_shuffled = W_orth[:, self.shuffle_perm]
        
        return W_shuffled
    
    def get_connection_matrix_for_edge(self, src_node, dst_node):
        """
        Get connection matrix for a specific edge
        (currently using one matrix for all edges, can be extended)
        
        Args:
            src_node: source node index
            dst_node: destination node index
        
        Returns:
            W: [embedding_dim, embedding_dim]
        """
        return self.forward()

    def get_orthogonality_metrics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute orthogonality metrics for the connection matrix.

        Returns:
            Tuple (frobenius_error, max_deviation)
        """
        W_orth = self.forward()
        identity = torch.eye(self.embedding_dim, device=W_orth.device, dtype=W_orth.dtype)
        diff = W_orth.T @ W_orth - identity
        fro_error = torch.norm(diff, p='fro')
        max_deviation = diff.abs().max()
        return fro_error, max_deviation


class EdgeSpecificBundleConnection(nn.Module):
    """
    Extended version: different connection matrices for different edge types
    (e.g., user-item vs item-user)
    """
    
    def __init__(self, embedding_dim, block_size, n_edge_types=2):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Create separate connection layers for each edge type
        self.connection_layers = nn.ModuleList([
            BundleConnectionLayer(embedding_dim, block_size)
            for _ in range(n_edge_types)
        ])
    
    def forward(self, edge_index, edge_type):
        """
        Args:
            edge_index: [2, num_edges]
            edge_type: [num_edges] - type of each edge (0 or 1)
        
        Returns:
            W_connections: [num_edges, embedding_dim, embedding_dim]
        """
        num_edges = edge_index.size(1)
        device = edge_index.device
        
        # Get connection matrices for each type
        W_type_0 = self.connection_layers[0]()
        W_type_1 = self.connection_layers[1]()
        
        # Assemble for each edge
        W_connections = torch.zeros(num_edges, self.embedding_dim, self.embedding_dim, 
                                     device=device)
        
        mask_0 = (edge_type == 0)
        mask_1 = (edge_type == 1)
        
        W_connections[mask_0] = W_type_0.unsqueeze(0)
        W_connections[mask_1] = W_type_1.unsqueeze(0)
        
        return W_connections

