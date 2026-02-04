"""
OrthogonalBundleGNN - Orthogonal Vector Bundle Graph Neural Network.

Implements the proposed method combining:
1. Bundle Structure: Each node has its own fiber space (local feature space)
2. Orthogonal Connection Matrices W_{ij}: Transport embeddings between fiber spaces
3. Parallel Transport: Move representations along edges preserving geometric structure
4. Local Transformations: Group & Shuffle mechanism for expressive power
5. Over-smoothing Prevention: Orthogonality ensures ||W_{ij} · x|| = ||x||

Architecture per layer:
    1. Parallel Transport: Transfer embeddings along edges via W_{ij}
    2. Local Transformation: Orthogonal transformation within fiber space
    3. Residual Connection: Stabilize training
    4. Layer Aggregation: Weighted combination of all layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict

from ..base import BaseRecommender
from .bundle_layer import BundleConnectionLayer
from .parallel_transport import parallel_transport_along_edges
from .group_shuffle_layer import GroupShuffleLayer


class OrthogonalBundleGNN(BaseRecommender):
    """
    Orthogonal Vector Bundle GNN for Recommendations
    
    Proposed Method:
    - Bundle Structure: Each node (user/item) has its own fiber space
    - Connection Matrices W_{ij}: Orthogonal transformations for parallel transport
    - Group & Shuffle: Orthogonal parametrization ensuring ||W_{ij} · x|| = ||x||
    - Over-smoothing Prevention: Orthogonality prevents representation collapse
    
    Architecture (L layers):
        For each layer l:
            1. Parallel Transport: x^(l) transported along edges via W_{ij}
            2. Local Transformation: Orthogonal Group & Shuffle within fiber space
            3. Residual Connection: x^(l+1) = (1-α)·transformed + α·x^(0)
        
        Final: Weighted aggregation of all layer embeddings
    
    Key Properties:
    - Norm preservation: ||W_{ij} · x|| = ||x||
    - Prevents over-smoothing even with deep layers
    - Maintains expressiveness through local transformations
    """
    
    def __init__(
        self, 
        n_users: int, 
        n_items: int, 
        embedding_dim: int = 64,
        n_layers: int = 3,
        block_size: int = 8,
        residual_alpha: float = 0.1,
        dropout: float = 0.0,
        init_scale: float = 0.01,
        use_parallel_transport: bool = True,
        use_edge_index: bool = False
    ):
        """
        Initialize OrthogonalBundleGNN.
        
        Args:
            n_users: number of users
            n_items: number of items
            embedding_dim: embedding dimension (must be divisible by block_size)
            n_layers: number of layers (L in paper)
            block_size: block size for orthogonal matrices
            residual_alpha: α in residual connection (0.0 = no residual, 1.0 = only initial)
            dropout: dropout probability
            init_scale: initialization scale for parameters
            use_parallel_transport: whether to use parallel transport with W_{ij}
            use_edge_index: whether to use edge_index format (True) or adj_matrix (False)
        """
        super().__init__(n_users, n_items, embedding_dim)
        
        if embedding_dim % block_size != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by block_size ({block_size})"
            )
        
        self.n_layers = n_layers
        self.block_size = block_size
        self.residual_alpha = residual_alpha
        self.dropout = dropout
        self.use_parallel_transport = use_parallel_transport
        self.use_edge_index = use_edge_index
        
        # 1. BASE EMBEDDINGS - Fiber spaces for each node
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # 2. CONNECTION MATRICES W_{ij} - Orthogonal transformations for parallel transport
        if use_parallel_transport:
            self.connection_layers = nn.ModuleList([
                BundleConnectionLayer(embedding_dim, block_size)
                for _ in range(n_layers)
            ])
        
        # 3. LOCAL TRANSFORMATION LAYERS - Group & Shuffle within fiber spaces
        self.local_transform_layers = nn.ModuleList([
            GroupShuffleLayer(embedding_dim, block_size, init_scale)
            for _ in range(n_layers)
        ])
        
        # 4. DROPOUT
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # 5. LAYER AGGREGATION WEIGHTS (learnable)
        self.layer_weights = nn.Parameter(torch.ones(n_layers + 1))
    
    def forward(
        self,
        adj_matrix: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Orthogonal Bundle GNN
        
        Implements the proposed architecture:
        For each layer l:
            1. Parallel Transport: x transported along edges via W_{ij}
            2. Local Transformation: Group & Shuffle within fiber space
            3. Residual Connection: combine with initial embeddings
        
        Args:
            adj_matrix: normalized adjacency matrix [N, N] (if not using edge_index)
            edge_index: edge list [2, num_edges] (if not using adj_matrix)
        
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        # Check input format
        if self.use_edge_index:
            if edge_index is None:
                raise ValueError("edge_index must be provided when use_edge_index=True")
        else:
            if adj_matrix is None:
                raise ValueError("adj_matrix must be provided when use_edge_index=False")
        
        # Initial embeddings (fiber spaces at each node)
        x_init = torch.cat([
            self.user_embedding.weight,  # [n_users, embedding_dim]
            self.item_embedding.weight   # [n_items, embedding_dim]
        ], dim=0)  # [N, embedding_dim], N = n_users + n_items
        
        x = x_init
        all_layer_embeddings = [x]
        
        # Pass through L layers
        for layer_idx in range(self.n_layers):
            if self.use_parallel_transport:
                # Get connection matrix W_{ij} for this layer
                W_connection = self.connection_layers[layer_idx]()
                
                if self.use_edge_index:
                    # STEP 1: PARALLEL TRANSPORT via edge_index
                    # Transport embeddings along edges: x_j = Σ_{i: (i,j)∈E} W_{ij} · x_i
                    x_transported = parallel_transport_along_edges(x, edge_index, W_connection)
                else:
                    # STEP 1: PARALLEL TRANSPORT via adjacency matrix
                    # First: graph convolution (message passing)
                    if adj_matrix.is_sparse:
                        x_conv = torch.sparse.mm(adj_matrix, x)
                    else:
                        x_conv = torch.mm(adj_matrix, x)
                    # Then: apply connection matrix W_{ij}
                    x_transported = x_conv @ W_connection
            else:
                # No parallel transport, just graph convolution
                if self.use_edge_index:
                    # Simulate graph convolution with edge_index
                    x_transported = self._graph_conv_edge_index(x, edge_index)
                else:
                    if adj_matrix.is_sparse:
                        x_transported = torch.sparse.mm(adj_matrix, x)
                    else:
                        x_transported = torch.mm(adj_matrix, x)
            
            # STEP 2: LOCAL TRANSFORMATION (Group & Shuffle within fiber space)
            x_transformed = self.local_transform_layers[layer_idx](x_transported)
            
            # STEP 3: RESIDUAL CONNECTION
            # x^(l+1) = (1-α)·transformed + α·x^(0)
            # This prevents over-smoothing by maintaining initial information
            x = (1 - self.residual_alpha) * x_transformed + \
                self.residual_alpha * x_init
            
            # STEP 4: DROPOUT
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)
            
            all_layer_embeddings.append(x)
        
        # LAYER AGGREGATION: Weighted combination of all layers
        layer_weights_normalized = F.softmax(self.layer_weights, dim=0)
        x_final = sum([
            w * emb for w, emb in zip(layer_weights_normalized, all_layer_embeddings)
        ])
        
        # Split back into users and items
        user_embeddings = x_final[:self.n_users]
        item_embeddings = x_final[self.n_users:]
        
        return user_embeddings, item_embeddings
    
    def _graph_conv_edge_index(self, x, edge_index):
        """Simple graph convolution using edge_index (fallback when no parallel transport)"""
        src, dst = edge_index
        x_aggregated = torch.zeros_like(x)
        x_aggregated.index_add_(0, dst, x[src])
        return x_aggregated
    
    def predict(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict scores for user-item pairs.
        """
        user_emb, item_emb = self.get_all_embeddings(adj_matrix, edge_index)
        user_emb_selected = user_emb[users]
        item_emb_selected = item_emb[items]
        scores = (user_emb_selected * item_emb_selected).sum(dim=1)
        return scores
    
    def get_all_embeddings(
        self,
        adj_matrix: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all embeddings for users and items."""
        return self.forward(adj_matrix, edge_index)
    
    def get_orthogonality_errors(self) -> torch.Tensor:
        """
        Get orthogonality errors for all layers.
        
        Useful for monitoring during training.
        
        Returns:
            Tensor with orthogonality errors [n_layers]
        """
        errors = []
        for layer in self.local_transform_layers:
            error = layer.get_orthogonality_error()
            errors.append(error)
        return torch.stack(errors)

    def get_orthogonality_metrics(self) -> Dict[str, torch.Tensor]:
        """
        Get runtime orthogonality metrics for monitoring.

        Returns:
            Dict with aggregated orthogonality stats.
        """
        metrics: Dict[str, torch.Tensor] = {}

        local_fro = []
        local_max = []
        for layer in self.local_transform_layers:
            if hasattr(layer, 'get_orthogonality_metrics'):
                fro_error, max_deviation = layer.get_orthogonality_metrics()
                local_max.append(max_deviation)
            else:
                fro_error = layer.get_orthogonality_error()
            local_fro.append(fro_error)

        if local_fro:
            local_fro_tensor = torch.stack(local_fro)
            metrics['local_fro_mean'] = local_fro_tensor.mean()
            metrics['local_fro_max'] = local_fro_tensor.max()
        if local_max:
            metrics['local_max_dev'] = torch.stack(local_max).max()

        if self.use_parallel_transport:
            conn_fro = []
            conn_max = []
            for layer in self.connection_layers:
                if hasattr(layer, 'get_orthogonality_metrics'):
                    fro_error, max_deviation = layer.get_orthogonality_metrics()
                    conn_fro.append(fro_error)
                    conn_max.append(max_deviation)
            if conn_fro:
                conn_fro_tensor = torch.stack(conn_fro)
                metrics['conn_fro_mean'] = conn_fro_tensor.mean()
                metrics['conn_fro_max'] = conn_fro_tensor.max()
            if conn_max:
                metrics['conn_max_dev'] = torch.stack(conn_max).max()

        return metrics
    
    def get_layer_embeddings(
        self,
        adj_matrix: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Get embeddings for each layer (for over-smoothing analysis).
        
        Returns embeddings at each layer to analyze:
        - Over-smoothing: similarity between embeddings across layers
        - Norm preservation: ||x^(l)|| should remain stable
        """
        # Initial embeddings
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        all_embeddings = [x.clone()]
        
        # Pass through each layer
        for layer_idx in range(self.n_layers):
            if self.use_parallel_transport:
                W_connection = self.connection_layers[layer_idx]()
                
                if self.use_edge_index and edge_index is not None:
                    x_transported = parallel_transport_along_edges(x, edge_index, W_connection)
                elif adj_matrix is not None:
                    if adj_matrix.is_sparse:
                        x_conv = torch.sparse.mm(adj_matrix, x)
                    else:
                        x_conv = torch.mm(adj_matrix, x)
                    x_transported = x_conv @ W_connection
                else:
                    raise ValueError("Must provide either adj_matrix or edge_index")
            else:
                if self.use_edge_index and edge_index is not None:
                    x_transported = self._graph_conv_edge_index(x, edge_index)
                elif adj_matrix is not None:
                    if adj_matrix.is_sparse:
                        x_transported = torch.sparse.mm(adj_matrix, x)
                    else:
                        x_transported = torch.mm(adj_matrix, x)
                else:
                    raise ValueError("Must provide either adj_matrix or edge_index")
            
            # Local transformation
            x = self.local_transform_layers[layer_idx](x_transported)
            all_embeddings.append(x.clone())
        
        return all_embeddings
    
    def reset_parameters(self):
        """
        Reset parameters to initial values.
        """
        # Reset embeddings
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        
        # Reset layers
        for layer in self.local_transform_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

