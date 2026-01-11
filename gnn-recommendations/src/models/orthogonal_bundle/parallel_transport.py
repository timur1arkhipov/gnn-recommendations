import torch
import torch.nn as nn


def parallel_transport_along_edges(x, edge_index, W_connection):
    """
    Параллельный перенос node features вдоль рёбер графа
    
    Математика:
    Для каждого ребра (i -> j):
        x_transported_j = W_{ij} @ x_i
    
    Затем агрегируем по destination нодам:
        x_aggregated_j = Σ_{i: (i,j) ∈ E} x_transported_j
    
    Args:
        x: [n_nodes, embedding_dim] - node features
        edge_index: [2, num_edges] - список рёбер
        W_connection: [embedding_dim, embedding_dim] - connection matrix
                      (одна для всех рёбер, или [num_edges, d, d] для edge-specific)
    
    Returns:
        x_aggregated: [n_nodes, embedding_dim] - агрегированные features
    """
    src, dst = edge_index  # [num_edges]
    num_nodes = x.size(0)
    
    # Получить features source нод
    x_src = x[src]  # [num_edges, embedding_dim]
    
    # Применить parallel transport
    if W_connection.dim() == 2:
        # Одна матрица для всех рёбер
        # x_transported = x_src @ W_connection^T
        x_transported = torch.mm(x_src, W_connection.t())  # [num_edges, d]
    
    elif W_connection.dim() == 3:
        # Разные матрицы для каждого ребра
        # x_transported[e] = W_connection[e] @ x_src[e]
        x_transported = torch.bmm(
            W_connection,  # [num_edges, d, d]
            x_src.unsqueeze(-1)  # [num_edges, d, 1]
        ).squeeze(-1)  # [num_edges, d]
    
    else:
        raise ValueError(f"Invalid W_connection shape: {W_connection.shape}")
    
    # Агрегировать по destination нодам (БЕЗ torch_scatter)
    x_aggregated = torch.zeros_like(x)
    x_aggregated.index_add_(0, dst, x_transported)
    
    return x_aggregated


class ParallelTransportLayer(nn.Module):
    """
    Слой для параллельного переноса с нормализацией
    """
    
    def __init__(self, embedding_dim, normalize=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.normalize = normalize
    
    def forward(self, x, edge_index, W_connection, edge_weight=None):
        """
        Args:
            x: [n_nodes, embedding_dim]
            edge_index: [2, num_edges]
            W_connection: connection matrix
            edge_weight: [num_edges] - опциональные веса рёбер
        
        Returns:
            x_aggregated: [n_nodes, embedding_dim]
        """
        # Parallel transport
        x_transported = parallel_transport_along_edges(x, edge_index, W_connection)
        
        # Опционально: взвешивание рёбер
        if edge_weight is not None:
            src, dst = edge_index
            # Умножить на веса
            x_transported = x_transported * edge_weight.unsqueeze(-1)
        
        # Нормализация (как в GCN)
        if self.normalize:
            # Вычислить степени нод (БЕЗ torch_scatter)
            src, dst = edge_index
            deg = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
            ones = torch.ones(edge_index.size(1), device=x.device, dtype=torch.long)
            deg.index_add_(0, dst, ones)
            
            deg_inv_sqrt = deg.float().pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            
            x_transported = x_transported * deg_inv_sqrt.unsqueeze(-1)
        
        return x_transported

