"""
Функции потерь для обучения моделей рекомендательных систем.

Основная функция потерь: BPR Loss (Bayesian Personalized Ranking)
"""

import torch
import torch.nn as nn
from typing import Optional


class BPRLoss(nn.Module):
    """
    BPR Loss (Bayesian Personalized Ranking Loss).
    
    Используется для обучения моделей на implicit feedback.
    
    Формула: L = -log(σ(pos_score - neg_score))
    где σ(x) = 1 / (1 + exp(-x)) - sigmoid функция
    
    Ссылка: Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback" (2009)
    """
    
    def __init__(self):
        """Инициализация BPR Loss."""
        super().__init__()
    
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисляет BPR Loss.
        
        Args:
            pos_scores: scores для положительных пар (user, positive_item) [batch_size]
            neg_scores: scores для отрицательных пар (user, negative_item) [batch_size]
        
        Returns:
            Средний BPR Loss (скаляр)
        """
        # Разница между положительными и отрицательными scores
        diff = pos_scores - neg_scores  # [batch_size]
        
        # BPR Loss: -log(σ(diff))
        # σ(diff) = sigmoid(diff) = 1 / (1 + exp(-diff))
        # -log(σ(diff)) = -log(1 / (1 + exp(-diff))) = log(1 + exp(-diff))
        # Для численной стабильности используем log_sigmoid
        loss = -torch.nn.functional.logsigmoid(diff)  # [batch_size]
        
        # Средний loss по батчу
        return loss.mean()


class BCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss (альтернатива BPR Loss).
    
    Используется реже, но может быть полезен для некоторых экспериментов.
    """
    
    def __init__(self):
        """Инициализация BCE Loss."""
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисляет BCE Loss.
        
        Args:
            pos_scores: scores для положительных пар [batch_size]
            neg_scores: scores для отрицательных пар [batch_size]
        
        Returns:
            Средний BCE Loss
        """
        # Объединяем positive и negative scores
        all_scores = torch.cat([pos_scores, neg_scores], dim=0)  # [2*batch_size]
        
        # Labels: 1 для positive, 0 для negative
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores)
        ], dim=0)  # [2*batch_size]
        
        return self.bce_loss(all_scores, labels)


class RegularizedLoss(nn.Module):
    """
    Обертка для добавления регуляризации к любой функции потерь.
    
    Добавляет L2 регуляризацию параметров модели.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        weight_decay: float = 1e-4
    ):
        """
        Инициализация RegularizedLoss.
        
        Args:
            base_loss: базовая функция потерь (например, BPRLoss)
            weight_decay: коэффициент L2 регуляризации
        """
        super().__init__()
        self.base_loss = base_loss
        self.weight_decay = weight_decay
    
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """
        Вычисляет loss с регуляризацией.
        
        Args:
            pos_scores: scores для положительных пар
            neg_scores: scores для отрицательных пар
            model: модель для регуляризации
        
        Returns:
            Loss с регуляризацией
        """
        # Базовый loss
        loss = self.base_loss(pos_scores, neg_scores)
        
        # L2 регуляризация
        if self.weight_decay > 0:
            l2_reg = 0.0
            for param in model.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param, p=2) ** 2
            loss = loss + self.weight_decay * l2_reg
        
        return loss

