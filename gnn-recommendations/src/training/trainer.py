"""
Trainer класс для обучения моделей рекомендательных систем.

Обеспечивает:
- Обучение модели с BPR Loss
- Валидацию
- Early stopping
- Сохранение чекпоинтов
- Логирование метрик
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
import time
from collections import defaultdict

from .losses import BPRLoss
from .metrics import compute_all_metrics, compute_metrics_from_topk

# Импорт датасета
from ..data.dataset import RecommendationDataset


class Trainer:
    """
    Класс для обучения моделей рекомендательных систем.
    
    Поддерживает:
    - Обучение с BPR Loss
    - Валидацию на validation set
    - Early stopping
    - Сохранение чекпоинтов
    - Логирование
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataset: RecommendationDataset,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        Инициализация Trainer.
        
        Args:
            model: модель для обучения (наследуется от BaseRecommender)
            dataset: датасет с train/valid/test данными
            config: конфигурация обучения
            device: устройство для вычислений (CPU/GPU)
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        
        # Устройство
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Оптимизатор (преобразуем параметры в правильные типы)
        learning_rate = float(config.get('learning_rate', 0.001))
        weight_decay = float(config.get('weight_decay', 1e-4))
        
        # Параметры обучения (преобразуем в правильные типы) - СНАЧАЛА!
        self.batch_size = int(config.get('batch_size', 2048))
        self.epochs = int(config.get('epochs', 300))
        self.eval_every = int(config.get('eval_every', 10))
        self.negative_samples = int(config.get('negative_samples', 1))
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler с warmup
        self.use_scheduler = config.get('use_scheduler', True)
        self.warmup_epochs = int(config.get('warmup_epochs', 5))
        self.base_lr = learning_rate
        
        if self.use_scheduler:
            # Cosine annealing scheduler после warmup
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - self.warmup_epochs,
                eta_min=learning_rate * 0.01
            )
        else:
            self.scheduler = None
        
        # Loss функция
        self.loss_fn = BPRLoss()
        
        # Gradient clipping для стабильности обучения GNN
        self.max_grad_norm = float(config.get('max_grad_norm', 1.0))
        
        # Early stopping (преобразуем в правильные типы)
        self.early_stopping = config.get('early_stopping', {})
        self.patience = int(self.early_stopping.get('patience', 20))
        self.min_delta = float(self.early_stopping.get('min_delta', 0.0001))
        self.validation_metrics = config.get('validation_metrics', ['recall@10', 'ndcg@10'])
        self.early_stopping_metric = config.get('early_stopping_metric', 'recall@10')
        
        # Состояние обучения
        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.train_losses = []
        self.valid_metrics = []
        
        # Пути для сохранения
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'results/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_train_items_by_user(self) -> Dict[int, set]:
        """Создаёт маппинг user_id -> set(train_items)."""
        if not hasattr(self, '_train_items_cache'):
            train_items = defaultdict(set)
            train_data = self.dataset.train_data
            if train_data is not None:
                if isinstance(train_data, list):
                    for row in train_data:
                        user_id = int(row['userId'])
                        item_id = int(row['itemId'])
                        train_items[user_id].add(item_id)
                else:
                    for _, row in train_data.iterrows():
                        user_id = int(row['userId'])
                        item_id = int(row['itemId'])
                        train_items[user_id].add(item_id)
            self._train_items_cache = train_items
        return self._train_items_cache
    
    def _sample_batch(
        self,
        train_data: List[Tuple[int, int]],
        n_items: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Сэмплирует батч для обучения.
        
        Args:
            train_data: список (user_id, item_id) пар
            n_items: количество айтемов
        
        Returns:
            Tuple из (users, pos_items, neg_items)
        """
        batch_size = min(self.batch_size, len(train_data))
        indices = torch.randint(0, len(train_data), (batch_size,))
        
        users = []
        pos_items = []
        neg_items = []
        
        # Создаём словарь user -> set(positive_items) для быстрой проверки
        if not hasattr(self, '_user_pos_items'):
            self._user_pos_items = defaultdict(set)
            for user_id, item_id in train_data:
                self._user_pos_items[user_id].add(item_id)
        
        for idx in indices:
            user_id, pos_item_id = train_data[idx.item()]
            
            # Сэмплируем отрицательный айтем (НЕ положительный для этого пользователя)
            # Повторяем сэмплирование пока не найдём negative item
            pos_items_for_user = self._user_pos_items[user_id]
            user_negs = []
            for _ in range(self.negative_samples):
                neg_item_id = torch.randint(0, n_items, (1,)).item()
                for _ in range(10):
                    if neg_item_id not in pos_items_for_user:
                        break
                    neg_item_id = torch.randint(0, n_items, (1,)).item()
                user_negs.append(neg_item_id)
            
            users.append(user_id)
            pos_items.append(pos_item_id)
            neg_items.append(user_negs)
        
        return (
            torch.tensor(users, device=self.device),
            torch.tensor(pos_items, device=self.device),
            torch.tensor(neg_items, device=self.device)
        )
    
    def train_epoch(self) -> float:
        """
        Одна эпоха обучения.
        
        Returns:
            Средний loss за эпоху
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        # Получаем train данные
        train_data = self.dataset.train_data
        
        # Преобразуем в список пар (user_id, item_id)
        if train_data is None:
            # Если train_data не загружен, загружаем из файла
            train_file = self.dataset.processed_data_path / "train.txt"
            if train_file.exists():
                import pandas as pd
                train_data = pd.read_csv(train_file, sep='\t', header=None, names=['userId', 'itemId'])
            else:
                raise ValueError("Train данные не найдены!")
        
        if isinstance(train_data, list):
            train_pairs = [(int(row['userId']), int(row['itemId'])) for row in train_data]
        else:
            # Если DataFrame
            train_pairs = list(zip(
                train_data['userId'].astype(int),
                train_data['itemId'].astype(int)
            ))
        
        # Получаем adjacency matrix
        adj_matrix = self.dataset.get_torch_adjacency(normalized=True)
        adj_matrix = adj_matrix.to(self.device)
        
        # Получаем все embeddings один раз (для эффективности)
        if hasattr(self.model, 'get_all_embeddings'):
            user_emb, item_emb = self.model.get_all_embeddings(adj_matrix)
        else:
            user_emb, item_emb = self.model(adj_matrix)
        
        # Обучение по батчам
        n_batches_total = len(train_pairs) // self.batch_size + 1
        
        for batch_idx in range(n_batches_total):
            # Сэмплируем батч
            users, pos_items, neg_items = self._sample_batch(
                train_pairs,
                self.dataset.n_items
            )
            
            if len(users) == 0:
                continue
            
            # Вычисляем scores
            pos_scores = (user_emb[users] * item_emb[pos_items]).sum(dim=1)
            if neg_items.dim() == 1:
                neg_scores = (user_emb[users] * item_emb[neg_items]).sum(dim=1)
            else:
                neg_scores = (user_emb[users].unsqueeze(1) * item_emb[neg_items]).sum(dim=2)
            
            # Loss
            loss = self.loss_fn(pos_scores, neg_scores)
            if hasattr(self.model, 'get_regularization_loss'):
                loss = loss + self.model.get_regularization_loss()
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping для стабильности
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # ВАЖНО: Обновляем embeddings после каждого батча
            # Иначе мы используем старые embeddings и модель не обучается
            with torch.no_grad():
                if hasattr(self.model, 'get_all_embeddings'):
                    user_emb, item_emb = self.model.get_all_embeddings(adj_matrix)
                else:
                    user_emb, item_emb = self.model(adj_matrix)
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def validate(self) -> Dict[str, float]:
        """
        Валидация модели на validation set.
        
        Returns:
            Словарь с метриками
        """
        self.model.eval()
        
        with torch.no_grad():
            adj_matrix = self.dataset.get_torch_adjacency(normalized=True)
            adj_matrix = adj_matrix.to(self.device)
            if hasattr(self.model, 'get_all_embeddings'):
                user_emb, item_emb = self.model.get_all_embeddings(adj_matrix)
            else:
                user_emb, item_emb = self.model(adj_matrix)

            valid_data = self.dataset.valid_data
            ground_truth = defaultdict(list)
            if valid_data is None:
                valid_file = self.dataset.processed_data_path / "valid.txt"
                if valid_file.exists():
                    import pandas as pd
                    valid_data = pd.read_csv(valid_file, sep='\t', header=None, names=['userId', 'itemId'])
                else:
                    return {}

            if isinstance(valid_data, list):
                for row in valid_data:
                    ground_truth[int(row['userId'])].append(int(row['itemId']))
            else:
                for _, row in valid_data.iterrows():
                    ground_truth[int(row['userId'])].append(int(row['itemId']))

            eval_users = sorted(ground_truth.keys())
            if not eval_users:
                return {}

            k_values = self._parse_k_values(self.validation_metrics)
            max_k = max(k_values) if k_values else 10
            topk_items = []
            train_items = self._get_train_items_by_user()
            batch_size = max(1, min(self.batch_size, 2048))

            for i in range(0, len(eval_users), batch_size):
                batch_users = eval_users[i:i + batch_size]
                batch_tensor = torch.tensor(batch_users, device=self.device)
                scores = user_emb[batch_tensor] @ item_emb.T
                for row_idx, user_id in enumerate(batch_users):
                    if user_id in train_items:
                        items = list(train_items[user_id])
                        if items:
                            scores[row_idx, items] = float('-inf')
                batch_topk = torch.topk(scores, k=max_k, dim=1).indices
                topk_items.append(batch_topk.cpu())

            topk_items = torch.cat(topk_items, dim=0)
            metrics = compute_metrics_from_topk(
                topk_items=topk_items,
                user_ids=eval_users,
                ground_truth=ground_truth,
                n_items=self.dataset.n_items,
                k_values=k_values if k_values else [10]
            )
            return metrics

    @staticmethod
    def _parse_k_values(metrics_list: List[str]) -> List[int]:
        k_values = set()
        for metric in metrics_list:
            if '@' in metric:
                try:
                    k_values.add(int(metric.split('@')[-1]))
                except ValueError:
                    continue
        return sorted(k_values)
    
    def train(self) -> Dict:
        """
        Полный цикл обучения.
        
        Returns:
            Словарь с результатами обучения
        """
        print(f"\n{'='*60}")
        print(f"НАЧАЛО ОБУЧЕНИЯ")
        print(f"{'='*60}")
        print(f"Модель: {self.model.__class__.__name__}")
        print(f"Устройство: {self.device}")
        print(f"Параметров: {self.model.get_parameters_count():,}")
        print(f"Эпох: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            
            # Learning rate warmup
            if epoch <= self.warmup_epochs:
                # Линейный warmup от 0 до base_lr
                warmup_lr = self.base_lr * (epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            elif self.scheduler is not None and epoch > self.warmup_epochs:
                # После warmup используем scheduler
                self.scheduler.step()
            
            # Обучение
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Валидация
            if epoch % self.eval_every == 0 or epoch == 1:
                valid_metrics = self.validate()
                self.valid_metrics.append(valid_metrics)
                
                # Основная метрика для early stopping
                current_metric = valid_metrics.get(self.early_stopping_metric, 0.0)
                
                # Текущий learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Логирование
                ndcg10 = valid_metrics.get('ndcg@10', 0.0)
                print(f"Epoch {epoch:3d}/{self.epochs} | "
                      f"LR: {current_lr:.6f} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"{self.early_stopping_metric}: {current_metric:.4f} | "
                      f"NDCG@10: {ndcg10:.4f}")
                
                # Early stopping
                if current_metric > self.best_metric + self.min_delta:
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, valid_metrics)
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping на эпохе {epoch}")
                    print(f"Лучшая метрика: {self.best_metric:.4f} на эпохе {self.best_epoch}")
                    break
        
        training_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"{'='*60}")
        print(f"Время обучения: {training_time:.2f} секунд")
        print(f"Лучшая метрика: {self.best_metric:.4f} на эпохе {self.best_epoch}")
        print(f"{'='*60}\n")
        
        return {
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'valid_metrics': self.valid_metrics
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """
        Сохраняет чекпоинт модели.
        
        Args:
            epoch: номер эпохи
            metrics: метрики для сохранения
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Загружает чекпоинт модели.
        
        Args:
            checkpoint_path: путь к чекпоинту
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', 0.0)

