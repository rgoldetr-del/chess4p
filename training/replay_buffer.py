"""
Priority Replay Buffer с importance sampling для AlphaZero.
Поддерживает PER (Prioritized Experience Replay).
"""

import numpy as np
import torch
from typing import Tuple, List, Optional
from collections import deque


class PriorityReplayBuffer:
    """
    Replay buffer с приоритетами на основе TD-error (или loss).
    alpha: степень приоритизации (0 = равномерное, 1 = полная приоритизация)
    beta: коэффициент для importance sampling (annealed to 1.0)
    """
    
    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        
        # Буферы для данных (numpy для эффективности памяти)
        # state: (73, 14, 14) float32 = ~71 KB
        # policy: (38416,) float32 = ~154 KB
        # value: (4,) float32 = 16 bytes
        # mask: (38416,) bool = ~38 KB
        # Итого: ~263 KB на образец
        
        self.states = deque(maxlen=capacity)
        self.policies = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)
        self.masks = deque(maxlen=capacity)
        
        # Приоритеты
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        
        # Максимальный приоритет для новых образцов
        self.max_priority = 1.0
    
    def push(
        self,
        state: np.ndarray,
        policy: np.ndarray,
        value: float,
        mask: np.ndarray
    ):
        """
        Добавляет образец в буфер.
        Assert: mask.shape[0] == 38416
        """
        assert mask.shape[0] == 38416, f"Expected mask size 38416, got {mask.shape[0]}"
        assert state.shape == (73, 14, 14), f"Invalid state shape: {state.shape}"
        assert policy.shape == (38416,), f"Invalid policy shape: {policy.shape}"
        
        # Сохранение данных
        self.states.append(state.copy())
        self.policies.append(policy.copy())
        self.values.append(value)
        self.masks.append(mask.copy())
        
        # Установка максимального приоритета для нового образца
        self.priorities[self.pos] = self.max_priority
        
        # Обновление позиции и размера
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(
        self,
        batch_size: int,
        beta: Optional[float] = None,
        to_tensor: bool = False,
        device: Optional[torch.device] = None
    ) -> Tuple:
        """
        Сэмплирование с приоритетами.
        Возвращает: (states, policies, values, masks, weights, indices)
        """
        if self.size == 0:
            raise RuntimeError("Replay buffer is empty")
        
        # Вычисление beta (annealing)
        if beta is None:
            frame = min(self.size, self.beta_frames)
            beta = self.beta_start + (1.0 - self.beta_start) * (frame / self.beta_frames)
        
        # Получение приоритетов для доступных образцов
        priorities = self.priorities[:self.size]
        
        # Вероятности сэмплирования
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Индексы
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Нормализация
        
        # Извлечение данных
        states = np.array([self.states[i] for i in indices])
        policies = np.array([self.policies[i] for i in indices])
        values = np.array([self.values[i] for i in indices])
        masks = np.array([self.masks[i] for i in indices])
        
        if to_tensor:
            states = torch.from_numpy(states).float()
            policies = torch.from_numpy(policies).float()
            values = torch.from_numpy(values).float()
            masks = torch.from_numpy(masks)
            weights = torch.from_numpy(weights).float()
            
            if device:
                states = states.to(device)
                policies = policies.to(device)
                values = values.to(device)
                masks = masks.to(device)
                weights = weights.to(device)
            
            return states, policies, values, masks, weights, indices
        else:
            return states, policies, values, masks, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Обновление приоритетов после вычисления loss."""
        for idx, prio in zip(indices, priorities):
            if 0 <= idx < self.size:
                self.priorities[idx] = abs(prio) + 1e-6  # epsilon для стабильности
                self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self) -> int:
        return self.size
    
    def drain_queue(self, queue, max_items: int = 1000) -> int:
        """
        Быстрое извлечение элементов из очереди без блокировки.
        Аналог empty() + get_nowait() в цикле.
        """
        count = 0
        while count < max_items:
            try:
                item = queue.get_nowait()
                if len(item) == 4:
                    state, policy, value, mask = item
                    self.push(state, policy, value, mask)
                    count += 1
            except Exception:
                break
        return count
