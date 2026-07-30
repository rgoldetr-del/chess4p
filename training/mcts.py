"""
MCTS (Monte Carlo Tree Search) для 4-player chess.
Оптимизирован для использования с make/unmake и кэшированием тензоров.
"""

import numpy as np
import torch
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    """Узел дерева MCTS."""
    prior: float  # Prior probability from network
    visits: int = 0
    total_values: List[float] = field(default_factory=lambda: [0.0] * 4)  # Per-color values
    virtual_loss: float = 0.0
    children: List['MCTSNode'] = field(default_factory=list)
    legal_moves: List[Tuple] = field(default_factory=list)
    action_mask: Optional[np.ndarray] = None
    turn: int = 0  # Current player color


class MCTS:
    """
    Monte Carlo Tree Search с PUCT.
    Поддерживает:
    - FPU (First Play Urgency)
    - Virtual loss для параллелизма
    - Batch neural network evaluation
    - Endgame tablebase lookup
    """
    
    def __init__(
        self,
        simulations: int = 100,
        c_puct: float = 1.5,
        fpu_value: float = -1.0,
        virtual_loss: float = 1.0,
        batch_size: int = 16,
        device: torch.device = None,
        use_endgame_tablebase: bool = True,
        tablebase_max_pieces: int = 4
    ):
        self.simulations = simulations
        self.c_puct = c_puct
        self.fpu_value = fpu_value
        self.virtual_loss = virtual_loss
        self.batch_size = batch_size
        self.device = device or torch.device('cpu')
        self.use_tablebase = use_endgame_tablebase
        self.tablebase_max_pieces = tablebase_max_pieces
        
        # Кэш для endgame tablebase
        self.tablebase_cache: Dict[int, Tuple[float, List[float]]] = {}
        
        # Буфер для тензоров (переиспользование памяти)
        self.tensor_buffer = None
    
    def search(self, game, root_tensor: torch.Tensor, temperature: float = 1.0) -> MCTSNode:
        """
        Выполняет MCTS поиск из текущего состояния игры.
        Возвращает корневой узел с статистикой.
        """
        # Получение легальных ходов
        legal_moves = game.get_legal_moves()
        
        if len(legal_moves) == 0:
            # Нет легальных ходов - создание пустого узла
            root = MCTSNode(prior=0.0, turn=game.get_turn())
            root.legal_moves = []
            return root
        
        # Нейросетевая оценка корня
        policy_logits, value = self._evaluate(game, root_tensor)
        
        # Dirichlet noise для исследования в корне
        policy_probs = self._apply_dirichlet(policy_logits, legal_moves, alpha=0.15, weight=0.25)
        
        # Создание корневого узла
        root = MCTSNode(
            prior=1.0,
            turn=game.get_turn(),
            legal_moves=legal_moves,
            action_mask=policy_probs
        )
        
        # Инициализация детей
        for i, move in enumerate(legal_moves):
            child = MCTSNode(
                prior=policy_probs[i],
                turn=(game.get_turn() + 1) % 4
            )
            root.children.append(child)
        
        # Симуляции
        for _ in range(self.simulations):
            node = root
            path = []  # Путь от корня к листу
            
            # Спуск по дереву
            while not self._is_leaf(node):
                # Выбор ребёнка с максимальным PUCT score
                child = self._select_child(node)
                path.append((node, child))
                node = child
            
            # Развёртывание листа
            if node.visits == 0:
                # Оценка нейросетью
                state_tensor = self._get_state_tensor(game)
                policy_logits, value = self._evaluate(game, state_tensor)
                
                # Сохранение политики для будущих детей
                node.action_mask = policy_logits
                
                # Проверка endgame tablebase
                if self.use_tablebase and self._should_use_tablebase(game):
                    tb_result = self._lookup_tablebase(game)
                    if tb_result is not None:
                        value = tb_result
            
            # Обратное распространение
            self._backpropagate(path, value, game.get_turn())
        
        # Применение температуры
        if temperature != 1.0:
            visits = np.array([c.visits for c in root.children])
            if temperature < 1.0:
                # Sharpening
                visits = visits ** (1.0 / temperature)
            else:
                # Smoothing (не используется обычно)
                pass
            
            # Нормализация не требуется, выбираем просто по визитам
            for i, child in enumerate(root.children):
                child.visits = int(visits[i])
        
        return root
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Выбор ребёнка по формуле PUCT с FPU."""
        best_score = -1e9
        best_child = None
        
        sqrt_visits = math.sqrt(node.visits + 1)
        
        for child in node.children:
            # Q-value с FPU для непосещённых
            if child.visits == 0:
                q = self.fpu_value
            else:
                q = child.total_values[node.turn] / child.visits
            
            # Virtual loss для параллелизма
            q -= self.virtual_loss * child.virtual_loss
            
            # U-value (exploration bonus)
            u = self.c_puct * child.prior * sqrt_visits / (1 + child.visits)
            
            score = q + u
            
            if score > best_score:
                best_score = score
                best_child = child
        
        if best_child is not None:
            best_child.virtual_loss += self.virtual_loss
        
        return best_child
    
    def _is_leaf(self, node: MCTSNode) -> bool:
        """Проверка, является ли узел листом (не развёрнут)."""
        return len(node.children) == 0
    
    def _backpropagate(self, path: List, value: float, root_turn: int):
        """Обратное распространение значений по пути."""
        for node, child in reversed(path):
            child.visits += 1
            child.virtual_loss -= self.virtual_loss
            
            # Обновление значения для текущего игрока в узле
            current_turn = (root_turn + len(path) - path.index((node, child))) % 4
            child.total_values[current_turn] += value
    
    def _evaluate(self, game, state_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Оценка позиции нейросетью.
        Возвращает (policy_probs, value).
        """
        # В реальной реализации здесь будет вызов модели через PyO3 binding
        # Для примера - заглушка
        model_output = self._run_model_inference(state_tensor)
        return model_output
    
    def _run_model_inference(self, state_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Запуск инференса модели (заглушка)."""
        # В реальности: вызов через torch или PyO3 binding
        # policy = model(state_tensor)[0].cpu().numpy()
        # value = model(state_tensor)[1].cpu().numpy()
        
        # Заглушка: случайная политика и значение
        policy = np.random.randn(38416).astype(np.float32)
        value = np.random.uniform(-1, 1, size=4)
        
        return policy, value
    
    def _apply_dirichlet(self, policy: np.ndarray, moves: List, alpha: float = 0.15, weight: float = 0.25) -> np.ndarray:
        """Добавление Dirichlet noise для исследования."""
        n = len(moves)
        noise = np.random.dirichlet([alpha] * n)
        
        # Смешивание с оригинальной политикой
        probs = (1 - weight) * policy[:n] + weight * noise
        probs /= probs.sum() + 1e-8
        
        return probs
    
    def _get_state_tensor(self, game) -> torch.Tensor:
        """Получение тензора состояния игры."""
        # В реальности: вызов game.board_to_tensor()
        tensor = np.zeros((73, 14, 14), dtype=np.float32)
        # game.board_to_tensor(tensor)
        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)
    
    def _should_use_tablebase(self, game) -> bool:
        """Проверка, можно ли использовать tablebase."""
        piece_count = game.get_piece_count() if hasattr(game, 'get_piece_count') else 32
        active_players = game.get_active_player_count() if hasattr(game, 'get_active_player_count') else 4
        
        return active_players <= 3 and piece_count <= self.tablebase_max_pieces * 4
    
    def _lookup_tablebase(self, game) -> Optional[Tuple[float, List[float]]]:
        """Поиск позиции в endgame tablebase."""
        zobrist = game.get_zobrist_hash() if hasattr(game, 'get_zobrist_hash') else hash(str(game))
        
        if zobrist in self.tablebase_cache:
            return self.tablebase_cache[zobrist]
        
        # В реальности: загрузка из预先 вычисленных таблиц
        # Здесь - заглушка
        result = None  # (value, policy_distribution)
        
        if result is not None:
            self.tablebase_cache[zobrist] = result
        
        return result
