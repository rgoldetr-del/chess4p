"""
Integration tests for Chess4P AI.
Проверка консистентности между Rust и Python реализациями.
"""

import numpy as np
import torch
import pytest
from pathlib import Path


# Тестовые константы
TEST_PGN = """
[Event "Test"]
[Site "Test"]
[Date "2024.01.01"]
[Round "1"]
[White "Test"]
[Black "Test"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *
"""


class TestRustPythonConsistency:
    """Тесты на согласованность Rust и Python реализаций."""
    
    def test_board_to_tensor_shape(self):
        """Проверка формы тензора доски."""
        try:
            from chess4p_ai import PyGameState
        except ImportError:
            pytest.skip("chess4p_ai module not available")
        
        game = PyGameState()
        tensor = np.zeros((73, 14, 14), dtype=np.float32)
        game.board_to_tensor(tensor)
        
        assert tensor.shape == (73, 14, 14), f"Expected shape (73, 14, 14), got {tensor.shape}"
        assert tensor.dtype == np.float32, f"Expected float32, got {tensor.dtype}"
        
        # Проверка что turn plane заполнен
        turn = game.get_turn()
        assert tensor[32 + turn].sum() == 196, "Turn plane should have exactly 196 ones"
    
    def test_legal_moves_format(self):
        """Проверка формата легальных ходов."""
        try:
            from chess4p_ai import PyGameState
        except ImportError:
            pytest.skip("chess4p_ai module not available")
        
        game = PyGameState()
        moves = game.get_legal_moves()
        
        assert len(moves) > 0, "Initial position should have legal moves"
        assert len(moves) % 4 == 0, "Number of moves should be divisible by 4 (4 players)"
        
        # Проверка формата каждого хода
        for move in moves:
            assert isinstance(move, (list, tuple)), f"Move should be list/tuple, got {type(move)}"
            assert len(move) >= 2, f"Move should have at least 2 elements"
    
    def test_final_rewards_range(self):
        """Проверка диапазона финальных наград."""
        try:
            from chess4p_ai import PyGameState
        except ImportError:
            pytest.skip("chess4p_ai module not available")
        
        game = PyGameState()
        
        # Играем несколько случайных ходов
        for _ in range(10):
            moves = game.get_legal_moves()
            if not moves:
                break
            move = moves[0]
            game.make_move_fast(move[0][0], move[0][1], move[1][0], move[1][1])
        
        rewards = game.get_final_rewards()
        
        assert len(rewards) == 4, f"Expected 4 rewards, got {len(rewards)}"
        
        for r in rewards:
            assert -1.0 <= r <= 1.0, f"Reward should be in [-1, 1], got {r}"
    
    def test_make_unmake_consistency(self):
        """Проверка консистентности make/unmake ходов."""
        try:
            from chess4p_ai import PyGameState
        except ImportError:
            pytest.skip("chess4p_ai module not available")
        
        game = PyGameState()
        
        # Сохранение начального состояния
        initial_tensor = np.zeros((73, 14, 14), dtype=np.float32)
        game.board_to_tensor(initial_tensor)
        
        # Делаем ход
        moves = game.get_legal_moves()
        if moves:
            move = moves[0]
            undo_info = game.make_move_fast(move[0][0], move[0][1], move[1][0], move[1][1])
            
            # Отменяем ход
            game.unmake_move(undo_info)
            
            # Проверяем что состояние вернулось
            final_tensor = np.zeros((73, 14, 14), dtype=np.float32)
            game.board_to_tensor(final_tensor)
            
            np.testing.assert_array_equal(
                initial_tensor, final_tensor,
                "Board state should be identical after make/unmake"
            )


class TestModelArchitecture:
    """Тесты архитектуры модели."""
    
    def test_model_forward_shape(self):
        """Проверка формы выхода модели."""
        try:
            from training.model import ChessNet
        except ImportError:
            pytest.skip("training.model module not available")
        
        model = ChessNet(num_blocks=2, channels=64)  # Уменьшенная версия для теста
        model.eval()
        
        # Входной тензор: batch=1, 73 planes, 14x14 board
        x = torch.randn(1, 73, 14, 14)
        
        with torch.no_grad():
            policy, value = model(x)
        
        assert policy.shape == (1, 38416), f"Policy shape mismatch: {policy.shape}"
        assert value.shape == (1, 4, 4), f"Value shape mismatch: {value.shape}"
    
    def test_model_parameter_count(self):
        """Проверка количества параметров модели."""
        try:
            from training.model import ChessNet
        except ImportError:
            pytest.skip("training.model module not available")
        
        model = ChessNet(num_blocks=30, channels=256)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Ожидаемое количество параметров ~43M
        expected_min = 40_000_000
        expected_max = 50_000_000
        
        assert expected_min <= total_params <= expected_max, \
            f"Parameter count {total_params:,} outside expected range [{expected_min:,}, {expected_max:,}]"


class TestReplayBuffer:
    """Тесты replay buffer."""
    
    def test_push_sample_assert(self):
        """Проверка assert на размер маски."""
        try:
            from training.replay_buffer import PriorityReplayBuffer
        except ImportError:
            pytest.skip("training.replay_buffer module not available")
        
        buffer = PriorityReplayBuffer(capacity=100)
        
        state = np.random.randn(73, 14, 14).astype(np.float32)
        policy = np.random.randn(38416).astype(np.float32)
        value = 0.5
        mask = np.ones(38416, dtype=bool)
        
        # Должно работать
        buffer.push(state, policy, value, mask)
        
        # Неправильный размер маски должен вызвать assert
        wrong_mask = np.ones(1000, dtype=bool)
        with pytest.raises(AssertionError):
            buffer.push(state, policy, value, wrong_mask)
    
    def test_priority_sampling(self):
        """Проверка приоритизированного сэмплирования."""
        try:
            from training.replay_buffer import PriorityReplayBuffer
        except ImportError:
            pytest.skip("training.replay_buffer module not available")
        
        buffer = PriorityReplayBuffer(capacity=100, alpha=0.6)
        
        # Заполняем буфер
        for i in range(50):
            state = np.random.randn(73, 14, 14).astype(np.float32)
            policy = np.random.randn(38416).astype(np.float32)
            value = float(i) / 50
            mask = np.ones(38416, dtype=bool)
            buffer.push(state, policy, value, mask)
        
        # Сэмплирование
        batch = buffer.sample(batch_size=10, to_tensor=False)
        states, policies, values, masks, weights, indices = batch
        
        assert len(states) == 10, f"Expected 10 samples, got {len(states)}"
        assert len(weights) == 10, f"Expected 10 weights, got {len(weights)}"
        assert all(w > 0 for w in weights), "All weights should be positive"


class TestMCTS:
    """Тесты MCTS."""
    
    def test_mcts_search_returns_node(self):
        """Проверка что MCTS возвращает узел."""
        try:
            from training.mcts import MCTS, MCTSNode
        except ImportError:
            pytest.skip("training.mcts module not available")
        
        mcts = MCTS(simulations=10, c_puct=1.5, fpu_value=-1.0)
        
        # Создаём фейковую игру
        class FakeGame:
            def get_legal_moves(self):
                return [((1, 1), (2, 2)), ((1, 2), (2, 3))]
            def get_turn(self):
                return 0
        
        game = FakeGame()
        tensor = torch.zeros(1, 73, 14, 14)
        
        root = mcts.search(game, tensor, temperature=1.0)
        
        assert isinstance(root, MCTSNode), f"Expected MCTSNode, got {type(root)}"
        assert root.visits > 0, "Root should have been visited"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
