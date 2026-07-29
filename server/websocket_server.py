"""
WebSocket сервер для удалённой игры против бота.
Позволяет играть из браузера через REST API.
"""

import asyncio
import json
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    from axum import Router, post, State, Json
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    # Fallback заглушки для демонстрации
    class BaseModel:
        pass


@dataclass
class AppState:
    model: Any
    mcts: Any
    game: Any


class MoveRequest(BaseModel):
    fen: str
    time_limit: float = 5.0
    simulations: int = 100


class MoveResponse(BaseModel):
    move: str
    score: float
    depth: int
    pv: list[str]


class AnalysisResponse(BaseModel):
    best_move: str
    score: float
    depth: int
    nodes: int
    nps: float
    pv: list[str]
    table: Dict[str, float]


async def api_move(
    state: State[AppState],
    request: Json[MoveRequest]
) -> Json[MoveResponse]:
    """
    API endpoint для получения лучшего хода.
    POST /api/move
    """
    app_state = state.get()
    
    # Загрузка позиции
    game = app_state.game
    game.load_fen(request.fen)
    
    # MCTS поиск
    tensor = game.board_to_tensor()
    root = app_state.mcts.search(
        game,
        tensor,
        simulations=request.simulations,
        time_limit=request.time_limit
    )
    
    # Извлечение лучшего хода
    if root.children:
        best_child = max(root.children, key=lambda c: c.visits)
        best_move = best_child.legal_moves[0]  # Упрощённо
        score = best_child.total_values[game.get_turn()] / (best_child.visits + 1e-8)
        depth = root.visits
        pv = [str(best_move)]  # Principal variation
    else:
        best_move = None
        score = 0.0
        depth = 0
        pv = []
    
    return Json(MoveResponse(
        move=best_move or "",
        score=score,
        depth=depth,
        pv=pv
    ))


async def api_analyze(
    state: State[AppState],
    request: Json[MoveRequest]
) -> Json[AnalysisResponse]:
    """
    API endpoint для анализа позиции.
    POST /api/analyze
    """
    app_state = state.get()
    
    game = app_state.game
    game.load_fen(request.fen)
    
    import time
    start_time = time.time()
    
    tensor = game.board_to_tensor()
    root = app_state.mcts.search(
        game,
        tensor,
        simulations=request.simulations,
        time_limit=request.time_limit
    )
    
    elapsed = time.time() - start_time
    nps = root.visits / elapsed if elapsed > 0 else 0
    
    # Построение таблицы ходов
    table = {}
    for child in root.children:
        move_str = str(child.legal_moves[0]) if child.legal_moves else "?"
        score = child.total_values[game.get_turn()] / (child.visits + 1e-8)
        table[move_str] = score
    
    # Сортировка по score
    sorted_moves = sorted(table.items(), key=lambda x: x[1], reverse=True)
    best_move = sorted_moves[0][0] if sorted_moves else ""
    best_score = sorted_moves[0][1] if sorted_moves else 0.0
    
    return Json(AnalysisResponse(
        best_move=best_move,
        score=best_score,
        depth=root.visits,
        nodes=root.visits,
        nps=nps,
        pv=[best_move],
        table=dict(sorted_moves[:10])  # Топ 10 ходов
    ))


def create_app(model_path: str, config: Dict[str, Any]) -> Router:
    """
    Создание и настройка приложения.
    """
    if not HAS_FASTAPI:
        print("FastAPI/axum not installed. Install with: pip install fastapi uvicorn")
        return None
    
    # Загрузка модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device, weights_only=True)
    model.eval()
    
    # Инициализация MCTS
    from mcts import MCTS
    mcts = MCTS(
        simulations=config.get('mcts_simulations', 100),
        c_puct=config.get('c_puct', 1.5),
        fpu_value=config.get('fpu_value', -1.0),
        device=device
    )
    
    # Инициализация игры
    from game_state import PyGameState
    game = PyGameState()
    
    app_state = AppState(model=model, mcts=mcts, game=game)
    
    # Роутер
    router = Router()
    router.add_route("/api/move", post(api_move))
    router.add_route("/api/analyze", post(api_analyze))
    
    # Сохранение состояния
    router.state = app_state
    
    return router


if __name__ == "__main__":
    import yaml
    
    # Загрузка конфигурации
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Создание приложения
    app = create_app("models/best_model.pt", config.get("mcts", {}))
    
    if app:
        print("Starting server on http://localhost:8000")
        print("Endpoints:")
        print("  POST /api/move - Get best move")
        print("  POST /api/analyze - Analyze position")
        
        # Запуск через uvicorn
        try:
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
        except ImportError:
            print("uvicorn not installed. Install with: pip install uvicorn")
