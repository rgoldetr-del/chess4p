"""
Export utilities for Chess4P.
PGN to GIF, JSON, video conversion.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


def pgn_to_json(pgn_path: str, output_path: str):
    """
    Конвертация PGN файла в JSON формат.
    """
    moves = []
    current_game = {
        "headers": {},
        "moves": []
    }
    
    with open(pgn_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Headers
            if line.startswith('['):
                key, value = line[1:-1].split(' ', 1)
                current_game["headers"][key] = value.strip('"')
            
            # Moves
            elif '.' not in line and not line.startswith('['):
                move_list = line.split()
                for move in move_list:
                    if move.isdigit() or move in ['*', '1-0', '0-1', '1/2-1/2']:
                        continue
                    current_game["moves"].append(move)
                
                moves.append(current_game.copy())
                current_game = {"headers": {}, "moves": []}
    
    with open(output_path, 'w') as f:
        json.dump(moves, f, indent=2)
    
    print(f"Exported {len(moves)} games to {output_path}")


def pgn_to_gif(
    pgn_path: str,
    output_path: str,
    piece_paths: Dict[str, str],
    board_size: int = 512,
    frame_duration: int = 500
):
    """
    Конвертация PGN в GIF анимацию.
    Требует PIL/Pillow.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("PIL not installed. Install with: pip install pillow")
        return
    
    # Загрузка игры
    from game_state import PyGameState
    game = PyGameState()
    
    frames = []
    
    # Чтение PGN
    with open(pgn_path, 'r') as f:
        pgn_content = f.read()
    
    game.load_pgn(pgn_content)
    
    total_moves = game.get_move_count()
    
    for i in range(total_moves + 1):
        game.go_to_move(i)
        
        # Рендеринг доски
        img = render_board_frame(game, piece_paths, board_size)
        frames.append(img)
    
    # Сохранение GIF
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0
        )
        print(f"Saved GIF with {len(frames)} frames to {output_path}")


def render_board_frame(
    game,
    piece_paths: Dict[str, str],
    size: int = 512
):
    """
    Рендеринг одного кадра доски.
    """
    from PIL import Image, ImageDraw
    
    square_size = size // 14
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    
    # Отрисовка клеток
    for row in range(14):
        for col in range(14):
            x = col * square_size
            y = row * square_size
            
            # Цвет клетки
            is_dark = (row + col) % 2 == 1
            color = (180, 180, 180) if is_dark else (240, 240, 240)
            
            draw.rectangle([x, y, x + square_size, y + square_size], fill=color)
    
    # Отрисовка фигур
    board = game.get_board_array()  # Предполагаемый метод
    for row in range(14):
        for col in range(14):
            piece = board[row][col]
            if piece != 0:
                # Определение цвета и типа фигуры
                color_idx = (piece - 1) // 6
                piece_type = (piece - 1) % 6
                
                color_names = ['red', 'blue', 'green', 'yellow']
                piece_names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
                
                color_name = color_names[color_idx]
                piece_name = piece_names[piece_type]
                
                # Путь к спрайту
                key = f"{color_name}-{piece_name}"
                if key in piece_paths:
                    piece_img = Image.open(piece_paths[key])
                    piece_img = piece_img.resize((square_size, square_size))
                    
                    x = col * square_size
                    y = row * square_size
                    
                    img.paste(piece_img, (x, y), piece_img.convert('RGBA'))
    
    return img


def generate_stats(pgn_path: str) -> Dict[str, Any]:
    """
    Генерация статистики по PGN файлу.
    """
    stats = {
        "total_games": 0,
        "total_moves": 0,
        "avg_game_length": 0.0,
        "wins_by_color": {0: 0, 1: 0, 2: 0, 3: 0},
        "checkmates": 0,
        "stalemates": 0,
        "resignations": 0,
        "avg_score_spread": 0.0
    }
    
    game_lengths = []
    score_spreads = []
    
    with open(pgn_path, 'r') as f:
        for line in f:
            if line.startswith('[Result '):
                stats["total_games"] += 1
                
                # Парсинг результата
                result = line.strip()[8:-1]
                if result != '*':
                    # Обработка результата
                    pass
    
    if stats["total_games"] > 0:
        stats["avg_game_length"] = stats["total_moves"] / stats["total_games"]
    
    return stats


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python export_utils.py <command> <input> <output>")
        print("Commands:")
        print("  pgn-to-json <input.pgn> <output.json>")
        print("  pgn-to-gif <input.pgn> <output.gif>")
        print("  stats <input.pgn>")
        sys.exit(1)
    
    command = sys.argv[1]
    input_file = sys.argv[2]
    
    if command == "pgn-to-json":
        output_file = sys.argv[3] if len(sys.argv) > 3 else "output.json"
        pgn_to_json(input_file, output_file)
    
    elif command == "pgn-to-gif":
        output_file = sys.argv[3] if len(sys.argv) > 3 else "output.gif"
        piece_paths = {
            "red-pawn": "assets/pieces/red-pawn.svg",
            "red-knight": "assets/pieces/red-knight.svg",
            # ... остальные фигуры
        }
        pgn_to_gif(input_file, output_file, piece_paths)
    
    elif command == "stats":
        stats = generate_stats(input_file)
        print(json.dumps(stats, indent=2))
    
    else:
        print(f"Unknown command: {command}")
