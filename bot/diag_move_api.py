"""
diag_move_api.py — диагностика API для извлечения ходов.
Показывает, что возвращают get_move_at, go_to_move, get_move_count.
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    import chess4p_ai
except ImportError:
    print("❌ chess4p_ai не установлен")
    sys.exit(1)


def main():
    # Берём первую чистую партию
    clean_json = Path("clean_ffa_games.json")
    if not clean_json.exists():
        print("❌ clean_ffa_games.json не найден")
        return

    data = json.loads(clean_json.read_text(encoding="utf-8"))
    games = data.get("games", [])

    if not games:
        print("❌ Нет партий")
        return

    game_record = games[0]
    pgn_path = game_record.get("clean_file") or game_record.get("file")

    if not pgn_path:
        print("❌ Нет пути к PGN")
        return

    pgn_path = Path(pgn_path)
    if not pgn_path.exists():
        print(f"❌ Файл не найден: {pgn_path}")
        return

    print(f"📂 Тестовая партия: {pgn_path}")
    print()

    pgn_text = pgn_path.read_text(encoding="utf-8")

    game = chess4p_ai.PyGameState()
    try:
        err = game.load_pgn_verbose(pgn_text)
    except Exception as e:
        err = str(e)

    if err:
        print(f"⚠ load_pgn_verbose: {err}")
    else:
        print("✅ load_pgn_verbose: OK")

    print()

    # Базовая информация
    move_count = game.get_move_count()
    real_move_count = game.get_real_move_count()
    ply_count = game.get_ply_count()

    print(f"get_move_count()      = {move_count}")
    print(f"get_real_move_count() = {real_move_count}")
    print(f"get_ply_count()       = {ply_count}")
    print()

    # Проверяем get_move_at для первых 10 ходов
    print("=== get_move_at для первых 10 полуходов ===")
    for ply in range(min(10, move_count)):
        try:
            mv = game.get_move_at(ply)
            print(f"  ply {ply:3d}: {repr(mv)} (type={type(mv).__name__})")
        except Exception as e:
            print(f"  ply {ply:3d}: ERROR: {e}")

    print()

    # Проверяем go_to_move
    print("=== go_to_move + get_legal_move_indices ===")
    for ply in range(min(5, move_count)):
        try:
            game.go_to_move(ply)

            turn = game.get_turn()
            legal = game.get_legal_move_indices()
            is_dead = game.is_turn_dead()
            is_resigned = game.is_turn_resigned()

            print(f"  ply {ply:3d}: turn={turn}, legal={len(legal)}, "
                  f"dead={is_dead}, resigned={is_resigned}")

            # Проверяем get_last_move_idx
            try:
                last_idx = game.get_last_move_idx()
                print(f"          get_last_move_idx() = {repr(last_idx)}")
            except Exception as e:
                print(f"          get_last_move_idx() ERROR: {e}")

            # Проверяем get_move_at после go_to_move
            try:
                mv = game.get_move_at(ply)
                print(f"          get_move_at({ply}) = {repr(mv)}")
            except Exception as e:
                print(f"          get_move_at({ply}) ERROR: {e}")

        except Exception as e:
            print(f"  ply {ply:3d}: go_to_move ERROR: {e}")

    print()

    # Проверяем get_legal_moves (может возвращать объекты Move)
    print("=== get_legal_moves (первые 5) ===")
    try:
        game.go_to_move(0)
        moves = game.get_legal_moves()
        print(f"  Всего легальных ходов: {len(moves)}")
        for i, mv in enumerate(moves[:5]):
            print(f"  [{i}] {repr(mv)} (type={type(mv).__name__})")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()

    # Проверяем make_move и make_move_fast_idx
    print("=== make_move_fast_idx тест ===")
    try:
        game.go_to_move(0)
        legal = game.get_legal_move_indices()
        if legal:
            test_idx = legal[0]
            print(f"  Пробуем make_move_fast_idx({test_idx})")

            # Клонируем чтобы не портить состояние
            clone = game.clone_state()
            clone.make_move_fast_idx(test_idx)
            print(f"  ✅ make_move_fast_idx OK")
            print(f"  Новый turn: {clone.get_turn()}")
            print(f"  Новый ply_count: {clone.get_ply_count()}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()

    # Проверяем board_to_tensor
    print("=== board_to_tensor тест ===")
    try:
        game.go_to_move(0)
        state = np.zeros((256, 14, 14), dtype=np.float32)
        game.board_to_tensor(state)
        nonzero = np.count_nonzero(state)
        print(f"  ✅ board_to_tensor OK, nonzero={nonzero}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()

    # Проверяем get_final_placement_soft
    print("=== get_final_placement_soft тест ===")
    try:
        # Переходим в конец
        game.go_to_move(move_count)
        soft = game.get_final_placement_soft()
        soft_arr = np.asarray(soft, dtype=np.float32).reshape(4, 4)
        print(f"  ✅ get_final_placement_soft OK")
        print(f"  Shape: {soft_arr.shape}")
        print(f"  Values:\n{soft_arr}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()
    print("=" * 60)
    print("Диагностика завершена.")
    print("Пришли этот вывод, и я исправлю build_soup.py")
    print("=" * 60)


if __name__ == "__main__":
    main()