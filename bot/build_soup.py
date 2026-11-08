"""
build_soup.py — построение human_buffer.npz из чистых FFA партий.

Читает:
    clean_ffa_games.json
    clean_ffa/*.pgn4

Пишет:
    human_buffer.npz   — позиции с human move targets
    soup_meta.json     — метаинформация

Использование:
    python build_soup.py
    python build_soup.py --positions-per-game 20
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np

try:
    import chess4p_ai
except ImportError:
    print("❌ chess4p_ai не установлен")
    sys.exit(1)

# ============================================================
# Константы
# ============================================================

TENSOR_PLANES = int(getattr(chess4p_ai, "TENSOR_PLANES", 256))
ACTION_SPACE = int(getattr(chess4p_ai, "ACTION_SPACE", 38416))
POLICY_K = 256
MASK_BYTES = (ACTION_SPACE + 7) // 8

CLEAN_JSON = Path("clean_ffa_games.json")
HUMAN_NPZ = Path("human_buffer.npz")
META_JSON = Path("soup_meta.json")


# ============================================================
# Координаты <-> Action Index
# ============================================================

def coords_to_idx(fx, fy, tx, ty):
    """Конвертирует координаты (1-based) в action index."""
    return (fx - 1) + (fy - 1) * 14 + (tx - 1) * 196 + (ty - 1) * 2744


def idx_to_coords(idx):
    """Конвертирует action index в координаты (1-based)."""
    fx = (idx % 14) + 1
    fy = ((idx // 14) % 14) + 1
    tx = ((idx // 196) % 14) + 1
    ty = ((idx // 2744) % 14) + 1
    return fx, fy, tx, ty


# ============================================================
# Helpers
# ============================================================

def pack_legal_indices(legal_indices):
    """Упаковывает список легальных ходов в битовую маску."""
    mask = bytearray(MASK_BYTES)
    for idx in legal_indices:
        idx = int(idx)
        if 0 <= idx < ACTION_SPACE:
            mask[idx >> 3] |= 1 << (idx & 7)
    return np.frombuffer(mask, dtype=np.uint8).copy()


def get_state_tensor(game):
    """Получает тензор позиции."""
    state = np.zeros((TENSOR_PLANES, 14, 14), dtype=np.float32)
    game.board_to_tensor(state)
    return state


def get_value_target(game):
    """
    Получает value target для текущей позиции.
    Использует get_final_placement_soft если доступно,
    иначе fallback на rewards.
    """
    try:
        soft_flat = game.get_final_placement_soft()
        soft = np.asarray(soft_flat, dtype=np.float32).reshape(4, 4)
        return soft
    except Exception:
        pass

    # Fallback: rewards -> one-hot placement
    try:
        rewards = np.asarray(game.get_final_rewards(), dtype=np.float32)
        soft = np.zeros((4, 4), dtype=np.float32)
        # Сортируем rewards для определения мест
        order = np.argsort(-rewards)  # убывание
        for place, player_idx in enumerate(order):
            soft[player_idx, place] = 1.0
        return soft
    except Exception:
        return np.eye(4, dtype=np.float32) * 0.25


# ============================================================
# Построение human buffer
# ============================================================

def build_human_buffer(games, positions_per_game=15, seed=42):
    """
    Строит human buffer из чистых FFA партий.

    Для каждой партии:
    1. Загружаем PGN
    2. Сэмплируем позиции из реальных ходов (не skipped)
    3. Для каждой позиции: state, legal mask, human move, value target
    """
    random.seed(seed)
    np.random.seed(seed)

    all_states = []
    all_pidx = []
    all_pprob = []
    all_masks = []
    all_vals = []

    total_games = len(games)
    processed = 0
    skipped = 0
    positions_collected = 0

    t0 = time.time()

    for gi, game_record in enumerate(games):
        # Путь к PGN файлу
        pgn_path = game_record.get("clean_file") or game_record.get("file")
        if not pgn_path:
            skipped += 1
            continue

        pgn_path = Path(pgn_path)
        if not pgn_path.exists():
            skipped += 1
            continue

        try:
            pgn_text = pgn_path.read_text(encoding="utf-8")
        except Exception:
            skipped += 1
            continue

        # Загружаем партию
        game = chess4p_ai.PyGameState()
        try:
            err = game.load_pgn_verbose(pgn_text)
        except Exception as e:
            err = str(e)

        if err and not str(err).startswith("WARN"):
            skipped += 1
            continue

        move_count = game.get_move_count()
        if move_count < 10:
            skipped += 1
            continue

        # Получаем value target (один для всей партии)
        # Переходим в конец для get_final_placement_soft
        game.go_to_move(move_count)
        value_target = get_value_target(game)

        # Собираем индексы реальных ходов (не skipped)
        real_moves = []
        for ply in range(move_count):
            mv = game.get_move_at(ply)
            if mv and len(mv) == 4:
                # mv = [fx, fy, tx, ty]
                real_moves.append((ply, mv))

        if len(real_moves) < 10:
            skipped += 1
            continue

        # Сэмплируем позиции
        n_positions = min(positions_per_game, len(real_moves))
        sampled = random.sample(real_moves, n_positions)
        sampled.sort(key=lambda x: x[0])

        game_ok = True

        for ply, mv in sampled:
            try:
                # Переходим к позиции перед ходом ply
                game.go_to_move(ply)

                # Пропускаем ходы мёртвых/сдавшихся
                if game.is_turn_dead() or game.is_turn_resigned():
                    continue

                # Получаем state tensor
                state = get_state_tensor(game)

                # Получаем легальные ходы
                legal = game.get_legal_move_indices()
                if not legal:
                    continue

                # Конвертируем координаты хода в action index
                fx, fy, tx, ty = mv[0], mv[1], mv[2], mv[3]
                move_idx = coords_to_idx(fx, fy, tx, ty)

                # Проверяем, что ход в списке легальных
                legal_set = set(int(x) for x in legal)
                if move_idx not in legal_set:
                    continue

                # Упаковываем
                mask = pack_legal_indices(legal)

                # Policy target: one-hot на ход человека
                pidx = np.zeros(POLICY_K, dtype=np.int32)
                pprob = np.zeros(POLICY_K, dtype=np.float32)
                pidx[0] = move_idx
                pprob[0] = 1.0

                all_states.append(state.astype(np.float16))
                all_pidx.append(pidx)
                all_pprob.append(pprob)
                all_masks.append(mask)
                all_vals.append(value_target.astype(np.float16))

                positions_collected += 1

            except Exception:
                game_ok = False
                break

        if game_ok:
            processed += 1
        else:
            skipped += 1

        # Прогресс
        if (gi + 1) % 50 == 0 or gi == total_games - 1:
            elapsed = time.time() - t0
            speed = (gi + 1) / elapsed if elapsed > 0 else 0
            print(
                f"⏳ [{gi+1}/{total_games}] "
                f"OK: {processed} | Skip: {skipped} | "
                f"Positions: {positions_collected} | "
                f"Speed: {speed:.1f} games/s",
                flush=True,
            )

    return {
        "states": all_states,
        "pidx": all_pidx,
        "pprob": all_pprob,
        "masks": all_masks,
        "vals": all_vals,
        "processed": processed,
        "skipped": skipped,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build human buffer")
    parser.add_argument(
        "--positions-per-game",
        type=int,
        default=15,
        help="Количество позиций для сэмплирования из каждой партии",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🍲 ПОСТРОЕНИЕ СОЛЯНКИ")
    print("=" * 70)

    # Загружаем список чистых партий
    if not CLEAN_JSON.exists():
        print(f"❌ Не найден {CLEAN_JSON}")
        print("Сначала запусти: python filter_clean_ffa.py")
        return

    data = json.loads(CLEAN_JSON.read_text(encoding="utf-8"))
    games = data.get("games", [])

    print(f"📂 Чистых FFA партий: {len(games)}")
    print(f"📊 Позиций на партию: {args.positions_per_game}")
    print()

    # Строим human buffer
    print("🔨 Строим human buffer...")
    print()

    result = build_human_buffer(
        games,
        positions_per_game=args.positions_per_game,
        seed=args.seed,
    )

    states = result["states"]
    pidx = result["pidx"]
    pprob = result["pprob"]
    masks = result["masks"]
    vals = result["vals"]

    print()
    print("=" * 70)
    print("📊 ИТОГИ HUMAN BUFFER")
    print("=" * 70)
    print(f"Обработано партий: {result['processed']}")
    print(f"Пропущено партий: {result['skipped']}")
    print(f"Всего позиций: {len(states)}")

    if not states:
        print("❌ Нет позиций для сохранения")
        return

    # Сохраняем
    print()
    print(f"💾 Сохраняем {HUMAN_NPZ}...")

    np.savez_compressed(
        HUMAN_NPZ,
        states=np.stack(states),
        pidx=np.stack(pidx),
        pprob=np.stack(pprob),
        masks=np.stack(masks),
        vals=np.stack(vals),
    )

    # Размер файла
    file_size_mb = HUMAN_NPZ.stat().st_size / (1024 * 1024)
    print(f"✅ Сохранено: {HUMAN_NPZ.resolve()}")
    print(f"📦 Размер: {file_size_mb:.1f} MB")

    # Метаинформация
    meta = {
        "human_buffer": {
            "positions": len(states),
            "games_processed": result["processed"],
            "games_skipped": result["skipped"],
            "positions_per_game": args.positions_per_game,
            "tensor_planes": TENSOR_PLANES,
            "action_space": ACTION_SPACE,
            "policy_k": POLICY_K,
            "file_size_mb": round(file_size_mb, 1),
        },
    }

    META_JSON.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"💾 Мета: {META_JSON.resolve()}")
    print()
    print("=" * 70)
    print("🏁 ГОТОВО")
    print("=" * 70)


if __name__ == "__main__":
    main()