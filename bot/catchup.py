"""catchup.py — catch-up по подсветке ходов + burst-детекция в live.

Chess.com 4PC подсвечивает from/to квадраты цветом сходившего игрока.
Это позволяет читать ход за ~5мс без полной детекции доски.
"""
import time
import cv2
import numpy as np
import chess4p_ai
import threading

from config import DEBUG_DIR
from calibration import square_to_region
from coordinates import is_valid_square, game_to_screen, idx_to_coords
from board_detector import iter_pieces
from screen_capture import capture_screen
from state_sync import try_set_board_resync, GameRecorder, replay_resync

# Hue-диапазоны (OpenCV 0..179): red, blue, yellow, green
_HUE = {
    0: [(0, 10), (160, 179)],
    1: [(90, 140)],
    2: [(15, 35)],
    3: [(36, 85)],
}


def _color_masks(H, S, V, sat=70, val=60):
    satmask = (S > sat) & (V > val)
    out = {}
    for ci, ranges in _HUE.items():
        m = np.zeros_like(satmask)
        for lo, hi in ranges:
            m |= (H >= lo) & (H <= hi)
        out[ci] = m & satmask
    return out


def detect_move_highlight(shot, cal, min_ratio=0.45, ring=0.18, max_squares=4):
    """Ищет квадраты, чья КАЙМА тонирована цветом игрока.
    Кайма, а не весь квадрат — чтобы не реагировать на сами фигуры.
    Возвращает dict(color=int, squares=[(x,y),...]) или None."""
    hsv = cv2.cvtColor(shot, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    masks = _color_masks(H, S, V)

    hits = {}
    for y in range(1, 15):
        for x in range(1, 15):
            if not is_valid_square(x, y):
                continue
            left, top, w, h = square_to_region(x, y, cal)
            mx = max(2, int(w * ring)); my = max(2, int(h * ring))
            ring_px = 2 * w * my + 2 * (h - 2 * my) * mx
            if ring_px <= 0:
                continue
            for ci, m in masks.items():
                cell = m[top:top + h, left:left + w]
                cnt = (cell[:my, :].sum() + cell[h - my:, :].sum() +
                       cell[my:h - my, :mx].sum() + cell[my:h - my, w - mx:w].sum())
                if float(cnt) / float(ring_px) >= min_ratio:
                    hits.setdefault(ci, []).append((x, y, float(cnt) / ring_px))

    best = None
    for ci, lst in hits.items():
        if len(lst) >= 2:
            lst = sorted(lst, key=lambda t: -t[2])[:max_squares]
            if best is None or len(lst) > len(best["squares"]):
                best = {"color": ci, "squares": [(x, y) for x, y, _ in lst]}
    return best


def match_highlight_to_legal(game, hl):
    """action-idx легального хода, у которого from и to входят в подсветку."""
    if game.get_turn() != hl["color"]:
        return None
    sq = set(hl["squares"])
    for idx in game.get_legal_move_indices():
        fx, fy, tx, ty = idx_to_coords(int(idx))
        if game_to_screen(fx, fy) in sq and game_to_screen(tx, ty) in sq:
            return int(idx)
    return None


def board_matches_engine(game, board):
    """Число расхождений между детектированной доской и движком."""
    T = np.zeros((256, 14, 14), np.float32)
    game.board_to_tensor(T)
    miss = 0
    eng_count = [0, 0, 0, 0]
    for c in range(4):
        for p in range(6):
            eng_count[c] += int((T[c * 6 + p] > 0.5).sum())
        eng_count[c] += int((T[73 + c] > 0.5).sum())
    det_count = [0, 0, 0, 0]
    for x, y, (c, p, prom, dead) in iter_pieces(board):
        if dead or c is None or c < 0:
            continue
        det_count[c] += 1
        plane = (73 + c) if prom else (c * 6 + p)
        if T[plane, 15 - y - 1, x - 1] < 0.5:
            miss += 1
    miss += sum(abs(a - b) for a, b in zip(eng_count, det_count))
    return miss


def run_catchup(game, detector, tracker, recorder, timer, engine,
                max_total=600.0, resync_cooldown=0.5):
    """Catch-up: юзер листает партию, бот трекает по подсветке.
    Выход — только по ENTER."""
    from state_sync import try_set_board_resync
    print("📼 Catch-up режим")
    print("   1. Прокрути партию до позиции, с которой бот должен играть.")
    print("   2. Во время прокрутки бот трекает ходы по подсветке.")
    print("   3. Когда позиция готова — нажми ENTER в этом окне.")

    # Поток слушает Enter, не блокируя захват экрана
    stop_event = threading.Event()
    def _enter_watcher():
        try:
            input()
        except EOFError:
            pass
        stop_event.set()
    threading.Thread(target=_enter_watcher, daemon=True).start()

    cal = detector.cal
    last_key = None
    tracked = gaps = 0
    last_resync = 0.0
    t0 = time.time()

    while not stop_event.is_set():
        if time.time() - t0 > max_total:
            print("  ⚠ Catch-up таймаут (10 мин) — выхожу")
            break
        shot = capture_screen()
        now = time.time()
        hl = detect_move_highlight(shot, cal)
        if hl:
            key = (hl["color"], tuple(sorted(hl["squares"])))
            if key != last_key:
                last_key = key
                idx = match_highlight_to_legal(game, hl)
                if idx is not None:
                    fx, fy, tx, ty = idx_to_coords(idx)
                    game.make_move_fast_idx(idx)
                    recorder.move((fx, fy), (tx, ty), turn=hl["color"])
                    tracked += 1
                else:
                    # пропущен ход(ы) -> точечный resync
                    gaps += 1
                    if now - last_resync > resync_cooldown:
                        last_resync = now
                        board = detector.detect_full_board(shot)
                        tracker.update(board)
                        dead = game.get_dead(); res = game.get_resigned()
                        if try_set_board_resync(game, board, tracker, dead, res,
                                                turn=hl["color"]):
                            engine.reset_root()
        time.sleep(0.004)

    # ============ ENTER нажат: подтверждаем позицию ============
    print("⏳ ENTER получен. Подтверждаю позицию...")
    time.sleep(0.5)  # даём UI отрисовать финальный кадр
    board = detector.detect_full_board(capture_screen())
    tracker.update(board)

    miss = board_matches_engine(game, board)
    if miss > 2:
        print(f"  📼 верификация: mismatch={miss} -> set_board")
        dead = game.get_dead(); res = game.get_resigned()
        t = timer.read_stable() if timer.is_ready() else None
        if try_set_board_resync(game, board, tracker, dead, res,
                                turn=t if t is not None else game.get_turn()):
            engine.reset_root()
        else:
            print("  ⚠ set_board не удался -> PGN replay")
            g2 = replay_resync(recorder)
            if g2 is not None:
                game = g2
                engine.reset_root()
    else:
        print(f"  📼 верификация: OK (mismatch={miss})")

    print(f"📼 Catch-up завершён: tracked={tracked}, gaps={gaps}")
    return game, recorder, board


def burst_detect_move(game, detector, turn, last_board, diff_fn,
                      max_wait=8.0, consensus_fn=None):
    """После смены таймера: снимки в цикле, пока ход не распознан.
    Каждый снимок сразу проверяется подсветкой и диффом (быстрая реакция).
    Со второй половины окна периодически пробуем консенсус из нескольких
    снимков (надёжнее, когда одиночные детекции шумят)."""
    t0 = time.time()
    last_consensus = 0.0
    while time.time() - t0 < max_wait:
        if (
            consensus_fn is not None
            and time.time() - t0 > max_wait * 0.4
            and time.time() - last_consensus > 1.5
        ):
            last_consensus = time.time()
            board = consensus_fn()
            mv = diff_fn(game, last_board, board, turn)
            if mv:
                return board, mv
        shot = capture_screen()
        hl = detect_move_highlight(shot, detector.cal)
        if hl and hl["color"] == turn:
            idx = match_highlight_to_legal(game, hl)
            if idx is not None:
                return detector.detect_full_board(shot), idx_to_coords(idx)
        board = detector.detect_full_board(shot)
        mv = diff_fn(game, last_board, board, turn)
        if mv:
            return board, mv
        time.sleep(0.08)
    return detector.detect_full_board(capture_screen()), None