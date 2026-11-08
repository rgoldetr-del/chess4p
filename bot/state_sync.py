"""state_sync.py — синхронизация внутренней игры с экраном.

Улучшения:
- set_board вызывается с promoted=bool;
- есть fallback live-only entries;
- GameRecorder хранит и PGN-токены, и структурированные события;
- OCR-проверка счёта осталась опциональной.
"""

import numpy as np

from board_detector import iter_pieces
from calibration import (
    load_calibration,
    square_to_region,
)


PIECE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]


def _sq(x, y):
    return f"{chr(ord('a') + x - 1)}{y}"


# ============================================================
# KingTracker
# ============================================================

class KingTracker:
    """Последние известные позиции живых королей по цветам."""

    def __init__(self):
        self.last = {}
        self.prev_live_count = {}
        self.status_hits = {}

    def update(self, board):
        if board is None:
            return
        for x, y, (color, piece, _prom, dead) in iter_pieces(board):
            if piece == 5 and not dead and color >= 0:
                self.last[color] = (x, y)


    def nearest_color(self, x, y, max_dist=8):
        best, bd = None, 10 ** 9

        for c, (kx, ky) in self.last.items():
            d = abs(kx - x) + abs(ky - y)

            if d < bd:
                bd, best = d, c

        return best if bd <= max_dist else None


# ============================================================
# Статусы игроков по доске
# ============================================================

def detect_statuses(board, tracker: KingTracker, min_confirm: int = 1):
    if board is None:
        return {}
    tracker.update(board)

    live_count = {c: 0 for c in range(4)}
    gray_kings = []
    live_kings = set()

    for x, y, (color, piece, _prom, dead) in iter_pieces(board):
        if dead:
            if piece == 5:
                gray_kings.append((x, y))
        else:
            if color >= 0:
                live_count[color] += 1

                if piece == 5:
                    live_kings.add(color)

    raw = {}

    # 1) Серые короли -> dead.
    for (x, y) in gray_kings:
        c = tracker.nearest_color(x, y, max_dist=8)

        if c is not None:
            raw[c] = "dead"

    # 2) Армия исчезла, король живой -> resigned
    for c in range(4):
        if c in raw:
            continue
        if c in live_kings and live_count[c] == 1:
            prev = tracker.prev_live_count.get(c, 0)
            if prev > 1:
                raw[c] = "resigned"

    # 3) НОВОЕ: у цвета не осталось ЖИВЫХ фигур -> мёртв.
    # Надёжнее распознавания серого короля: если раньше фигуры были,
    # а теперь все серые — цвет выбит (мат/пат).
    for c in range(4):
        if c in raw:
            continue
        prev = tracker.prev_live_count.get(c, 0)
        if prev > 0 and live_count[c] == 0:
            raw[c] = "dead"

    tracker.prev_live_count = dict(live_count)

    if min_confirm <= 1:
        tracker.status_hits = {}
        return raw

    # Опциональное подтверждение статуса в нескольких кадрах.
    confirmed = {}
    new_hits = {}

    for c, kind in raw.items():
        key = (c, kind)
        new_hits[key] = tracker.status_hits.get(key, 0) + 1

        if new_hits[key] >= min_confirm:
            confirmed[c] = kind

    tracker.status_hits = new_hits

    return confirmed


# ============================================================
# OCR счёта
# ============================================================

def read_scores_ocr(cal=None, corner_scale=3):
    """
    Возвращает scores[4] или None, если pytesseract недоступен.

    Важно:
        OCR используется только как sanity check.
    """
    try:
        import pytesseract
        import cv2
    except ImportError:
        return None

    from screen_capture import capture_screen

    cal = cal or load_calibration()

    # Углы счёта.
    # Если цвета перепутаны, эту таблицу нужно поправить под ваш UI.
    corners = {
        0: (1, 12),   # red
        1: (1, 1),    # blue
        2: (12, 1),   # yellow
        3: (12, 12),  # green
    }

    shot = capture_screen()
    scores = [None, None, None, None]

    for c, (cx, cy) in corners.items():
        left, top, w, h = square_to_region(cx, cy, cal)

        region = shot[
            top:top + h * corner_scale,
            left:left + w * corner_scale,
        ]

        if region.size == 0:
            continue

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]

        try:
            text = pytesseract.image_to_string(
                gray,
                config=(
                    "--psm 7 "
                    "-c tessedit_char_whitelist=0123456789"
                ),
            ).strip()
        except Exception:
            text = ""

        if text.isdigit():
            scores[c] = int(text)

    return scores


def scores_diverged(game, ocr_scores, tol=2):
    """True, если OCR-счёт заметно расходится с внутренним."""
    if ocr_scores is None or any(s is None for s in ocr_scores):
        return False

    internal = game.get_scores()

    return any(
        abs(internal[i] - ocr_scores[i]) > tol
        for i in range(4)
    )


# ============================================================
# GameRecorder
# ============================================================

class GameRecorder:
    """
    Пишет PGN-токены и одновременно хранит структурированные события.

    Токены:
        a1-a2  обычный ход
        Z      skip dead turn
        R      resign
        S      stalemate/dead sync marker
    """

    def __init__(self):
        self.tokens = []
        self.events = []

    def move(self, f, t, turn=None):
        self.tokens.append(f"{_sq(*f)}-{_sq(*t)}")
        self.events.append(
            {
                "type": "move",
                "turn": turn,
                "from": tuple(f),
                "to": tuple(t),
            }
        )

    def skip(self, turn=None):
        self.tokens.append("Z")
        self.events.append(
            {
                "type": "skip",
                "turn": turn,
            }
        )

    def resign(self, turn=None):
        self.tokens.append("R")
        self.events.append(
            {
                "type": "resign",
                "turn": turn,
            }
        )

    def stalemate(self, turn=None):
        self.tokens.append("S")
        self.events.append(
            {
                "type": "stalemate",
                "turn": turn,
            }
        )

    def pgn(self):
        return " ".join(self.tokens)


# ============================================================
# Resync helpers
# ============================================================

def board_to_entries(board, tracker: KingTracker, include_dead=True):
    """Все фигуры с жесткой валидацией, чтобы не уронить Rust-движок."""
    from coordinates import screen_to_game
    entries = []
    for x, y, (color, piece, prom, dead) in iter_pieces(board):
        if dead and not include_dead:
            continue
            
        if not (1 <= x <= 14 and 1 <= y <= 14):
            continue
        if not (0 <= piece <= 5):
            continue

        ci = color
        if ci is None or ci < 0 or ci > 3:
            ci = tracker.nearest_color(x, y, max_dist=10)
            if ci is None or ci < 0 or ci > 3:
                continue
                
        # Переводим экранные координаты во внутренние (инверсия Y)
        gx, gy = screen_to_game(x, y)
        entries.append((int(gx), int(gy), int(ci), int(piece), bool(prom)))
    return entries

def try_set_board_resync(game, board, tracker, dead, resigned, turn=None):
    from coordinates import screen_to_game
    from config import COLOR_ORDER

    # ============================================================
    # 1. Собираем живые и серые фигуры
    # ============================================================
    live_count = {c: 0 for c in range(4)}
    gray_items = []  # (x, y, piece, promoted)

    for x, y, (color, piece, prom, is_dead) in iter_pieces(board):
        if is_dead:
            gray_items.append((x, y, piece, prom))
        else:
            if color is not None and color >= 0:
                live_count[color] += 1

    # ============================================================
    # 2. Сначала определяем цвета серых королей
    # ============================================================
    gray_king_color = {}

    for x, y, piece, prom in gray_items:
        if piece == 5:
            c = tracker.nearest_color(x, y, max_dist=12)
            if c is not None and c >= 0 and live_count[c] == 0:
                dead[c] = True
                gray_king_color[(x, y)] = c
                print(
                    f"  [sync] 🧠 Визуально определен мертвый цвет: "
                    f"{COLOR_ORDER[c]} (серый король на ({x},{y}))"
                )

    # ============================================================
    # 3. Определяем цвета остальных серых фигур
    # ============================================================
    gray_color = {}

    for x, y, piece, prom in gray_items:
        # Серый король уже может иметь цвет
        if piece == 5 and (x, y) in gray_king_color:
            gray_color[(x, y)] = gray_king_color[(x, y)]
            continue

        # Ищем ближайшего серого короля с известным цветом
        best_d = 10 ** 9
        best_c = None

        for (kx, ky), kc in gray_king_color.items():
            d = abs(kx - x) + abs(ky - y)
            if d < best_d:
                best_d = d
                best_c = kc

        if best_c is not None:
            gray_color[(x, y)] = best_c
        else:
            # Если серого короля рядом нет, пробуем nearest_color
            c = tracker.nearest_color(x, y, max_dist=12)
            if c is not None and c >= 0:
                if live_count[c] == 0:
                    dead[c] = True
                    gray_color[(x, y)] = c
                elif resigned[c]:
                    # Для resigned цвета не-короли тоже могут быть серыми,
                    # но сам цвет еще не dead.
                    gray_color[(x, y)] = c

    # ============================================================
    # 4. Собираем entries: живые + серые с определённым цветом
    # ============================================================
    entries = []
    kings_found = [False, False, False, False]

    for x, y, (color, piece, prom, is_dead) in iter_pieces(board):
        if not (1 <= x <= 14 and 1 <= y <= 14):
            continue
        if not (0 <= piece <= 5):
            continue

        if is_dead:
            ci = gray_color.get((x, y))
            if ci is None:
                continue
        else:
            ci = color
            if ci is None or ci < 0 or ci > 3:
                ci = tracker.nearest_color(x, y, max_dist=10)
            if ci is None or ci < 0 or ci > 3:
                continue

        gx, gy = screen_to_game(x, y)
        entries.append((int(gx), int(gy), int(ci), int(piece), bool(prom)))

        if piece == 5 and not is_dead:
            kings_found[ci] = True

    # ============================================================
    # 5. Восстанавливаем королей живых цветов
    # ============================================================
    for i in range(4):
        if dead[i] or resigned[i]:
            kings_found[i] = True
            continue

        if not kings_found[i]:
            if i in tracker.last:
                kx, ky = tracker.last[i]
                gx, gy = screen_to_game(kx, ky)
                entries.append((int(gx), int(gy), int(i), 5, False))
                kings_found[i] = True
                print(
                    f"  [sync] ⚠ Король {COLOR_ORDER[i]} "
                    f"восстановлен по трекеру на ({kx},{ky})."
                )
            else:
                print(f"  [sync] ❌ Король {COLOR_ORDER[i]} не найден!")
                return False

    if not entries:
        return False

    if turn is None:
        turn = int(game.get_turn())

    scores = [int(s) for s in game.get_scores()]
    dead_b = [bool(d) for d in dead]
    resigned_b = [bool(r) for r in resigned]

    try:
        res = game.set_board(entries, int(turn), scores, dead_b, resigned_b)
        return res is not False
    except Exception as e:
        print(f"  [sync] set_board exception: {e}")
        return False

def replay_resync(recorder: GameRecorder):
    """
    Fallback: пересоздать игру из записанных токенов.

    По диагностике load_pgn_verbose принимает Z/R/S без ошибки.
    """
    import chess4p_ai  # type: ignore[import-untyped]

    g = chess4p_ai.PyGameState()

    try:
        err = g.load_pgn_verbose(recorder.pgn())
    except Exception as e:
        print(f"  [sync] replay exception: {e}")
        return None

    if err:
        print(f"  [sync] replay failed: {err}")
        return None

    return g