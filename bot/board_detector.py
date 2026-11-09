"""board_detector.py — распознавание позиции по шаблонам.

Улучшения:
- пороги берутся из config.DETECTION_CFG;
- серые и цветные шаблоны сравниваются вместе;
- live-фигура не становится dead только из-за слабого цвета;
- клетка автоматически ресайзится к размеру шаблонов;
- безопасный matchTemplate.
"""

import os
import cv2
import numpy as np
import time

from config import (
    TEMPLATES_DIR,
    COLOR_ORDER,
    COLOR_TO_IDX,
    DETECTION_CFG,
    DEBUG_DIR,
)

from calibration import (
    square_to_region,
    load_calibration,
)

from coordinates import is_valid_square, screen_color_to_engine


PIECE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]


# ============================================================
# Templates
# ============================================================

class PieceTemplates:
    """Загружает шаблоны и ресайзит их под размер клетки."""

    PIECE_NAMES = PIECE_NAMES

    def __init__(self, cal=None):
        self.cal = cal or load_calibration()

        self.cell = (
            max(1, int(round(float(self.cal["cell_w"])))),
            max(1, int(round(float(self.cal["cell_h"])))),
        )

        self.templates = {}       # (color, piece) -> [color_img]
        self.gray_templates = []  # (piece, gray_img)

        self.promoted_templates = {c: [] for c in COLOR_ORDER}
        self._load(TEMPLATES_DIR)

        

    def _load(self, tdir):
        # Цветные шаблоны.
        for color in COLOR_ORDER:
            cdir = os.path.join(tdir, color)
            if not os.path.isdir(cdir):
                continue

            for piece in self.PIECE_NAMES:
                self.templates[(color, piece)] = []

                for fname in os.listdir(cdir):
                    if fname.startswith(piece) and fname.endswith(".png"):
                        img = cv2.imread(os.path.join(cdir, fname))
                        if img is None:
                            continue

                        if img.shape[:2][::-1] != self.cell:
                            img = cv2.resize(
                                img,
                                self.cell,
                                interpolation=cv2.INTER_AREA,
                            )

                        self.templates[(color, piece)].append(img)

        # Серые шаблоны.
        gdir = os.path.join(tdir, "gray")
        if os.path.isdir(gdir):
            for piece in self.PIECE_NAMES:
                for fname in os.listdir(gdir):
                    if fname.startswith(piece) and fname.endswith(".png"):
                        img = cv2.imread(os.path.join(gdir, fname))
                        if img is None:
                            continue

                        if img.shape[:2][::-1] != self.cell:
                            img = cv2.resize(
                                img,
                                self.cell,
                                interpolation=cv2.INTER_AREA,
                            )

                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        # Если файл содержит promoted, считаем шаблон превращённым ферзём.
                        promoted = "promoted" in fname.lower()

                        self.gray_templates.append((piece, gray, promoted))

        # Цветные шаблоны превращённых ферзей.
        pdir = os.path.join(tdir, "promoted_queens")

        if os.path.isdir(pdir):
            for fname in os.listdir(pdir):
                if not fname.endswith(".png"):
                    continue

                low = fname.lower()

                for color in COLOR_ORDER:
                    if low.startswith(color):
                        img = cv2.imread(os.path.join(pdir, fname))

                        if img is None:
                            continue

                        if img.shape[:2][::-1] != self.cell:
                            img = cv2.resize(
                                img,
                                self.cell,
                                interpolation=cv2.INTER_AREA,
                            )

                        self.promoted_templates[color].append(img)
                        break

    def prune_empty(self):
        self.templates = {k: v for k, v in self.templates.items() if v}


# ============================================================
# Helpers
# ============================================================

def _colored_ratio(cell_img, cfg: dict) -> float:
    """Доля пикселей с высокой насыщенностью (цветные пиксели)."""
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    sat_thresh = int(cfg.get("sat_thresh", 100))
    return float(np.mean(hsv[:, :, 1] > sat_thresh))


def _is_dead_cell(cell_img, cfg: dict) -> bool:
    """Серая фигура = почти нет цветных пикселей."""
    dead_sat_ratio = float(cfg.get("dead_sat_ratio", 0.12))
    return _colored_ratio(cell_img, cfg) < dead_sat_ratio

def _get_inner_cell(cell_img, ratio=0.15):
    """Обрезает края клетки, чтобы избежать влияния рамок (шах, ход)."""
    h, w = cell_img.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    return cell_img[ch:h-ch, cw:w-cw]

def _is_empty_cell(gray: np.ndarray, cfg: dict) -> bool:
    """Дешёвая проверка: в клетке нет фигуры."""
    empty_std = float(cfg.get("empty_std", 10.0))
    canny_low = int(cfg.get("canny_low", 90))
    canny_high = int(cfg.get("canny_high", 180))
    edge_ratio_min = float(cfg.get("edge_ratio_min", 0.015))

    if gray.std() < empty_std:
        return True

    edges = cv2.Canny(gray, canny_low, canny_high)
    edge_ratio = float((edges > 0).sum()) / float(max(1, gray.size))

    return edge_ratio < edge_ratio_min


def _classify_color(cell_img, cfg: dict):
    """
    Определяет цвет живой фигуры по тону насыщенных пикселей.

    Возвращает (color_name | None, scores).
    """
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0].astype(np.int32)   # 0..179
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    sat_thresh = int(cfg.get("sat_thresh", 100))
    mask = (s > sat_thresh) & (v > 40)

    n = int(mask.sum())

    if n < 10:
        return None, {}

    hs = h[mask]

    # Диапазоны hue в OpenCV (0-179).
    scores = {
        "red":    float(np.mean((hs <= 10) | (hs >= 160))),
        "yellow": float(np.mean((hs > 14) & (hs <= 35))),
        "green":  float(np.mean((hs > 35) & (hs <= 85))),
        "blue":   float(np.mean((hs > 90) & (hs <= 140))),
    }

    best = max(scores, key=scores.get)

    min_ratio = float(cfg.get("color_min_ratio", 0.4))

    if scores[best] < min_ratio:
        return None, scores

    return best, scores

def _safe_match(cell_img: np.ndarray, tmpl: np.ndarray) -> float:
    """Безопасный matchTemplate для одинаковых размеров."""
    try:
        res = cv2.matchTemplate(cell_img, tmpl, cv2.TM_CCOEFF_NORMED)
        val = float(res.max())
        if np.isfinite(val):
            return val
        return -1.0
    except cv2.error:
        return -1.0


def detect_piece_in_cell(cell_img, templates: PieceTemplates, cfg: dict = None):
    """Обёртка: перекрашивает абсолютный цвет экрана в цвет движка
    (при повёрнутой доске сидения смещаются)."""
    result = _detect_piece_in_cell_raw(cell_img, templates, cfg)
    if result is None:
        return None
    color, piece, promoted, dead = result
    if color != "gray":
        color = screen_color_to_engine(color)
    return (color, piece, promoted, dead)


def _detect_piece_in_cell_raw(
    cell_img,
    templates: PieceTemplates,
    cfg: dict = None,
):
    """
    Возвращает (color_name, piece_name, promoted, dead) или None.

    Живые фигуры:
        1) цвет определяется по hue (не может перепутаться);
        2) тип фигуры матчится только внутри шаблонов этого цвета.

    Серые фигуры:
        матчатся в grayscale.
    """
    if cell_img is None or cell_img.size == 0:
        return None

    if cfg is None:
        cfg = DETECTION_CFG

    color_threshold = float(cfg.get("color_threshold", 0.80))
    gray_threshold = float(cfg.get("gray_threshold", 0.78))

    target_w, target_h = templates.cell

    if cell_img.shape[1] != target_w or cell_img.shape[0] != target_h:
        cell_img = cv2.resize(
            cell_img,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    if _is_empty_cell(gray, cfg):
        return None

    # 🛡 Обрезаем края клетки, чтобы красный фон/рамка шаха не ломали цвет
    inner_cell = _get_inner_cell(cell_img, ratio=0.15)

    # ============================================================
    # Мёртвая фигура: только серые шаблоны
    # ============================================================
    if _is_dead_cell(inner_cell, cfg):  # <-- Передаем inner_cell
        ...
        
    # ============================================================
    # Живая фигура: цвет по hue, тип по шаблонам этого цвета
    # ============================================================
    color, _scores = _classify_color(inner_cell, cfg)  # <-- Передаем inner_cell
    if color is None:
        return None

    # ============================================================
    # Мёртвая фигура: только серые шаблоны
    # ============================================================
    if _is_dead_cell(cell_img, cfg):
        best_gray_score = -1.0
        best_gray_piece = None
        best_gray_promoted = False

        for item in templates.gray_templates:
            if len(item) == 3:
                piece, tmpl, gray_promoted = item
            else:
                piece, tmpl = item
                gray_promoted = False

            if tmpl.shape[:2] != gray.shape[:2]:
                continue

            val = _safe_match(gray, tmpl)

            if val > best_gray_score:
                best_gray_score = val
                best_gray_piece = piece
                best_gray_promoted = bool(gray_promoted)

        if (
            best_gray_piece is not None
            and best_gray_score >= gray_threshold
        ):
            return (
                None,
                best_gray_piece,
                best_gray_promoted,
                True,
            )

        return None

    # ============================================================
    # Живая фигура: цвет по hue, тип по шаблонам этого цвета
    # ============================================================
    color, _scores = _classify_color(cell_img, cfg)

    if color is None:
        return None

    best_val = -1.0
    best_piece = None

    for (c, piece), tmpl_list in templates.templates.items():
        if c != color:
            continue

        for tmpl in tmpl_list:
            if tmpl.shape[:2] != cell_img.shape[:2]:
                continue

            val = _safe_match(cell_img, tmpl)

            if val > best_val:
                best_val = val
                best_piece = piece

    if best_piece is None or best_val < color_threshold:
        return None

    promoted = False

    # Promoted — по шаблонам превращённых ферзей этого же цвета.
    if best_piece == "queen":
        prom_val = -1.0

        for tmpl in templates.promoted_templates.get(color, []):
            if tmpl.shape[:2] != cell_img.shape[:2]:
                continue

            v = _safe_match(cell_img, tmpl)

            if v > prom_val:
                prom_val = v

        if prom_val > best_val:
            promoted = True

    return (color, best_piece, promoted, False)


def detect_promoted_marker(cell_img, cfg: dict = None) -> bool:
    """
    Определяет promoted-маркер на ферзе.

    Важно:
        Сейчас это консервативный вариант.
        Если promoted_enabled=False, всегда возвращает False.
    """
    if cfg is None:
        cfg = DETECTION_CFG

    if not bool(cfg.get("promoted_enabled", False)):
        return False

    promoted_white_ratio = float(cfg.get("promoted_white_ratio", 0.18))

    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Более узкая зона, чем раньше.
    roi_h = max(1, h // 5)
    roi_w = max(1, w // 4)

    roi = gray[0:roi_h, 0:roi_w]

    # Лёгкое сглаживание, чтобы убрать одиночный шум.
    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    # Более жёсткий порог белого.
    white_ratio = float((roi > 215).sum()) / float(max(1, roi.size))

    return white_ratio > promoted_white_ratio


# ============================================================
# Board detector
# ============================================================

class BoardDetector:
    """Полная детекция доски."""

    def __init__(self, cal=None):
        self.cal = cal or load_calibration()
        self.templates = PieceTemplates(self.cal)
        self.templates.prune_empty()

        self._dbg_last = {}
        self._dbg_count = 0

        self.valid_squares = [
            (x, y)
            for y in range(1, 15)
            for x in range(1, 15)
            if is_valid_square(x, y)
        ]

    def detect_full_board(self, screenshot):
        """
        board[y][x] = (color_idx, piece_type, promoted, dead) или None.

        piece_type:
            0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king.

        color_idx:
            0..3 для живых фигур;
            -1 для серых фигур.
        """
        board = [[None] * 15 for _ in range(15)]

        for x, y in self.valid_squares:
            left, top, w, h = square_to_region(x, y, self.cal)

            if w <= 0 or h <= 0:
                continue

            cell = screenshot[top:top + h, left:left + w]
            if cell.size == 0:
                continue

            result = detect_piece_in_cell(
                cell,
                self.templates,
                DETECTION_CFG,
            )

            if result is None:
                self._maybe_debug_cell(cell, x, y)
                continue

            color, piece, promoted, dead = result

            # УБРАНО: переопределение promoted по белому маркеру.
            # Раньше здесь был detect_promoted_marker, который давал
            # false positive для синих и зелёных ферзей.
            # Теперь promoted определяется только шаблонами
            # в detect_piece_in_cell.

            piece_idx = PIECE_NAMES.index(piece)
            color_idx = COLOR_TO_IDX.get(color, -1)
            board[y][x] = (color_idx, piece_idx, promoted, dead)

        return board

    def cell_hashes(self, screenshot):
        """Быстрый хеш клеток: средний цвет внутренности клетки.
        ~10-20 мс на всю доску (полное распознавание ~200-400 мс)."""
        hashes = {}
        for x, y in self.valid_squares:
            left, top, w, h = square_to_region(x, y, self.cal)
            if w <= 0 or h <= 0:
                continue
            ix = max(1, int(w * 0.18))
            iy = max(1, int(h * 0.18))
            cell = screenshot[top + iy:top + h - iy, left + ix:left + w - ix]
            if cell.size == 0:
                continue
            hashes[(x, y)] = cell.mean(axis=(0, 1))
        return hashes

    def changed_cells(self, screenshot, prev_hashes, threshold=12.0):
        """Возвращает (hashes, список изменившихся (x, y)).
        threshold — средняя абсолютная разница каналов (0..255)."""
        hashes = self.cell_hashes(screenshot)
        changed = []
        for key, h in hashes.items():
            ph = prev_hashes.get(key)
            if ph is None or float(np.abs(h - ph).mean()) > threshold:
                changed.append(key)
        return hashes, changed

    def detect_full_board_smart(self, screenshot, prev_board, prev_hashes, threshold=12.0):
        """Доска с точечной детекцией: распознаются только ИЗМЕНИВШИЕСЯ
        клетки, неизменные копируются из prev_board.

        Стабильные клетки перестают мигать из-за шума детекции, а полная
        детекция нужна только для 2-4 клеток вместо 196.
        """
        hashes, changed = self.changed_cells(screenshot, prev_hashes, threshold)
        changed_set = set(changed)
        board = [[None] * 15 for _ in range(15)]

        for x, y in self.valid_squares:
            if (
                (x, y) not in changed_set
                and prev_board is not None
                and prev_board[y] is not None
            ):
                board[y][x] = prev_board[y][x]
                continue

            left, top, w, h = square_to_region(x, y, self.cal)
            if w <= 0 or h <= 0:
                continue

            cell = screenshot[top:top + h, left:left + w]
            if cell.size == 0:
                continue

            result = detect_piece_in_cell(cell, self.templates, DETECTION_CFG)
            if result is None:
                self._maybe_debug_cell(cell, x, y)
                continue

            color, piece, promoted, dead = result
            board[y][x] = (
                COLOR_TO_IDX.get(color, -1),
                PIECE_NAMES.index(piece),
                promoted,
                dead,
            )

        return board, hashes

    def _maybe_debug_cell(self, cell_img, x, y, cooldown=10.0, max_files=300):
        """Клетка не пустая, но не распознана -> сохранить как дебаг."""
        try:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            if _is_empty_cell(gray, DETECTION_CFG):
                return
            now = time.time()
            if now - self._dbg_last.get((x, y), 0.0) < cooldown:
                return
            if self._dbg_count >= max_files:
                return
            self._dbg_last[(x, y)] = now
            self._dbg_count += 1
            out_dir = os.path.join(DEBUG_DIR, "unknown_cells")
            os.makedirs(out_dir, exist_ok=True)
            ts = time.strftime("%H%M%S")
            cv2.imwrite(os.path.join(out_dir, f"cell_x{x}_y{y}_{ts}.png"), cell_img)
        except Exception:
            pass

def iter_pieces(board):
    """Генератор (x, y, cell) для непустых клеток."""
    if board is None:
        return
    for y in range(1, 15):
        if board[y] is None:
            continue
        for x in range(1, 15):
            if board[y][x] is not None:
                yield x, y, board[y][x]