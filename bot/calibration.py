"""calibration.py — калибровка доски 4PC: 2 клика + пиксельная подгонка сетки.

Улучшения:
- пути берутся из config.py;
- square_to_region использует заранее посчитанные границы клеток;
- поддержка старых calibration_data.json;
- меньше накопления ошибки округления.
"""

import cv2
import sys
import numpy as np
import json
import os

from config import (
    BOARD_SIZE,
    CALIBRATION_FILE,
    CALIBRATION_SCREENSHOT,
    BOARD_CROP_FILE,
)

from screen_capture import capture_screen
from coordinates import game_to_screen  # noqa: F401  (совместимость)

GRID = int(BOARD_SIZE)
_EDGE_CACHE = {}


# ============================================================
# Calibration normalization
# ============================================================

def normalize_calibration(data: dict) -> dict:
    """Приводит старый/новый calibration dict к единому формату."""
    data = dict(data)

    # board_left / board_top
    if "board_left" not in data or "board_top" not in data:
        tl = data.get("board_tl", [0, 0])
        data["board_left"] = int(round(float(tl[0])))
        data["board_top"] = int(round(float(tl[1])))

    width = float(data.get("board_width", 0) or 0)
    height = float(data.get("board_height", 0) or 0)
    cell_size = float(data.get("cell_size", 0) or 0)
    cell_w = float(data.get("cell_w", 0) or 0)
    cell_h = float(data.get("cell_h", 0) or 0)

    # board_width
    if width <= 0:
        if cell_size > 0:
            width = cell_size * GRID
        elif cell_w > 0:
            width = cell_w * GRID

    # board_height
    if height <= 0:
        if cell_size > 0:
            height = cell_size * GRID
        elif cell_h > 0:
            height = cell_h * GRID

    data["board_width"] = int(round(width))
    data["board_height"] = int(round(height))

    # cell_w / cell_h
    if cell_w <= 0:
        cell_w = width / float(GRID) if width > 0 else 1.0

    if cell_h <= 0:
        cell_h = height / float(GRID) if height > 0 else 1.0

    data["cell_w"] = float(cell_w)
    data["cell_h"] = float(cell_h)

    # cell_size для совместимости с move_executor
    if cell_size <= 0:
        cell_size = (cell_w + cell_h) / 2.0

    data["cell_size"] = int(round(cell_size))

    # Старые угловые точки, если их нет.
    left = float(data["board_left"])
    top = float(data["board_top"])
    w = float(data["board_width"])
    h = float(data["board_height"])

    data.setdefault("board_tl", [left, top])
    data.setdefault("board_tr", [left + w, top])
    data.setdefault("board_bl", [left, top + h])
    data.setdefault("board_br", [left + w, top + h])

    return data


def load_calibration(path: str = None) -> dict:
    path = path or CALIBRATION_FILE

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} не найден. Сначала: python main.py --calibrate"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return normalize_calibration(data)


def save_calibration(data: dict, path: str = None):
    path = path or CALIBRATION_FILE
    data = normalize_calibration(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


# ============================================================
# Grid geometry
# ============================================================

def _grid_edges(cal: dict):
    """Возвращает (xs, ys) границы клеток."""
    key = (
        int(round(float(cal.get("board_left", 0)))),
        int(round(float(cal.get("board_top", 0)))),
        int(round(float(cal.get("board_width", 0)))),
        int(round(float(cal.get("board_height", 0)))),
    )

    if key not in _EDGE_CACHE:
        left = float(cal.get("board_left", 0))
        top = float(cal.get("board_top", 0))
        width = float(cal.get("board_width", 0))
        height = float(cal.get("board_height", 0))

        xs = np.linspace(left, left + width, GRID + 1)
        ys = np.linspace(top, top + height, GRID + 1)

        _EDGE_CACHE[key] = (
            np.round(xs).astype(int),
            np.round(ys).astype(int),
        )

    return _EDGE_CACHE[key]


def square_to_region(x: int, y: int, cal: dict):
    """Возвращает (left, top, width, height) клетки в экранных координатах."""
    x = max(1, min(GRID, int(x)))
    y = max(1, min(GRID, int(y)))

    xs, ys = _grid_edges(cal)

    left = int(xs[x - 1])
    right = int(xs[x])
    top = int(ys[y - 1])
    bottom = int(ys[y])

    w = max(1, right - left)
    h = max(1, bottom - top)

    return left, top, w, h


def square_to_pixel(x: int, y: int, cal: dict):
    """Центр клетки в экранных координатах."""
    x = max(1, min(GRID, int(x)))
    y = max(1, min(GRID, int(y)))

    xs, ys = _grid_edges(cal)

    cx = int((xs[x - 1] + xs[x]) / 2)
    cy = int((ys[y - 1] + ys[y]) / 2)

    return cx, cy


def pixel_to_square(px: int, py: int, cal: dict):
    """Экранный пиксель -> клетка доски."""
    xs, ys = _grid_edges(cal)

    x = int(np.searchsorted(xs, int(px), side="right"))
    y = int(np.searchsorted(ys, int(py), side="right"))

    x = max(1, min(GRID, x))
    y = max(1, min(GRID, y))

    return x, y


def board_region(cal: dict):
    """Регион доски для capture_screen."""
    return (
        int(cal.get("board_left", 0)),
        int(cal.get("board_top", 0)),
        int(cal.get("board_width", 0)),
        int(cal.get("board_height", 0)),
    )


# ============================================================
# Calibration UI
# ============================================================

def _draw_grid(vis, gx, gy, cell, hint):
    """Рисует сетку 14x14, вырезанные углы и красную рамку."""
    if cell is not None and cell > 0:
        w = cell * GRID
        h = cell * GRID

        for i in range(GRID + 1):
            cv2.line(
                vis,
                (gx + i * cell, gy),
                (gx + i * cell, gy + h),
                (0, 255, 0),
                1,
            )
            cv2.line(
                vis,
                (gx, gy + i * cell),
                (gx + w, gy + i * cell),
                (0, 255, 0),
                1,
            )

        # Вырезанные углы 3x3.
        for (cx, cy) in [(0, 0), (11, 0), (0, 11), (11, 11)]:
            cv2.rectangle(
                vis,
                (gx + cx * cell, gy + cy * cell),
                (gx + (cx + 3) * cell, gy + (cy + 3) * cell),
                (0, 0, 0),
                -1,
            )

        cv2.rectangle(vis, (gx, gy), (gx + w, gy + h), (0, 0, 255), 2)

    # Плашка с координатами.
    cv2.rectangle(vis, (5, 5), (1000, 38), (0, 0, 0), -1)
    cv2.putText(
        vis,
        f"X:{gx} Y:{gy} Cell:{cell}px",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Подсказка внизу.
    sh = vis.shape[0]
    cv2.rectangle(vis, (5, sh - 33), (1000, sh - 5), (0, 0, 0), -1)
    cv2.putText(
        vis,
        hint,
        (12, sh - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )


def calibrate(screenshot=None, save=True):
    """Калибровка: 2 клика (TL, BR) -> подгонка сетки -> сохранение."""
    if screenshot is None:
        if "--no-pause" in sys.argv:
            screenshot = capture_screen()
        else:
            print("Разверните chess.com (4PC), доска в начальной позиции.")
            print("Сверните VS Code и прочее, чтобы не мешало. Нажмите Enter...")
            input()
            screenshot = capture_screen()

    full = screenshot.copy()
    win = "Calibrate 4PC board"

    # ---------- ФАЗА 1: два клика ----------
    points = []

    def click_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((int(x), int(y)))

    print("")
    print("[1/2] Кликните ВЕРХНИЙ-ЛЕВЫЙ угол доски (в пустой вырезанный угол).")
    print("      Затем НИЖНИЙ-ПРАВЫЙ угол. ESC = отмена.")

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, click_cb)

    while len(points) < 2:
        vis = full.copy()
        _draw_grid(
            vis,
            0,
            0,
            0,
            "Click TL corner, then BR corner  |  ESC = cancel",
        )

        for i, (px, py) in enumerate(points):
            cv2.circle(vis, (px, py), 7, (0, 0, 255), -1)
            cv2.putText(
                vis,
                ["TL", "BR"][i],
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            cv2.rectangle(
                vis,
                (min(x1, x2), min(y1, y2)),
                (max(x1, x2), max(y1, y2)),
                (255, 0, 0),
                2,
            )

        cv2.imshow(win, vis)

        if cv2.waitKey(50) & 0xFF == 27:
            cv2.destroyAllWindows()
            print("Отменено.")
            return None

    x1, y1 = points[0]
    x2, y2 = points[1]

    grid_x = min(x1, x2)
    grid_y = min(y1, y2)
    w_sel = abs(x2 - x1)
    h_sel = abs(y2 - y1)

    cell_size = max(20, int(round(((w_sel / GRID) + (h_sel / GRID)) / 2.0)))
    history = []

    print(f"      Стартовая рамка: ({grid_x},{grid_y}) cell={cell_size}px")

    # ---------- ФАЗА 2: подгонка сетки ----------
    print("")
    print("[2/2] Подгоните зелёную сетку ТОЧНО по границам клеток:")
    print("      W/S = вверх/вниз, A/D = влево/вправо, Q/E = клетка -/+")
    print("      Z = отменить, R = обновить скриншот, Enter = сохранить, ESC = выход")

    hint = "W/A/S/D move | Q/E cell | Z undo | R refresh | Enter save | ESC quit"

    while True:
        vis = full.copy()
        _draw_grid(vis, grid_x, grid_y, cell_size, hint)
        cv2.imshow(win, vis)

        key = cv2.waitKey(0) & 0xFF

        if key == 13:  # Enter
            break
        elif key == 27:  # Esc
            cv2.destroyAllWindows()
            print("Выход без сохранения.")
            return None
        elif key in (ord("r"), ord("R")):
            full = capture_screen()
        elif key in (ord("z"), ord("Z")):
            if history:
                grid_x, grid_y, cell_size = history.pop()
        elif key in (ord("w"), ord("W")):
            history.append((grid_x, grid_y, cell_size))
            grid_y -= 1
        elif key in (ord("s"), ord("S")):
            history.append((grid_x, grid_y, cell_size))
            grid_y += 1
        elif key in (ord("a"), ord("A")):
            history.append((grid_x, grid_y, cell_size))
            grid_x -= 1
        elif key in (ord("d"), ord("D")):
            history.append((grid_x, grid_y, cell_size))
            grid_x += 1
        elif key in (ord("q"), ord("Q")):
            history.append((grid_x, grid_y, cell_size))
            cell_size = max(20, cell_size - 1)
        elif key in (ord("e"), ord("E")):
            history.append((grid_x, grid_y, cell_size))
            cell_size += 1

    cv2.destroyAllWindows()

    width = cell_size * GRID
    height = cell_size * GRID

    # ---------- сохранение ----------
    y1c = max(0, grid_y)
    y2c = min(full.shape[0], grid_y + height)
    x1c = max(0, grid_x)
    x2c = min(full.shape[1], grid_x + width)

    crop = full[y1c:y2c, x1c:x2c]
    if crop.size > 0:
        cv2.imwrite(BOARD_CROP_FILE, crop)
        print(
            f"  Сохранён квадрат доски: {BOARD_CROP_FILE} "
            f"({crop.shape[1]}x{crop.shape[0]})"
        )

    cv2.imwrite(CALIBRATION_SCREENSHOT, full)

    data = {
        "board_left": int(grid_x),
        "board_top": int(grid_y),
        "board_width": int(width),
        "board_height": int(height),
        "cell_size": int(cell_size),

        # Совместимость со старыми скриптами.
        "board_tl": [float(grid_x), float(grid_y)],
        "board_tr": [float(grid_x + width), float(grid_y)],
        "board_bl": [float(grid_x), float(grid_y + height)],
        "board_br": [float(grid_x + width), float(grid_y + height)],
        "cell_w": float(width) / GRID,
        "cell_h": float(height) / GRID,
        "screenshot_shape": list(full.shape),
    }

    data = normalize_calibration(data)

    if save:
        save_calibration(data, CALIBRATION_FILE)
        print(
            f"✅ Калибровка сохранена. Клетка = {cell_size}px, "
            f"доска = {width}x{height} @ ({grid_x},{grid_y})"
        )

    return data


# Совместимость со старыми вызовами.
calibrate_manual = calibrate

if __name__ == "__main__":
    import sys
    # --no-pause: пропустить ожидание Enter (для запуска из GUI, где stdin пуст)
    if "--no-pause" in sys.argv:
        print("📸 Снимаю экран (не убирайте окно с доской)...")
        data = calibrate(capture_screen())
        if data is None:
            print("Отменено.")
    else:
        calibrate()
