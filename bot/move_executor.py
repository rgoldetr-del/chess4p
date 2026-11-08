"""move_executor.py — выполнение ходов мышью.

Логика оставлена близко к текущей:
- click;
- drag;
- auto: click чаще, чем drag;
- human-like задержки.

Добавлены только защиты:
- проверка pyautogui;
- ограничение слишком большого jitter.
"""

import time
import random
import math

try:
    import pyautogui

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.005

except ImportError:
    pyautogui = None


from config import (
    MOUSE_MOVE_DURATION,
    CLICK_PAUSE,
    THINK_TIME_MIN,
    THINK_TIME_MAX,
)

from calibration import square_to_pixel


# ============================================================
# Bezier / mouse
# ============================================================

def bezier_curve(p0, p1, p2, p3, n_points=30):
    """Кубическая кривая Безье."""
    points = []

    for i in range(n_points + 1):
        t = i / n_points
        u = 1 - t

        x = (
            u ** 3 * p0[0]
            + 3 * u ** 2 * t * p1[0]
            + 3 * u * t ** 2 * p2[0]
            + t ** 3 * p3[0]
        )

        y = (
            u ** 3 * p0[1]
            + 3 * u ** 2 * t * p1[1]
            + 3 * u * t ** 2 * p2[1]
            + t ** 3 * p3[1]
        )

        points.append((int(x), int(y)))

    return points


def human_mouse_move(target_x, target_y):
    """Перемещает мышь к цели по кривой Безье."""
    if pyautogui is None:
        return

    start_x, start_y = pyautogui.position()

    dx = target_x - start_x
    dy = target_y - start_y

    dist = math.sqrt(dx * dx + dy * dy)

    offset = max(5, dist * 0.06)

    cp1 = (
        start_x + dx * 0.3 + random.gauss(0, offset),
        start_y + dy * 0.3 + random.gauss(0, offset),
    )

    cp2 = (
        start_x + dx * 0.7 + random.gauss(0, offset),
        start_y + dy * 0.7 + random.gauss(0, offset),
    )

    points = bezier_curve(
        (start_x, start_y),
        cp1,
        cp2,
        (target_x, target_y),
        n_points=25,
    )

    duration = random.uniform(*MOUSE_MOVE_DURATION)
    delay = duration / max(1, len(points))

    for px, py in points:
        pyautogui.moveTo(px, py, _pause=False)
        time.sleep(max(0.001, delay + random.gauss(0, delay * 0.1)))


def human_click(x, y):
    """Клик с минимальной задержкой."""
    if pyautogui is None:
        return

    human_mouse_move(x, y)
    time.sleep(random.uniform(*CLICK_PAUSE))

    pyautogui.click(x, y)

    time.sleep(random.uniform(*CLICK_PAUSE))


# ============================================================
# Move execution
# ============================================================

def _clamp_jitter(offset, limit):
    return max(-limit, min(limit, offset))


def _apply_jitter(x, y, cal, jitter_ratio=0.06, max_ratio=0.20):
    cell = int(cal.get("cell_size", 60))

    jitter = cell * jitter_ratio
    limit = cell * max_ratio

    x += int(_clamp_jitter(random.gauss(0, jitter), limit))
    y += int(_clamp_jitter(random.gauss(0, jitter), limit))

    return x, y


def execute_move_click_click(from_sq, to_sq, cal):
    """Ход: клик на фигуру → клик на целевую клетку."""
    fx, fy = square_to_pixel(from_sq[0], from_sq[1], cal)
    tx, ty = square_to_pixel(to_sq[0], to_sq[1], cal)

    fx, fy = _apply_jitter(fx, fy, cal)
    tx, ty = _apply_jitter(tx, ty, cal)

    human_click(fx, fy)
    time.sleep(random.uniform(0.05, 0.12))
    human_click(tx, ty)


def execute_move_drag(from_sq, to_sq, cal):
    """Ход перетаскиванием."""
    if pyautogui is None:
        return

    fx, fy = square_to_pixel(from_sq[0], from_sq[1], cal)
    tx, ty = square_to_pixel(to_sq[0], to_sq[1], cal)

    fx, fy = _apply_jitter(fx, fy, cal, jitter_ratio=0.05, max_ratio=0.18)
    tx, ty = _apply_jitter(tx, ty, cal, jitter_ratio=0.05, max_ratio=0.18)

    human_mouse_move(fx, fy)
    time.sleep(random.uniform(0.03, 0.08))

    pyautogui.mouseDown()
    time.sleep(random.uniform(0.03, 0.06))

    human_mouse_move(tx, ty)
    time.sleep(random.uniform(0.03, 0.06))

    pyautogui.mouseUp()


def execute_move(from_sq, to_sq, cal, method="click"):
    """Выполняет ход. method = 'click', 'drag' или 'auto'."""
    if pyautogui is None:
        raise RuntimeError(
            "pyautogui не установлен. "
            "Установите: pip install pyautogui"
        )

    if method == "auto":
        method = random.choice(["click", "click", "click", "drag"])

    if method == "click":
        execute_move_click_click(from_sq, to_sq, cal)
    else:
        execute_move_drag(from_sq, to_sq, cal)


# ============================================================
# Human-like think delay
# ============================================================

def human_think_delay():
    """Имитация раздумья."""
    delay = random.uniform(THINK_TIME_MIN, THINK_TIME_MAX)

    if random.random() < 0.05:
        delay *= 1.5

    time.sleep(delay)