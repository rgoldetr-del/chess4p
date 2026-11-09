"""coordinates.py — единые преобразования координат и индексов ходов.

Используется везде вместо дублирования функций в main/calibration.
"""

BOARD_SIZE = 14
ACTION_SPACE = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE  # 38416


def game_to_screen(gx: int, gy: int):
    """Внутренние координаты движка -> экранные. Ось Y инвертирована."""
    return gx, 15 - gy

def screen_to_game(x: int, y: int):
    """Экранные координаты -> внутренние. Ось Y инвертирована."""
    return x, 15 - y


def coords_to_idx(fx: int, fy: int, tx: int, ty: int) -> int:
    """(from_x, from_y, to_x, to_y) -> action index."""
    return (
        (fx - 1)
        + (fy - 1) * 14
        + (tx - 1) * 196
        + (ty - 1) * 2744
    )


def idx_to_coords(idx: int):
    """action index -> (from_x, from_y, to_x, to_y)."""
    fx = (idx % 14) + 1
    fy = ((idx // 14) % 14) + 1
    tx = ((idx // 196) % 14) + 1
    ty = (idx // 2744) + 1
    return fx, fy, tx, ty


def is_valid_square(x: int, y: int) -> bool:
    """Проверка игровой клетки 4PC с учётом вырезанных углов 3x3."""
    if x < 1 or x > 14 or y < 1 or y > 14:
        return False

    in_x = x <= 3 or x >= 12
    in_y = y <= 3 or y >= 12

    return not (in_x and in_y)

# ============================================================
# Ориентация доски: chess.com рисует снизу ЦВЕТ ПОЛЬЗОВАТЕЛЯ.
# Весь бот работает в координатах движка (красный снизу), поэтому при
# повёрнутой доске применяется поворот сетки и перекраска цветов.
# ============================================================
ORIENTATION = "red"  # какой абсолютный цвет находится СНИЗУ экрана

def set_orientation(bottom_color):
    """bottom_color — абсолютный цвет ('red'/'blue'/'yellow'/'green'),
    который сейчас находится у нижнего края экрана."""
    global ORIENTATION
    if bottom_color in ("red", "blue", "yellow", "green"):
        ORIENTATION = bottom_color

def get_orientation():
    return ORIENTATION

# Поворот координат: движок (x, y) -> экран (sx, sy)
# Ось Y экрана идёт СВЕРХУ ВНИЗ, ось Y движка — СНИЗУ ВВЕРХ (красный дом y=1).
# Поэтому даже без поворота есть инверсия: экран y=14 <-> движок y=1.
_ROT_E2S = {
    "red":    lambda x, y: (x, 15 - y),
    "yellow": lambda x, y: (15 - x, y),
    "blue":   lambda x, y: (15 - y, 15 - x),
    "green":  lambda x, y: (y, x),
}
# Инверсия: экран -> движок
_ROT_S2E = {
    "red":    lambda sx, sy: (sx, 15 - sy),
    "yellow": lambda sx, sy: (15 - sx, sy),
    "blue":   lambda sx, sy: (15 - sy, 15 - sx),
    "green":  lambda sx, sy: (sy, sx),
}

def engine_pos_to_screen(x, y):
    return _ROT_E2S[ORIENTATION](x, y)

def screen_pos_to_engine(sx, sy):
    return _ROT_S2E[ORIENTATION](sx, sy)

# Перекраска: абсолютный цвет на экране -> цвет движка
# (сидения: красный низ, синий лево, жёлтый верх, зелёный право)
_COLOR_MAP_S2E = {
    "red":    {"red": "red", "blue": "blue", "yellow": "yellow", "green": "green"},
    "yellow": {"yellow": "red", "green": "blue", "red": "yellow", "blue": "green"},
    "blue":   {"blue": "red", "yellow": "blue", "green": "yellow", "red": "green"},
    "green":  {"green": "red", "red": "blue", "blue": "yellow", "yellow": "green"},
}

def screen_color_to_engine(name):
    return _COLOR_MAP_S2E.get(ORIENTATION, {}).get(name, name)

def engine_color_to_screen(name):
    m = _COLOR_MAP_S2E.get(ORIENTATION, {})
    inv = {v: k for k, v in m.items()}
    return inv.get(name, name)


def screen_color_idx_to_engine(ci):
    """Индекс абсолютного цвета экрана -> индекс цвета движка."""
    from config import COLOR_ORDER
    name = COLOR_ORDER[max(0, min(3, int(ci)))]
    eng = screen_color_to_engine(name)
    return COLOR_ORDER.index(eng)

def engine_color_idx_to_screen(ci):
    """Индекс цвета движка -> индекс абсолютного цвета экрана."""
    from config import COLOR_ORDER
    name = COLOR_ORDER[max(0, min(3, int(ci)))]
    scr = engine_color_to_screen(name)
    return COLOR_ORDER.index(scr)
