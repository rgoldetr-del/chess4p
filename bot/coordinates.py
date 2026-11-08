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