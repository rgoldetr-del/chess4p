"""
create_promoted_gray_templates.py

Скрипт для создания:
1. Цветных превращённых ферзей из 4 центральных клеток:
   - 7,7  red
   - 8,7  blue
   - 7,8  yellow
   - 8,8  green

2. Серых шаблонов 7 типов:
   - gray promoted queen
   - king
   - queen
   - bishop
   - knight
   - rook
   - pawn

Цветные превращённые ферзи сохраняются в:
    templates/promoted_queens/

Серые фигуры сохраняются в:
    templates/gray/

Важно:
    board_detector.py читает серые шаблоны именно из templates/gray,
    имя файла должно начинаться с имени фигуры:
        pawn_*.png
        knight_*.png
        bishop_*.png
        rook_*.png
        queen_*.png
        king_*.png

Использование:
    cd bot
    python create_promoted_gray_templates.py

Опции:
    python create_promoted_gray_templates.py --clean
    python create_promoted_gray_templates.py --auto
    python create_promoted_gray_templates.py --skip-promoted
    python create_promoted_gray_templates.py --skip-gray
"""

import os
import cv2
import argparse

from config import TEMPLATES_DIR
from calibration import (
    load_calibration,
    square_to_region,
    pixel_to_square,
)
from screen_capture import capture_screen

try:
    from coordinates import is_valid_square
except ImportError:
    def is_valid_square(x: int, y: int) -> bool:
        if x < 1 or x > 14 or y < 1 or y > 14:
            return False

        in_x = x <= 3 or x >= 12
        in_y = y <= 3 or y >= 12

        return not (in_x and in_y)


# ============================================================
# Directories
# ============================================================

PROMOTED_DIR = os.path.join(TEMPLATES_DIR, "promoted_queens")
GRAY_DIR = os.path.join(TEMPLATES_DIR, "gray")


# ============================================================
# Central promoted queens
# ============================================================

# Экранные координаты 4 центральных клеток.
# y=1 сверху.
CENTRAL_PROMOTED = [
    ("red",    7, 7),  # слева сверху
    ("blue",   8, 7),  # справа сверху
    ("yellow", 7, 8),  # слева снизу
    ("green",  8, 8),  # справа снизу
]


# ============================================================
# Gray targets
# ============================================================

GRAY_TARGETS = [
    ("gray_promoted_queen", "GREEN PROMOTED QUEEN (gray)"),
    ("king",                "GRAY GREEN KING"),
    ("queen",               "GRAY GREEN QUEEN"),
    ("bishop",              "GRAY GREEN BISHOP"),
    ("knight",              "GRAY GREEN KNIGHT"),
    ("rook",                "GRAY GREEN ROOK"),
    ("pawn",                "GRAY GREEN PAWN near rook"),
]


# ============================================================
# Auto cells for --auto
# ============================================================

# Если знаете точные клетки, где лежат серые фигуры, заполните их здесь.
# Координаты экранные: x=1 слева, y=1 сверху.
#
# Например, если серая зелёная армия стоит на исходных клетках:
#   king   = 14,7
#   queen  = 14,8
#   bishop = 14,6
#   knight = 14,5
#   rook   = 14,4
#   pawn   = 13,4
#
# Для gray_promoted_queen обычно нужно указать вручную или кликом.

GRAY_AUTO_CELLS = {
    "gray_promoted_queen": None,      # например: (9, 7)
    "king":                (14, 7),
    "queen":               (14, 8),
    "bishop":              (14, 6),
    "knight":              (14, 5),
    "rook":                (14, 4),
    "pawn":                (13, 4),
}


# ============================================================
# Helpers
# ============================================================

def ensure_dirs():
    os.makedirs(PROMOTED_DIR, exist_ok=True)
    os.makedirs(GRAY_DIR, exist_ok=True)


def clean_dir(path):
    if not os.path.isdir(path):
        return 0

    count = 0

    for fname in os.listdir(path):
        if fname.lower().endswith(".png"):
            try:
                os.remove(os.path.join(path, fname))
                count += 1
            except Exception as e:
                print(f"⚠ Не удалось удалить {fname}: {e}")

    return count


def point_in_board(px: int, py: int, cal: dict) -> bool:
    try:
        left = int(cal.get("board_left", cal["board_tl"][0]))
        top = int(cal.get("board_top", cal["board_tl"][1]))
        width = int(cal.get("board_width", 0))
        height = int(cal.get("board_height", 0))
    except Exception:
        return False

    return (
        left <= px < left + width
        and top <= py < top + height
    )


def save_cell(img, x: int, y: int, cal: dict, out_path: str) -> bool:
    """Вырезает клетку (x,y) и сохраняет как PNG."""
    left, top, w, h = square_to_region(x, y, cal)

    if w <= 0 or h <= 0:
        print(f"⚠ Пустой регион для клетки ({x},{y})")
        return False

    cell = img[top:top + h, left:left + w]

    if cell.size == 0:
        print(f"⚠ Пустая вырезка для клетки ({x},{y})")
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cell)

    print(f"✅ Saved: {out_path}")
    return True


# ============================================================
# Phase 1: promoted queens in central cells
# ============================================================

def cut_central_promoted_queens(cal: dict):
    print("")
    print("=" * 70)
    print("Фаза 1: вырезаю 4 центральных превращённых ферзя")
    print("=" * 70)

    img = capture_screen()

    if img is None or img.size == 0:
        print("❌ Не удалось захватить экран")
        return False

    saved = 0

    for color, x, y in CENTRAL_PROMOTED:
        fname = f"{color}_promoted_queen_{x}_{y}.png"
        out_path = os.path.join(PROMOTED_DIR, fname)

        if save_cell(img, x, y, cal, out_path):
            saved += 1

    print(f"Итого сохранено promoted-шаблонов: {saved}/4")
    return saved > 0


# ============================================================
# Phase 2: interactive gray selection
# ============================================================

def interactive_select_gray(img, cal: dict):
    """
    Пользователь кликает по 7 клеткам в заданном порядке.
    Возвращает dict:
        target_key -> (x, y)
    """
    win = "Select gray templates"

    state = {
        "idx": 0,
        "click": None,
        "results": {},
    }

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (int(x), int(y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while state["idx"] < len(GRAY_TARGETS):
        vis = img.copy()

        # Уже выбранные клетки.
        for i, (key, sq) in enumerate(state["results"].items()):
            x, y = sq
            left, top, w, h = square_to_region(x, y, cal)

            cv2.rectangle(
                vis,
                (left, top),
                (left + w, top + h),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                vis,
                f"{i + 1}:{key}",
                (left + 3, top + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        key, label = GRAY_TARGETS[state["idx"]]

        cv2.rectangle(vis, (5, 5), (1400, 80), (0, 0, 0), -1)

        cv2.putText(
            vis,
            f"{state['idx'] + 1}/{len(GRAY_TARGETS)}  CLICK: {label}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            "U = undo last | ESC = cancel",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

        cv2.imshow(win, vis)

        k = cv2.waitKey(20) & 0xFF

        if k == 27:  # ESC
            cv2.destroyAllWindows()
            print("Отменено.")
            return None

        if k in (ord("u"), ord("U")):
            if state["results"]:
                # Удаляем последний выбранный ключ.
                last_key = list(state["results"].keys())[-1]
                del state["results"][last_key]
                state["idx"] -= 1

        if state["click"] is not None:
            px, py = state["click"]
            state["click"] = None

            if not point_in_board(px, py, cal):
                print(f"⚠ Клик вне доски: ({px},{py}) — игнорирую")
                continue

            sx, sy = pixel_to_square(px, py, cal)

            if not is_valid_square(sx, sy):
                print(
                    f"⚠ Клик в вырезанный угол: ({sx},{sy}) — игнорирую"
                )
                continue

            state["results"][key] = (sx, sy)
            state["idx"] += 1

            print(
                f"✅ {key}: выбран квадрат ({sx},{sy})"
            )

    cv2.destroyAllWindows()
    return state["results"]


# ============================================================
# Phase 2: auto gray selection
# ============================================================

def auto_select_gray():
    print("")
    print("Использую GRAY_AUTO_CELLS для серых шаблонов.")

    results = {}

    for key, _label in GRAY_TARGETS:
        sq = GRAY_AUTO_CELLS.get(key, None)

        if sq is None:
            print(
                f"❌ GRAY_AUTO_CELLS['{key}'] не задан.\n"
                f"   Заполните координаты или запустите без --auto."
            )
            return None

        x, y = sq

        if not is_valid_square(x, y):
            print(f"❌ GRAY_AUTO_CELLS['{key}'] = ({x},{y}) вне доски")
            return None

        results[key] = (int(x), int(y))
        print(f"✅ {key}: ({x},{y})")

    return results


# ============================================================
# Phase 2: save gray templates
# ============================================================

def save_gray_templates(img, cal: dict, selections: dict):
    print("")
    print("=" * 70)
    print("Фаза 2: сохраняю серые шаблоны")
    print("=" * 70)

    saved = 0

    for key, (x, y) in selections.items():
        if key == "gray_promoted_queen":
            # Важно: файл в gray должен начинаться с queen,
            # чтобы board_detector загрузил его как серого ферзя.
            gray_path = os.path.join(
                GRAY_DIR,
                f"queen_gray_promoted_green_{x}_{y}.png",
            )

            # Дополнительно сохраняем в promoted_queens.
            promoted_path = os.path.join(
                PROMOTED_DIR,
                f"gray_green_promoted_queen_{x}_{y}.png",
            )

            ok1 = save_cell(img, x, y, cal, gray_path)
            ok2 = save_cell(img, x, y, cal, promoted_path)

            if ok1:
                saved += 1

        else:
            piece = key

            gray_path = os.path.join(
                GRAY_DIR,
                f"{piece}_gray_green_{x}_{y}.png",
            )

            if save_cell(img, x, y, cal, gray_path):
                saved += 1

    print(f"Итого сохранено серых шаблонов: {saved}/7")
    return saved > 0


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create promoted and gray templates"
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Удалить старые PNG в templates/promoted_queens и templates/gray",
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="Использовать GRAY_AUTO_CELLS вместо кликов",
    )

    parser.add_argument(
        "--skip-promoted",
        action="store_true",
        help="Не вырезать 4 центральных promoted ферзя",
    )

    parser.add_argument(
        "--skip-gray",
        action="store_true",
        help="Не делать серые шаблоны",
    )

    args = parser.parse_args()

    cal = load_calibration()
    ensure_dirs()

    if args.clean:
        n1 = clean_dir(PROMOTED_DIR)
        n2 = clean_dir(GRAY_DIR)
        print(f"Удалено PNG: promoted_queens={n1}, gray={n2}")

    # ------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------
    if not args.skip_promoted:
        cut_central_promoted_queens(cal)
    else:
        print("Фаза 1 пропущена (--skip-promoted)")

    if args.skip_gray:
        print("Фаза 2 пропущена (--skip-gray)")
        return

    # ------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------
    print("")
    input(
        "Теперь подготовьте серые фигуры и нажмите Enter "
        "для второго скриншота..."
    )

    img = capture_screen()

    if img is None or img.size == 0:
        print("❌ Не удалось захватить экран")
        return

    if args.auto:
        selections = auto_select_gray()
    else:
        print("")
        print(
            "Сейчас откроется окно. "
            "Кликните по нужным клеткам в указанном порядке."
        )
        selections = interactive_select_gray(img, cal)

    if selections is None:
        print("❌ Ничего не сохранено.")
        return

    save_gray_templates(img, cal, selections)

    print("")
    print("Готово.")
    print(
        "Серые шаблоны лежат в: "
        f"{GRAY_DIR}"
    )
    print(
        "Превращённые ферзи лежат в: "
        f"{PROMOTED_DIR}"
    )


if __name__ == "__main__":
    main()