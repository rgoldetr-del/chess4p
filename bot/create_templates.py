"""create_templates.py — извлекает шаблоны фигур из скриншота начальной позиции.

Раскладка:
    red    = низ
    blue   = лево
    yellow = верх
    green  = право

Координаты:
    x=1 лево экрана
    y=1 верх экрана

Использование:
    python create_templates.py
    python create_templates.py --screenshot calibration_screenshot.png
    python create_templates.py --gray
"""

import cv2
import os
import argparse

from config import (
    TEMPLATES_DIR,
    COLOR_ORDER,
    CALIBRATION_SCREENSHOT,
)

from calibration import (
    load_calibration,
    square_to_region,
)


# ============================================================
# Initial layout
# ============================================================

def initial_layout():
    """Возвращает dict {(x,y): (color, piece)}."""
    initial = {
        # RED — снизу: back rank y=14, пешки y=13.
        (4, 14): ("red", "rook"),
        (5, 14): ("red", "knight"),
        (6, 14): ("red", "bishop"),
        (7, 14): ("red", "queen"),
        (8, 14): ("red", "king"),
        (9, 14): ("red", "bishop"),
        (10, 14): ("red", "knight"),
        (11, 14): ("red", "rook"),
        **{(x, 13): ("red", "pawn") for x in range(4, 12)},

        # BLUE — слева: back rank x=1, пешки x=2.
        (1, 4): ("blue", "rook"),
        (1, 5): ("blue", "knight"),
        (1, 6): ("blue", "bishop"),
        (1, 7): ("blue", "queen"),
        (1, 8): ("blue", "king"),
        (1, 9): ("blue", "bishop"),
        (1, 10): ("blue", "knight"),
        (1, 11): ("blue", "rook"),
        **{(2, y): ("blue", "pawn") for y in range(4, 12)},

        # YELLOW — сверху: back rank y=1, пешки y=2.
        (4, 1): ("yellow", "rook"),
        (5, 1): ("yellow", "knight"),
        (6, 1): ("yellow", "bishop"),
        (7, 1): ("yellow", "king"),
        (8, 1): ("yellow", "queen"),
        (9, 1): ("yellow", "bishop"),
        (10, 1): ("yellow", "knight"),
        (11, 1): ("yellow", "rook"),
        **{(x, 2): ("yellow", "pawn") for x in range(4, 12)},

        # GREEN — справа: back rank x=14, пешки x=13.
        (14, 4): ("green", "rook"),
        (14, 5): ("green", "knight"),
        (14, 6): ("green", "bishop"),
        (14, 7): ("green", "king"),
        (14, 8): ("green", "queen"),
        (14, 9): ("green", "bishop"),
        (14, 10): ("green", "knight"),
        (14, 11): ("green", "rook"),
        **{(13, y): ("green", "pawn") for y in range(4, 12)},
    }

    return initial


# ============================================================
# Extraction
# ============================================================

def extract_templates(screenshot_path=None, output_dir=None):
    """Открывает скриншот начальной позиции и вырезает каждую фигуру."""
    screenshot_path = screenshot_path or CALIBRATION_SCREENSHOT
    output_dir = output_dir or TEMPLATES_DIR

    cal = load_calibration()

    img = cv2.imread(screenshot_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть {screenshot_path}")

    initial = initial_layout()

    # Чистим старые PNG шаблоны.
    if os.path.isdir(output_dir):
        for root, _, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".png"):
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass

    saved = 0

    for (x, y), (color, piece) in initial.items():
        left, top, w, h = square_to_region(x, y, cal)
        cell = img[top:top + h, left:left + w]

        if cell.size == 0:
            print(f"  ⚠ Пустая вырезка для ({x},{y}) — пропуск")
            continue

        out_dir = os.path.join(output_dir, color)
        os.makedirs(out_dir, exist_ok=True)

        fname = os.path.join(out_dir, f"{piece}_{x}_{y}.png")
        cv2.imwrite(fname, cell)
        saved += 1

        print(f"  Saved: {fname}")

    print(f"✅ Шаблоны извлечены: {saved} фигур в '{output_dir}/'.")
    return saved


# ============================================================
# Optional approximate gray templates
# ============================================================

def make_gray_templates(output_dir=None):
    """
    Создаёт approximate gray-шаблоны из цветных.

    Важно:
        Это приближение. Для надёжного определения dead-фигур лучше
        использовать реальные скриншоты серых фигур.
    """
    output_dir = output_dir or TEMPLATES_DIR
    gray_dir = os.path.join(output_dir, "gray")
    os.makedirs(gray_dir, exist_ok=True)

    count = 0

    for color in COLOR_ORDER:
        cdir = os.path.join(output_dir, color)
        if not os.path.isdir(cdir):
            continue

        for fname in os.listdir(cdir):
            if not fname.endswith(".png"):
                continue

            src = os.path.join(cdir, fname)
            img = cv2.imread(src)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dst = os.path.join(gray_dir, f"{color}_{fname}")
            cv2.imwrite(dst, gray)
            count += 1

    print(f"✅ Создано approximate gray-шаблонов: {count}")
    return count


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create piece templates from initial position screenshot"
    )

    parser.add_argument(
        "--screenshot",
        type=str,
        default=CALIBRATION_SCREENSHOT,
        help="Path to initial position screenshot",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=TEMPLATES_DIR,
        help="Output templates directory",
    )

    parser.add_argument(
        "--gray",
        action="store_true",
        help="Also create approximate gray templates from color templates",
    )

    args = parser.parse_args()

    extract_templates(args.screenshot, args.output)

    if args.gray:
        make_gray_templates(args.output)


if __name__ == "__main__":
    main()