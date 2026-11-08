"""check_calibration.py — проверка калибровки и распознавания доски.

Запуск:
    cd bot
    python tools/check_calibration.py

Результат:
- печатает число найденных фигур;
- сохраняет debug-картинку в bot/debug/check_calibration.png.
"""

import os
import sys

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

import cv2

from config import DEBUG_DIR, COLOR_ORDER
from calibration import load_calibration, square_to_region
from screen_capture import capture_screen
from board_detector import BoardDetector, PIECE_NAMES
from coordinates import is_valid_square


COLOR_BGR = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
}

PIECE_LABEL = {
    "pawn": "P",
    "knight": "N",
    "bishop": "B",
    "rook": "R",
    "queen": "Q",
    "king": "K",
}

def main():
    cal = load_calibration()

    print("Захват экрана...")
    shot = capture_screen()

    print("Создание BoardDetector...")
    detector = BoardDetector(cal)

    print("Детекция доски...")
    board = detector.detect_full_board(shot)

    total = 0
    dead_count = 0
    promoted_count = 0
    counts = {c: 0 for c in COLOR_ORDER}

    for color in COLOR_ORDER:
        if counts[color] > 16:
            print(
                f"⚠ НЕВОЗМОЖНО: {color} = {counts[color]} фигур (>16). "
                f"Ошибка детекции."
            )

    for y in range(1, 15):
        for x in range(1, 15):
            cell = board[y][x]

            if cell is None:
                continue

            color_idx, piece_idx, promoted, dead = cell

            total += 1

            if dead:
                dead_count += 1

            if promoted:
                promoted_count += 1

            if color_idx is not None and color_idx >= 0:
                counts[COLOR_ORDER[color_idx]] += 1

    print("")
    print("=" * 60)
    print("Результат детекции")
    print("=" * 60)
    print(f"Всего фигур:       {total}")
    print(f"Серых/dead:        {dead_count}")
    print(f"Promoted:          {promoted_count}")
    print("")

    for color in COLOR_ORDER:
        print(f"{color:6s}: {counts[color]}")

    if total == 0:
        print("")
        print("⚠ Ни одной фигуры не найдено.")
        print("  Возможные причины:")
        print("  - калибровка уехала;")
        print("  - templates не созданы;")
        print("  - размер клетки в шаблонах не совпадает с экраном;")
        print("  - доска не в начальном/игровом состоянии.")

    # ------------------------------------------------------------
    # Debug image
    # ------------------------------------------------------------
    img = shot.copy()

    for y in range(1, 15):
        for x in range(1, 15):
            if not is_valid_square(x, y):
                continue

            left, top, w, h = square_to_region(x, y, cal)
            cell = board[y][x]

            if cell is None:
                cv2.rectangle(
                    img,
                    (left, top),
                    (left + w, top + h),
                    (80, 80, 80),
                    1,
                )
                continue

            color_idx, piece_idx, promoted, dead = cell

            if dead:
                box_color = (128, 128, 128)
                text_color = (255, 255, 255)
                label = "D:" + PIECE_LABEL[PIECE_NAMES[piece_idx]]
            else:
                color_name = COLOR_ORDER[color_idx]
                box_color = COLOR_BGR[color_name]
                text_color = (255, 255, 255)
                label = PIECE_LABEL[PIECE_NAMES[piece_idx]]

                if promoted:
                    label += "P"

            cv2.rectangle(
                img,
                (left, top),
                (left + w, top + h),
                box_color,
                2,
            )

            cv2.putText(
                img,
                label,
                (left + 3, top + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
            )

    os.makedirs(DEBUG_DIR, exist_ok=True)

    out_path = os.path.join(DEBUG_DIR, "check_calibration.png")
    cv2.imwrite(out_path, img)

    print("")
    print(f"✅ Debug-картинка сохранена: {out_path}")


if __name__ == "__main__":
    main()