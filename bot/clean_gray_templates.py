"""
clean_gray_templates.py

Удаляет старые серые шаблоны и оставляет новые 7:
- queen_gray_promoted_green_*.png
- king_gray_green_*.png
- queen_gray_green_*.png
- bishop_gray_green_*.png
- knight_gray_green_*.png
- rook_gray_green_*.png
- pawn_gray_green_*.png

По умолчанию старые файлы НЕ удаляются навсегда,
а переносятся в:
    templates/gray_old_backup/

Чтобы удалить навсегда:
    python clean_gray_templates.py --delete

Сначала посмотреть, что будет сделано:
    python clean_gray_templates.py --dry-run
"""

import os
import cv2
import shutil
import fnmatch
import argparse

from config import TEMPLATES_DIR
from calibration import load_calibration


GRAY_DIR = os.path.join(TEMPLATES_DIR, "gray")
BACKUP_DIR = os.path.join(TEMPLATES_DIR, "gray_old_backup")

PIECE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]

# Новые шаблоны, которые нужно оставить.
# Если у вас имена отличаются — поправьте этот список.
DEFAULT_KEEP_PATTERNS = [
    "queen_gray_promoted_green_*.png",
    "king_gray_green_*.png",
    "queen_gray_green_*.png",
    "bishop_gray_green_*.png",
    "knight_gray_green_*.png",
    "rook_gray_green_*.png",
    "pawn_gray_green_*.png",
]


def keep_file(fname: str, patterns) -> bool:
    fname = fname.lower()

    for pat in patterns:
        if fnmatch.fnmatch(fname, pat.lower()):
            return True

    return False


def unique_dst(dst: str) -> str:
    if not os.path.exists(dst):
        return dst

    base, ext = os.path.splitext(dst)
    i = 1

    while True:
        cand = f"{base}_{i}{ext}"

        if not os.path.exists(cand):
            return cand

        i += 1


def ensure_piece_prefix(path: str) -> str:
    """
    board_detector требует, чтобы имя файла начиналось с имени фигуры:
        pawn_*.png
        knight_*.png
        bishop_*.png
        rook_*.png
        queen_*.png
        king_*.png

    Если имя не начинается, но содержит имя фигуры, переименовываем.
    """
    directory = os.path.dirname(path)
    fname = os.path.basename(path)
    lower = fname.lower()

    for piece in PIECE_NAMES:
        if lower.startswith(piece):
            return path

    for piece in PIECE_NAMES:
        if piece in lower:
            new_name = f"{piece}_{fname}"
            new_path = os.path.join(directory, new_name)
            new_path = unique_dst(new_path)

            os.rename(path, new_path)
            print(f"  [rename] {fname} -> {os.path.basename(new_path)}")

            return new_path

    print(f"  ⚠ Не удалось определить тип фигуры: {fname}")
    return path


def resize_template(path: str, cell):
    img = cv2.imread(path)

    if img is None:
        print(f"  ⚠ Не удалось прочитать: {path}")
        return False

    if img.shape[:2][::-1] != cell:
        img = cv2.resize(
            img,
            cell,
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(path, img)
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Clean and adapt gray templates"
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Удалить старые серые шаблоны навсегда, а не переносить в backup",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать, что будет сделано",
    )

    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Не ресайзить оставшиеся шаблоны под текущую калибровку",
    )

    parser.add_argument(
        "--purge-backup",
        action="store_true",
        help="Удалить папку templates/gray_old_backup",
    )

    args = parser.parse_args()

    if args.purge_backup:
        if os.path.isdir(BACKUP_DIR):
            shutil.rmtree(BACKUP_DIR)
            print(f"✅ Удалён backup: {BACKUP_DIR}")
        else:
            print("Backup папка не найдена.")

        if not args.delete and not args.dry_run:
            return

    if not os.path.isdir(GRAY_DIR):
        print(f"Папка не найдена: {GRAY_DIR}")
        return

    kept = []
    old = []

    for fname in sorted(os.listdir(GRAY_DIR)):
        if not fname.lower().endswith(".png"):
            continue

        if keep_file(fname, DEFAULT_KEEP_PATTERNS):
            kept.append(fname)
        else:
            old.append(fname)

    print("=" * 70)
    print(f"Папка gray: {GRAY_DIR}")
    print("=" * 70)
    print(f"Новых/нужных шаблонов: {len(kept)}")
    print(f"Старых/лишних шаблонов: {len(old)}")

    if kept:
        print("")
        print("Оставить:")
        for f in kept:
            print(f"  [KEEP] {f}")

    if old:
        print("")
        print("Удалить/перенести:")
        for f in old:
            print(f"  [OLD]  {f}")

    if args.dry_run:
        print("")
        print("Dry-run: ничего не изменено.")
        return

    # ------------------------------------------------------------
    # Удаляем или переносим старые
    # ------------------------------------------------------------
    if old:
        if not args.delete:
            os.makedirs(BACKUP_DIR, exist_ok=True)

        print("")

        for fname in old:
            src = os.path.join(GRAY_DIR, fname)

            if args.delete:
                os.remove(src)
                print(f"  [DELETE] {fname}")
            else:
                dst = unique_dst(os.path.join(BACKUP_DIR, fname))
                shutil.move(src, dst)
                print(f"  [MOVE] {fname} -> gray_old_backup/")

    # ------------------------------------------------------------
    # Адаптируем оставшиеся
    # ------------------------------------------------------------
    print("")
    print("Адаптация оставшихся шаблонов...")

    kept_paths = []

    for fname in kept:
        path = os.path.join(GRAY_DIR, fname)
        path = ensure_piece_prefix(path)
        kept_paths.append(path)

    if not args.no_resize:
        try:
            cal = load_calibration()

            cell = (
                max(1, int(round(float(cal["cell_w"])))),
                max(1, int(round(float(cal["cell_h"])))),
            )

            print(f"Размер клетки по калибровке: {cell[0]}x{cell[1]}")

            resized = 0

            for path in kept_paths:
                if resize_template(path, cell):
                    resized += 1

            print(f"Ресайз сделано: {resized}/{len(kept_paths)}")

        except FileNotFoundError as e:
            print(f"⚠ Resize пропущен: {e}")

    print("")
    print("✅ Готово.")
    print(f"Серые шаблоны: {GRAY_DIR}")

    if not args.delete and old:
        print(f"Старые шаблоны перенесены в: {BACKUP_DIR}")
        print("Если всё хорошо, можно удалить backup:")
        print("    python clean_gray_templates.py --purge-backup")


if __name__ == "__main__":
    main()