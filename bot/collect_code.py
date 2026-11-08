r"""
collect_code.py — сбор всех .py / .rs / .toml файлов в один .txt

Использование:
    python collect_code.py C:\chess4p\bot
    python collect_code.py C:\chess4p\bot -o all_code.txt
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================
# Исключения по умолчанию (чтобы не собирать мусор из системных папок)
# ============================================================
DEFAULT_SKIP_DIRS = {
    "venv", ".venv", "__pycache__", ".git", ".hg", ".svn",
    "target", ".pytest_cache", ".mypy_cache", "node_modules",
    ".idea", ".vscode", "dist", "build", ".eggs",
}

# Жестко заданные расширения
ALLOWED_EXTENSIONS = {".py", ".rs", ".toml"}


# ============================================================
# Сбор файлов
# ============================================================
def collect_files(root: Path):
    """Возвращает отсортированный список файлов с нужными расширениями."""
    found = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Пропускаем нежелательные папки
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_SKIP_DIRS]
        
        for fname in filenames:
            if Path(fname).suffix.lower() in ALLOWED_EXTENSIONS:
                found.append(Path(dirpath) / fname)

    # Сортируем по относительному пути для стабильного порядка
    found.sort(key=lambda p: str(p.relative_to(root)).replace("\\", "/"))
    return found


def read_file_safe(path: Path) -> str:
    """Читает файл, пытаясь несколько кодировок."""
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Последний шанс: читаем байты и декодируем с заменами
    return path.read_bytes().decode("utf-8", errors="replace")


# ============================================================
# Запись в один файл
# ============================================================
def build_output(root: Path, files: list) -> str:
    lines = []
    sep = "=" * 80

    # Заголовок
    lines.append(sep)
    lines.append("  CODE COLLECTION")
    lines.append(f"  Root : {root}")
    lines.append(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Files: {len(files)}")
    lines.append(sep)
    lines.append("")

    # Оглавление
    lines.append("  TABLE OF CONTENTS")
    lines.append("-" * 80)
    for i, fpath in enumerate(files, 1):
        rel = fpath.relative_to(root)
        lines.append(f"  {i:4d}. {rel}")
    lines.append("-" * 80)
    lines.append("")
    lines.append("")

    # Содержимое файлов
    for i, fpath in enumerate(files, 1):
        rel = fpath.relative_to(root)
        lines.append(sep)
        lines.append(f"=== [{i}/{len(files)}] {rel} ===")
        # Явно указываем полный абсолютный путь к файлу
        lines.append(f"=== Full path: {fpath.resolve()} ===") 
        lines.append(sep)
        lines.append("")

        content = read_file_safe(fpath)
        lines.append(content)

        # Гарантируем перевод строки после кода
        if content and not content.endswith("\n"):
            lines.append("")

        lines.append("")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Собрать .py, .rs, .toml файлы из указанной папки в один .txt файл"
    )
    parser.add_argument(
        "root",
        type=str,
        help="Путь к папке для сканирования",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Путь к выходному .txt файлу (по умолчанию: <имя_папки>_code.txt в текущей директории)",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Ошибка: папка не найдена: {root}")
        sys.exit(1)

    # Сбор
    print(f"Сканирую папку: {root}")
    print(f"Разрешенные расширения: {sorted(ALLOWED_EXTENSIONS)}")

    files = collect_files(root)

    if not files:
        print("Файлы не найдены.")
        sys.exit(0)

    print(f"Найдено файлов: {len(files)}")

    # Выходной путь
    if args.output:
        out_path = Path(args.output)
    else:
        # Сохраняем в текущую папку (откуда запущен скрипт), 
        # чтобы избежать ошибок прав доступа при записи в корень диска (C:\)
        out_path = Path.cwd() / f"{root.name}_code.txt"

    out_path = out_path.resolve()

    # Запись
    text = build_output(root, files)
    out_path.write_text(text, encoding="utf-8")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Готово: {out_path}  ({size_mb:.2f} MB)")

    # Краткая сводка по расширениям
    ext_count = {}
    for f in files:
        ext = f.suffix.lower()
        ext_count[ext] = ext_count.get(ext, 0) + 1
    for ext in sorted(ext_count):
        print(f"  {ext}: {ext_count[ext]} шт.")


if __name__ == "__main__":
    main()