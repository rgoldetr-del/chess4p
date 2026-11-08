"""
validate_and_filter_pgn4.py — валидация и фильтрация скачанных PGN4.
Отбирает только FFA singles с контролем 1+7 или 1+10.
"""

import os
import re
import sys
import glob
import shutil
from pathlib import Path

# Импортируем ваш Rust-движок
try:
    import chess4p_ai
except ImportError:
    print("❌ chess4p_ai не установлен. Установите wheel вашего движка.")
    sys.exit(1)

PGN4_DIR = Path("pgn4_downloads")
FILTERED_DIR = Path("pgn4_filtered")
FILTERED_DIR.mkdir(exist_ok=True)

# Критерии фильтрации для обучающей выборки
WANTED_VARIANTS = ["FFA"]           # Только Free-for-All (не Teams)
WANTED_CONTROLS = ["1+7", "1+10", "60+7", "60+10"]  # Только эти контроли
MIN_PLIES = 40                       # Минимум 40 полуходов (10 ходов)
REQUIRE_RATED = True                 # Только рейтинговые партии


def parse_pgn4_headers(pgn_text: str) -> dict:
    """Извлекает заголовки из PGN4."""
    headers = {}
    for line in pgn_text.split("\n"):
        if line.startswith("[") and '"' in line:
            match = re.match(r'\[(\w+)\s+"([^"]*)"\]', line)
            if match:
                headers[match.group(1)] = match.group(2)
    return headers


def validate_and_filter():
    pgn_files = sorted(PGN4_DIR.glob("game_*.pgn4"))
    print(f"📂 Найдено {len(pgn_files)} PGN4 файлов в {PGN4_DIR}")
    
    stats = {
        "total": len(pgn_files),
        "parsed_ok": 0,
        "parse_failed": 0,
        "filtered_in": 0,
        "filtered_out_variant": 0,
        "filtered_out_control": 0,
        "filtered_out_short": 0,
        "filtered_out_unrated": 0,
    }
    
    for pgn_file in pgn_files:
        pgn_text = pgn_file.read_text(encoding="utf-8")
        headers = parse_pgn4_headers(pgn_text)
        
        # 1. Проверка парсинга Rust-движком
        game = chess4p_ai.PyGameState()
        error = game.load_pgn_verbose(pgn_text)
        if error:
            stats["parse_failed"] += 1
            # print(f"  ⚠ {pgn_file.name}: {error}")
            continue
        stats["parsed_ok"] += 1
        
        # 2. Фильтрация по варианту
        variant = headers.get("Variant", "")
        if variant not in WANTED_VARIANTS:
            stats["filtered_out_variant"] += 1
            continue
        
        # 3. Фильтрация по контролю времени
        tc = headers.get("TimeControl", "")
        if tc not in WANTED_CONTROLS:
            stats["filtered_out_control"] += 1
            continue
        
        # 4. Фильтрация по длине партии
        plies = game.get_ply_count()
        if plies < MIN_PLIES:
            stats["filtered_out_short"] += 1
            continue
        
        # 5. Проверка рейтинговости (по Result или Termination)
        # В FFA результат типа "Player1: 70 - Player2: 30 ..."
        result = headers.get("Result", "")
        if REQUIRE_RATED and "-" not in result and ":" not in result:
            stats["filtered_out_unrated"] += 1
            continue
        
        # ✅ Партия прошла все фильтры — копируем в filtered
        shutil.copy(pgn_file, FILTERED_DIR / pgn_file.name)
        stats["filtered_in"] += 1
    
    # Итоги
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ И ФИЛЬТРАЦИИ")
    print("="*60)
    print(f"📋 Всего файлов: {stats['total']}")
    print(f"✅ Успешно распарсено движком: {stats['parsed_ok']}")
    print(f"❌ Ошибка парсинга: {stats['parse_failed']}")
    print(f"   ├─ Отсеяно по варианту (не FFA): {stats['filtered_out_variant']}")
    print(f"   ├─ Отсеяно по контролю (не 1+7/1+10): {stats['filtered_out_control']}")
    print(f"   ├─ Отсеяно по длине (< {MIN_PLIES} ply): {stats['filtered_out_short']}")
    print(f"   └─ Отсеяно (unrated): {stats['filtered_out_unrated']}")
    print(f"\n🎯 ОТОБРАНО ДЛЯ ОБУЧЕНИЯ: {stats['filtered_in']} партий")
    print(f"📂 Папка: {FILTERED_DIR.absolute()}")
    print("="*60)
    
    return stats


if __name__ == "__main__":
    validate_and_filter()