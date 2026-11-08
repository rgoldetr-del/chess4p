"""
parse_ffa_dataset.py — полный парсинг и валидация датасета партий FFA.

Проходит по папке "партии FFA", проверяет парсинг движком chess4p_ai,
собирает мета-данные (заголовки, рейтинги, исход, длина) и сохраняет JSON-отчеты.

Статусы:
- ok                 -> партия распарсилась полностью без пропущенных токенов
- ok_with_warnings   -> партия распарсилась, но были пропущенные/битые токены
- parse_error        -> движок вернул ошибку парсинга
- read_error         -> файл не читается
"""

import json
import re
import sys
import time
from pathlib import Path
from collections import Counter

try:
    import chess4p_ai
except ImportError:
    print("❌ chess4p_ai не установлен. Установите wheel вашего движка.")
    sys.exit(1)


# Папка с партиями (может содержать подпапки по никам игроков)
BASE_DIR = Path("партии FFA")

OUTPUT_JSON = Path("dataset_report.json")
ERRORS_JSON = Path("dataset_errors.json")
WARNINGS_JSON = Path("dataset_warnings.json")


def parse_headers(pgn_text: str) -> dict:
    """Извлекает заголовки PGN."""
    headers = {}

    for line in pgn_text.split("\n"):
        line = line.strip()

        if not line:
            continue

        if line.startswith("[") and '"' in line:
            match = re.match(r'\[(\w+)\s+"([^"]*)"\]', line)
            if match:
                headers[match.group(1)] = match.group(2)
        elif not line.startswith("["):
            # Как только начались ходы, прерываем.
            break

    return headers


def extract_players_and_ratings(headers: dict) -> list:
    """
    Пытается извлечь игроков и их рейтинги.

    В 4PC могут быть заголовки:
    - Red, Blue, Yellow, Green
    - RedElo, BlueElo, YellowElo, GreenElo
    """
    players = []
    colors = ["Red", "Blue", "Yellow", "Green"]

    for color in colors:
        name = headers.get(color)
        elo = headers.get(f"{color}Elo")

        if name:
            players.append(
                {
                    "color": color,
                    "name": name,
                    "elo": int(elo) if elo and str(elo).isdigit() else None,
                }
            )

    # Fallback на White/Black, если формат нестандартный.
    if not players:
        for color in ["White", "Black"]:
            name = headers.get(color)
            elo = headers.get(f"{color}Elo")

            if name:
                players.append(
                    {
                        "color": color,
                        "name": name,
                        "elo": int(elo) if elo and str(elo).isdigit() else None,
                    }
                )

    return players


def analyze_game(file_path: Path):
    """Анализирует один PGN файл."""
    try:
        pgn_text = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return {"status": "read_error", "error": str(e)}

    headers = parse_headers(pgn_text)

    game = chess4p_ai.PyGameState()

    try:
        error = game.load_pgn_verbose(pgn_text)
    except Exception as e:
        error = str(e)

    warning = None

    if error not in (None, ""):
        error_str = str(error).strip()

        if error_str.startswith("WARN"):
            status = "ok_with_warnings"
            warning = error_str
        else:
            return {
                "status": "parse_error",
                "error": error_str,
                "headers": headers,
                "pgn_preview": pgn_text[:300],
            }
    else:
        status = "ok"

    # Успешный парсинг или парсинг с предупреждениями: собираем метаданные.
    try:
        ply_count = game.get_ply_count()
    except Exception:
        ply_count = -1

    try:
        scores = list(game.get_scores())
    except Exception:
        scores = []

    try:
        dead = list(game.get_dead())
    except Exception:
        dead = []

    try:
        resigned = list(game.get_resigned())
    except Exception:
        resigned = []

    try:
        rewards = list(game.get_final_rewards())
    except Exception:
        rewards = []

    players = extract_players_and_ratings(headers)

    winner_idx = None

    if scores and len(scores) == 4:
        max_score = max(scores)
        winners = [i for i, s in enumerate(scores) if s == max_score]

        if len(winners) == 1:
            winner_idx = winners[0]

    result = {
        "status": status,
        "headers": headers,
        "ply_count": ply_count,
        "scores": scores,
        "dead": dead,
        "resigned": resigned,
        "rewards": rewards,
        "winner_idx": winner_idx,
        "players": players,
        "variant": headers.get("Variant", "Unknown"),
        "time_control": headers.get("TimeControl", "Unknown"),
        "result_str": headers.get("Result", "*"),
    }

    if warning:
        result["warning"] = warning

    return result


def main():
    if not BASE_DIR.exists():
        print(f"❌ Папка {BASE_DIR} не найдена!")
        sys.exit(1)

    # Ищем все pgn и pgn4 файлы рекурсивно.
    pgn_files = list(BASE_DIR.rglob("*.pgn")) + list(BASE_DIR.rglob("*.pgn4"))
    pgn_files = sorted(set(pgn_files))

    print(f"📂 Найдено {len(pgn_files)} файлов в {BASE_DIR.absolute()}")

    stats = {
        "total": len(pgn_files),
        "parsed_ok": 0,
        "parsed_with_warnings": 0,
        "parse_failed": 0,
        "read_failed": 0,
        "total_plies": 0,
        "variants": Counter(),
        "time_controls": Counter(),
        "results": Counter(),
    }

    games_data = []
    warnings_data = []
    errors_data = []

    t0 = time.time()

    for i, pgn_file in enumerate(pgn_files, 1):
        res = analyze_game(pgn_file)

        record = {
            "file": str(pgn_file.relative_to(BASE_DIR)),
            "size_kb": round(pgn_file.stat().st_size / 1024, 1),
        }

        record.update(res)

        if res["status"] == "ok":
            stats["parsed_ok"] += 1

            if res.get("ply_count", 0) > 0:
                stats["total_plies"] += res["ply_count"]

            stats["variants"][res.get("variant", "Unknown")] += 1
            stats["time_controls"][res.get("time_control", "Unknown")] += 1
            stats["results"][res.get("result_str", "*")] += 1

            games_data.append(record)

        elif res["status"] == "ok_with_warnings":
            stats["parsed_with_warnings"] += 1

            if res.get("ply_count", 0) > 0:
                stats["total_plies"] += res["ply_count"]

            stats["variants"][res.get("variant", "Unknown")] += 1
            stats["time_controls"][res.get("time_control", "Unknown")] += 1
            stats["results"][res.get("result_str", "*")] += 1

            games_data.append(record)
            warnings_data.append(record)

        elif res["status"] == "parse_error":
            stats["parse_failed"] += 1
            errors_data.append(record)

        else:
            stats["read_failed"] += 1
            errors_data.append(record)

        if i % 50 == 0 or i == len(pgn_files):
            elapsed = time.time() - t0
            speed = i / elapsed if elapsed > 0 else 0

            print(
                f"⏳ [{i}/{len(pgn_files)}] "
                f"OK: {stats['parsed_ok']} | "
                f"WARN: {stats['parsed_with_warnings']} | "
                f"FAIL: {stats['parse_failed']} | "
                f"Speed: {speed:.1f} files/sec",
                flush=True,
            )

    # Сохранение.
    print("\n💾 Сохранение отчетов...")

    # Сортируем games_data по количеству ходов (убывание).
    games_data.sort(key=lambda x: x.get("ply_count", 0), reverse=True)

    total_usable = stats["parsed_ok"] + stats["parsed_with_warnings"]

    summary = {
        "total_files": stats["total"],
        "parsed_ok": stats["parsed_ok"],
        "parsed_with_warnings": stats["parsed_with_warnings"],
        "parse_failed": stats["parse_failed"],
        "read_failed": stats["read_failed"],
        "total_plies": stats["total_plies"],
        "avg_plies": stats["total_plies"] / max(1, total_usable),
        "variants": dict(stats["variants"]),
        "time_controls": dict(stats["time_controls"].most_common(20)),
        "top_results": dict(stats["results"].most_common(10)),
    }

    final_report = {"summary": summary, "games": games_data}

    OUTPUT_JSON.write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ERRORS_JSON.write_text(
        json.dumps(errors_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    WARNINGS_JSON.write_text(
        json.dumps(warnings_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("📊 ИТОГИ ПАРСИНГА ДАТАСЕТА")
    print("=" * 60)
    print(f"📋 Всего файлов: {stats['total']}")
    print(f"✅ Полностью чисто: {stats['parsed_ok']}")
    print(f"⚠️ Распарсено с предупреждениями: {stats['parsed_with_warnings']}")
    print(f"❌ Ошибок парсинга движком: {stats['parse_failed']}")
    print(f"📉 Ошибок чтения файла: {stats['read_failed']}")
    print(f"♟️ Суммарно полуходов: {stats['total_plies']:,}")
    print(f"⚡ Среднее полуходов на партию: {summary['avg_plies']:.1f}")
    print(f"🎯 Варианты: {dict(stats['variants'])}")
    print()
    print(f"💾 Отчет о партиях: {OUTPUT_JSON.resolve()}")
    print(f"💾 Отчет о предупреждениях: {WARNINGS_JSON.resolve()}")
    print(f"💾 Отчет об ошибках: {ERRORS_JSON.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()