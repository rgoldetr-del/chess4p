"""
filter_clean_ffa.py — отбор чистых FFA-партий для обучения.

Читает:
    dataset_report.json

Пишет:
    clean_ffa_games.json
    clean_ffa/  (копии отобранных PGN4)

Критерии:
- status == "ok"
- Variant == "FFA"
- StartFen4 == "4PC"
- есть Red/Blue/Yellow/Green
- RuleVariants содержит DeadKingWalking
- нет Crazyhouse / Chess960 / Giveaway / Play4Mate / Teams / 2PC
- ply_count >= MIN_PLIES
- optional: средний Elo >= MIN_AVG_ELO
"""

import json
import shutil
from pathlib import Path
from collections import Counter

# ============================================================
# Настройки
# ============================================================

REPORT_JSON = Path("dataset_report.json")
SRC_BASE = Path("партии FFA")
DST_DIR = Path("clean_ffa")
OUT_JSON = Path("clean_ffa_games.json")

# Минимальная длина партии в полуходах.
# Среднее у тебя ~272, так что 120 — довольно мягкий фильтр.
MIN_PLIES = 120

# Если хочешь только сильные партии, поставь 2000 или 2200.
# 0 = без фильтра по Elo.
MIN_AVG_ELO = 0

# Требовать DeadKingWalking.
# Для FFA-датасета это почти всегда нужно.
REQUIRE_DEAD_KING = True

# Копировать файлы в clean_ffa/.
COPY_FILES = True

# Варианты/режимы, которые не должны попасть в FFA-обучение.
BANNED_RULE_SUBSTRINGS = [
    "Crazyhouse",
    "Chess960",
    "Giveaway",
    "Play4Mate",
    "Teams",
    "2PC",
    "Bughouse",
]


# ============================================================
# Helpers
# ============================================================

def get_avg_elo(headers: dict):
    elos = []

    for color in ["Red", "Blue", "Yellow", "Green"]:
        elo = headers.get(f"{color}Elo")

        if elo and str(elo).isdigit():
            elos.append(int(elo))

    if not elos:
        return None

    return sum(elos) / len(elos)


def is_banned_rules(rule_variants: str) -> bool:
    if not rule_variants:
        return False

    lowered = rule_variants.lower()

    for bad in BANNED_RULE_SUBSTRINGS:
        if bad.lower() in lowered:
            return True

    return False


# ============================================================
# Main
# ============================================================

def main():
    if not REPORT_JSON.exists():
        print(f"❌ Не найден {REPORT_JSON}")
        return

    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    games = report.get("games", [])

    print(f"📂 Всего партий в отчете: {len(games)}")

    selected = []
    reject_reasons = Counter()

    if COPY_FILES:
        DST_DIR.mkdir(parents=True, exist_ok=True)

    for g in games:
        status = g.get("status")

        if status != "ok":
            reject_reasons["not_ok"] += 1
            continue

        headers = g.get("headers", {})

        # Только FFA.
        if headers.get("Variant") != "FFA":
            reject_reasons["variant_not_ffa"] += 1
            continue

        # Только стандартный 4PC.
        if headers.get("StartFen4") != "4PC":
            reject_reasons["startfen_not_4pc"] += 1
            continue

        # Должны быть все 4 игрока.
        if not all(headers.get(c) for c in ["Red", "Blue", "Yellow", "Green"]):
            reject_reasons["missing_4_players"] += 1
            continue

        rule_variants = headers.get("RuleVariants", "")

        # Исключаем посторонние режимы.
        if is_banned_rules(rule_variants):
            reject_reasons["banned_rules"] += 1
            continue

        # Для FFA лучше оставлять DeadKingWalking.
        if REQUIRE_DEAD_KING and "DeadKingWalking" not in rule_variants:
            reject_reasons["no_dead_king"] += 1
            continue

        # Длина партии.
        ply_count = g.get("ply_count", 0)

        if ply_count < MIN_PLIES:
            reject_reasons["too_short"] += 1
            continue

        # Elo.
        avg_elo = get_avg_elo(headers)

        if MIN_AVG_ELO > 0:
            if avg_elo is None or avg_elo < MIN_AVG_ELO:
                reject_reasons["low_elo"] += 1
                continue

        # Файл.
        src_path = SRC_BASE / g.get("file", "")

        if not src_path.exists():
            reject_reasons["file_missing"] += 1
            continue

        out_record = dict(g)
        out_record["avg_elo"] = avg_elo

        if COPY_FILES:
            rel = Path(g["file"])
            dst_name = "__".join(rel.parts)
            dst_path = DST_DIR / dst_name

            shutil.copy2(src_path, dst_path)

            out_record["clean_file"] = str(dst_path)

        selected.append(out_record)

    # ============================================================
    # Статистика
    # ============================================================

    total_plies = sum(x.get("ply_count", 0) for x in selected)

    elo_values = [x["avg_elo"] for x in selected if x.get("avg_elo")]
    avg_elo_all = sum(elo_values) / max(1, len(elo_values))

    summary = {
        "total_games_in_report": len(games),
        "selected_clean_ffa": len(selected),
        "rejected": dict(reject_reasons),
        "min_plies": MIN_PLIES,
        "min_avg_elo": MIN_AVG_ELO,
        "require_dead_king": REQUIRE_DEAD_KING,
        "total_selected_plies": total_plies,
        "avg_selected_plies": total_plies / max(1, len(selected)),
        "avg_selected_elo": avg_elo_all,
    }

    OUT_JSON.write_text(
        json.dumps(
            {"summary": summary, "games": selected},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("📊 ИТОГИ ФИЛЬТРАЦИИ")
    print("=" * 60)
    print(f"Всего партий в отчете: {len(games)}")
    print(f"Отобрано чистых FFA: {len(selected)}")
    print(f"Суммарно полуходов: {total_plies:,}")
    print(f"Средняя длина: {summary['avg_selected_plies']:.1f}")
    print(f"Средний Elo: {avg_elo_all:.1f}")
    print()
    print("Причины отказа:")
    for reason, cnt in reject_reasons.most_common():
        print(f"  {reason:25s}: {cnt}")
    print()
    print(f"💾 Список партий: {OUT_JSON.resolve()}")
    if COPY_FILES:
        print(f"💾 Папка с копиями: {DST_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()