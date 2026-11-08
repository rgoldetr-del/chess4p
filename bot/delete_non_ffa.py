"""
delete_non_ffa.py — удаление / перенос PGN-файлов, которые не являются FFA 4PC.

По умолчанию работает в dry-run режиме: ничего не удаляет.

Что считается нормальным FFA:
    [Variant "FFA"]
    [StartFen4 "4PC"]
    [Red "..."]
    [Blue "..."]
    [Yellow "..."]
    [Green "..."]
    RuleVariants не содержит Teams / Crazyhouse / Chess960 / Giveaway / Play4Mate / 2PC / Bughouse

Использование:

    # Только посмотреть, что будет удалено
    python delete_non_ffa.py

    # Перенести не-FFA файлы в папку non_ffa_removed
    python delete_non_ffa.py --move

    # Удалить навсегда
    python delete_non_ffa.py --delete

    # Удалить без подтверждения
    python delete_non_ffa.py --delete --yes

    # Не требовать StartFen4 == "4PC"
    python delete_non_ffa.py --no-startfen4
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from collections import Counter


REQUIRED_COLORS = ("Red", "Blue", "Yellow", "Green")

BANNED_RULE_TOKENS = (
    "Teams",
    "Crazyhouse",
    "Chess960",
    "Giveaway",
    "Play4Mate",
    "2PC",
    "Bughouse",
)


def parse_headers_fast(path: Path, max_lines: int = 300):
    """
    Быстро читает только заголовки PGN4.
    Возвращает dict или None, если файл не читается.
    """
    headers = {}

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break

                s = line.strip()

                if not s:
                    continue

                if s.startswith("[") and '"' in s:
                    m = re.match(r'\[(\w+)\s+"([^"]*)"\]', s)
                    if m:
                        headers[m.group(1)] = m.group(2)

                elif not s.startswith("["):
                    # Начался move text.
                    if headers:
                        break

    except Exception:
        return None

    return headers


def classify_file(path: Path, require_startfen4: bool = True):
    """
    Возвращает:
        is_ffa, reason, headers
    """
    headers = parse_headers_fast(path)

    if headers is None:
        return False, "read_error", {}

    if not headers:
        return False, "no_headers", headers

    reasons = []

    variant = headers.get("Variant", "")
    if variant != "FFA":
        reasons.append(f"Variant={variant or 'unknown'}")

    if require_startfen4:
        startfen4 = headers.get("StartFen4", "")
        if startfen4 != "4PC":
            reasons.append(f"StartFen4={startfen4 or 'unknown'}")

    missing_colors = [c for c in REQUIRED_COLORS if not headers.get(c)]
    if missing_colors:
        reasons.append("missing:" + ",".join(missing_colors))

    rule_variants = headers.get("RuleVariants", "")
    bad_tokens = [
        t for t in BANNED_RULE_TOKENS
        if t.lower() in rule_variants.lower()
    ]

    if bad_tokens:
        reasons.append("banned:" + ",".join(bad_tokens))

    if reasons:
        return False, "; ".join(reasons), headers

    return True, "", headers


def unique_path(path: Path) -> Path:
    """Если файл уже существует, добавляет _1, _2 и т.д."""
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def count_reason(reason: str, counter: Counter):
    """Разбивает reason на категории для статистики."""
    parts = [x.strip() for x in reason.split(";") if x.strip()]

    for part in parts:
        if part.startswith("Variant="):
            counter["variant_not_ffa"] += 1
        elif part.startswith("StartFen4="):
            counter["startfen_not_4pc"] += 1
        elif part.startswith("missing:"):
            counter["missing_players"] += 1
        elif part.startswith("banned:"):
            counter["banned_rules"] += 1
        elif part == "read_error":
            counter["read_error"] += 1
        elif part == "no_headers":
            counter["no_headers"] += 1
        else:
            counter["other"] += 1


def main():
    parser = argparse.ArgumentParser(
        description="Удаляет или переносит PGN-файлы, которые не являются FFA 4PC."
    )

    parser.add_argument(
        "--base",
        type=str,
        default="партии FFA",
        help="Папка с PGN-файлами",
    )

    parser.add_argument(
        "--quarantine",
        type=str,
        default="non_ffa_removed",
        help="Папка для переноса файлов в режиме --move",
    )

    parser.add_argument(
        "--report",
        type=str,
        default="delete_non_ffa_report.json",
        help="JSON-отчет",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать, ничего не менять",
    )

    parser.add_argument(
        "--move",
        action="store_true",
        help="Перенести не-FFA файлы в карантин",
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Удалить не-FFA файлы навсегда",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Не спрашивать подтверждение перед удалением",
    )

    parser.add_argument(
        "--no-startfen4",
        action="store_true",
        help="Не требовать StartFen4 == 4PC",
    )

    args = parser.parse_args()

    if args.move and args.delete:
        print("❌ Нельзя использовать --move и --delete одновременно.")
        return

    mode = "dry-run"

    if args.delete:
        mode = "delete"
    elif args.move:
        mode = "move"

    if args.dry_run:
        mode = "dry-run"

    base = Path(args.base).resolve()
    quarantine = Path(args.quarantine).resolve()

    if not base.exists():
        print(f"❌ Папка не найдена: {base}")
        return

    print("=" * 70)
    print("🧹 Удаление / перенос не-FFA PGN")
    print("=" * 70)
    print(f"Папка: {base}")
    print(f"Режим: {mode}")
    print(f"Требовать StartFen4=4PC: {not args.no_startfen4}")
    print()

    files = []

    for p in base.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in (".pgn", ".pgn4"):
            continue

        # Если карантин внутри base, не сканируем его.
        try:
            if quarantine in p.parents:
                continue
        except Exception:
            pass

        files.append(p)

    files = sorted(files)

    print(f"📂 Найдено PGN/PGN4 файлов: {len(files)}")
    print()

    ffa_count = 0
    non_ffa = []
    reason_counter = Counter()

    for p in files:
        is_ffa, reason, headers = classify_file(
            p,
            require_startfen4=not args.no_startfen4,
        )

        if is_ffa:
            ffa_count += 1
            continue

        count_reason(reason, reason_counter)

        try:
            rel = p.relative_to(base)
        except Exception:
            rel = Path(p.name)

        non_ffa.append(
            {
                "file": str(p),
                "rel_file": str(rel),
                "reason": reason,
                "variant": headers.get("Variant", ""),
                "startfen4": headers.get("StartFen4", ""),
                "rule_variants": headers.get("RuleVariants", ""),
                "action": "pending",
            }
        )

    print("=" * 70)
    print("📊 ПРЕДВАРИТЕЛЬНЫЕ ИТОГИ")
    print("=" * 70)
    print(f"Всего файлов: {len(files)}")
    print(f"✅ FFA: {ffa_count}")
    print(f"❌ Не FFA: {len(non_ffa)}")
    print()

    if reason_counter:
        print("Причины:")
        for reason, cnt in reason_counter.most_common():
            print(f"  {reason:25s}: {cnt}")
        print()

    if non_ffa:
        print("Примеры файлов, которые будут обработаны:")
        for rec in non_ffa[:20]:
            print(f"  {rec['rel_file']}")
            print(f"    reason: {rec['reason']}")

        if len(non_ffa) > 20:
            print(f"  ... и ещё {len(non_ffa) - 20} файлов")
        print()

    # ============================================================
    # Применить действия
    # ============================================================

    moved = 0
    deleted = 0

    if mode == "dry-run":
        print("ℹ Dry-run: ничего не изменено.")

    elif mode == "move":
        quarantine.mkdir(parents=True, exist_ok=True)

        print(f"📦 Переношу не-FFA файлы в: {quarantine}")

        for rec in non_ffa:
            src = Path(rec["file"])

            if not src.exists():
                rec["action"] = "missing"
                continue

            dest = quarantine / rec["rel_file"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest = unique_path(dest)

            shutil.move(str(src), str(dest))

            rec["action"] = "moved"
            rec["dest"] = str(dest)
            moved += 1

        print(f"✅ Перенесено файлов: {moved}")

    elif mode == "delete":
        print("⚠ Сейчас файлы будут удалены навсегда.")

        if not args.yes:
            answer = input(
                f"Удалить {len(non_ffa)} не-FFA файлов? (yes/no): "
            ).strip().lower()

            if answer not in ("yes", "y", "да"):
                print("❌ Удаление отменено.")
                return

        for rec in non_ffa:
            src = Path(rec["file"])

            if not src.exists():
                rec["action"] = "missing"
                continue

            try:
                src.unlink()
                rec["action"] = "deleted"
                deleted += 1
            except Exception as e:
                rec["action"] = f"error: {e}"

        print(f"✅ Удалено файлов: {deleted}")

    # ============================================================
    # Отчет
    # ============================================================

    report = {
        "mode": mode,
        "base_dir": str(base),
        "quarantine_dir": str(quarantine),
        "total_files": len(files),
        "ffa_files": ffa_count,
        "non_ffa_files": len(non_ffa),
        "moved": moved,
        "deleted": deleted,
        "reasons": dict(reason_counter.most_common()),
        "files": non_ffa,
    }

    report_path = Path(args.report).resolve()

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"💾 Отчет сохранен: {report_path}")

    print()
    print("=" * 70)
    print("🏁 ГОТОВО")
    print("=" * 70)
    print(f"Всего файлов: {len(files)}")
    print(f"✅ FFA: {ffa_count}")
    print(f"❌ Не FFA: {len(non_ffa)}")

    if mode == "move":
        print(f"📦 Перенесено: {moved}")
    elif mode == "delete":
        print(f"🗑 Удалено: {deleted}")
    else:
        print("ℹ Dry-run: изменения не применялись.")

    print("=" * 70)


if __name__ == "__main__":
    main()