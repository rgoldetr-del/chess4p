"""
build.py — автоматическая сборка chess4p_ai:
  1. cargo build --release в C:\chess4p\game
  2. maturin develop --release в C:\chess4p\pyo3-bindings

Запуск:
    python C:\chess4p\build.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Пути к проектам
ROOT = Path(r"C:\chess4p")
GAME_DIR = ROOT / "game"
BINDINGS_DIR = ROOT / "pyo3-bindings"


def run_command(cmd, cwd, description):
    """Запускает команду в указанной директории."""
    print(f"\n{'='*70}")
    print(f"🔨 {description}")
    print(f"   Директория: {cwd}")
    print(f"   Команда: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    start = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            check=True,
            text=True,
        )
        elapsed = time.time() - start
        print(f"\n✅ {description} — успешно ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"\n❌ {description} — ОШИБКА ({elapsed:.1f}s)")
        print(f"   Код возврата: {e.returncode}")
        if e.stdout:
            print(f"   stdout: {e.stdout[:500]}")
        if e.stderr:
            print(f"   stderr: {e.stderr[:500]}")
        return False
    except Exception as e:
        print(f"\n❌ {description} — исключение: {e}")
        return False


def main():
    print("="*70)
    print("🚀 СБОРКА chess4p_ai")
    print("="*70)
    
    # Проверка существования директорий
    if not GAME_DIR.exists():
        print(f"❌ Директория не найдена: {GAME_DIR}")
        sys.exit(1)
    if not BINDINGS_DIR.exists():
        print(f"❌ Директория не найдена: {BINDINGS_DIR}")
        sys.exit(1)
    
    # Проверка Cargo.toml
    if not (GAME_DIR / "Cargo.toml").exists():
        print(f"❌ Не найден Cargo.toml в {GAME_DIR}")
        sys.exit(1)
    if not (BINDINGS_DIR / "Cargo.toml").exists():
        print(f"❌ Не найден Cargo.toml в {BINDINGS_DIR}")
        sys.exit(1)
    
    # Проверка pgn.rs
    pgn_rs = GAME_DIR / "src" / "pgn.rs"
    if pgn_rs.exists():
        print(f"✅ Найден pgn.rs: {pgn_rs}")
    else:
        print(f"⚠ pgn.rs не найден: {pgn_rs}")
    
    # Проверка lib.rs
    lib_rs = BINDINGS_DIR / "src" / "lib.rs"
    if lib_rs.exists():
        print(f"✅ Найден lib.rs: {lib_rs}")
    else:
        print(f"⚠ lib.rs не найден: {lib_rs}")
    
    start_total = time.time()
    
    # ШАГ 1: Сборка game
    step1_ok = run_command(
        ["cargo", "build", "--release"],
        GAME_DIR,
        "ШАГ 1: cargo build --release (game)",
    )
    
    if not step1_ok:
        print("\n❌ Сборка game провалилась. Остановка.")
        sys.exit(1)
    
    # ШАГ 2: Сборка pyo3-bindings через maturin
    step2_ok = run_command(
        ["maturin", "develop", "--release"],
        BINDINGS_DIR,
        "ШАГ 2: maturin develop --release (pyo3-bindings)",
    )
    
    total = time.time() - start_total
    
    print(f"\n{'='*70}")
    if step2_ok:
        print(f"✅ ВСЕ ШАГИ УСПЕШНЫ! Общее время: {total:.1f}s")
        print(f"📦 chess4p_ai готов к использованию")
        print(f"\n🧪 Проверка:")
        print(f"   python -c \"import chess4p_ai; g = chess4p_ai.PyGameState(); print('OK:', g.get_ply_count())\"")
    else:
        print(f"❌ Сборка pyo3-bindings провалилась. Общее время: {total:.1f}s")
        sys.exit(1)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()