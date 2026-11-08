"""benchmark.py — быстрый бенчмарк capture / detection / NN / MCTS.

Запуск:
    cd bot
    python tools/benchmark.py

Полезные варианты:
    python tools/benchmark.py --no-capture
    python tools/benchmark.py --no-detect
    python tools/benchmark.py --no-nn
    python tools/benchmark.py --no-mcts
    python tools/benchmark.py --sims 128
    python tools/benchmark.py --batch 8
"""

import os
import sys
import time
import argparse

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

import numpy as np
import torch

from config import (
    MODEL_CFG,
    MCTS_BATCH_SIZE,
    MCTS_MAX_TIME,
)

from screen_capture import capture_screen
from calibration import load_calibration, board_region
from board_detector import BoardDetector


def section(name):
    print("")
    print("=" * 70)
    print(f"=== {name}")
    print("=" * 70)


# ============================================================
# Capture
# ============================================================

def bench_capture(n_full=20, n_board=20):
    section("SCREEN CAPTURE")

    # Full screen
    t0 = time.perf_counter()

    img = None
    for _ in range(n_full):
        img = capture_screen()

    dt = time.perf_counter() - t0

    if img is not None:
        print(
            f"full screen: {n_full} captures, "
            f"{dt:.3f}s, {n_full / max(dt, 1e-6):.1f} FPS, "
            f"shape={img.shape}"
        )
    else:
        print("full screen capture failed")

    # Board region
    try:
        cal = load_calibration()
        region = board_region(cal)
    except Exception as e:
        print(f"board region skipped: {e}")
        return

    t0 = time.perf_counter()

    img = None
    for _ in range(n_board):
        img = capture_screen(region)

    dt = time.perf_counter() - t0

    if img is not None:
        print(
            f"board region: {n_board} captures, "
            f"{dt:.3f}s, {n_board / max(dt, 1e-6):.1f} FPS, "
            f"shape={img.shape}"
        )
    else:
        print("board region capture failed")


# ============================================================
# Detection
# ============================================================

def bench_detection(n=5):
    section("BOARD DETECTION")

    try:
        cal = load_calibration()
    except Exception as e:
        print(f"detection skipped: {e}")
        return

    detector = BoardDetector(cal)

    # Warmup
    shot = capture_screen()
    detector.detect_full_board(shot)

    times = []

    for _ in range(n):
        shot = capture_screen()

        t0 = time.perf_counter()
        board = detector.detect_full_board(shot)
        dt = time.perf_counter() - t0

        times.append(dt)

    total_pieces = sum(
        1
        for y in range(1, 15)
        for x in range(1, 15)
        if board[y][x] is not None
    )

    avg = float(np.mean(times))

    print(
        f"detect_full_board: {n} runs, "
        f"avg={avg * 1000:.1f} ms, "
        f"{1.0 / max(avg, 1e-6):.2f} FPS"
    )

    print(f"pieces on last frame: {total_pieces}")


# ============================================================
# NN
# ============================================================

def bench_nn(engine, batches=(1, 4, 8, 16), iters=10):
    section("NN FORWARD")

    planes = int(MODEL_CFG.get("planes", 256))

    for b in batches:
        try:
            x = torch.randn(
                (b, planes, 14, 14),
                device=engine.device,
                dtype=torch.float32,
            )

            # Warmup
            with torch.inference_mode(), engine._autocast():
                engine.model(x)

            if engine.device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            for _ in range(iters):
                with torch.inference_mode(), engine._autocast():
                    engine.model(x)

                if engine.device.type == "cuda":
                    torch.cuda.synchronize()

            dt = time.perf_counter() - t0

            print(
                f"batch={b:2d}: {dt / iters * 1000:8.1f} ms/img, "
                f"{(b * iters) / max(dt, 1e-6):7.1f} positions/s"
            )

        except Exception as e:
            print(f"batch={b:2d}: failed -> {e}")


# ============================================================
# MCTS
# ============================================================

def bench_mcts(engine, sims=128, batch=None, max_time=5.0):
    section("MCTS SEARCH")

    import chess4p_ai  # type: ignore[import-untyped]
    from selfplay_utils import mcts_search

    game = chess4p_ai.PyGameState()

    cfg = {
        "min_sims": max(16, sims // 4),
        "max_sims": sims,
        "batch_size": batch or int(MCTS_BATCH_SIZE),
        "max_time": float(max_time or MCTS_MAX_TIME),
        "max_depth": 256,
        "forced_depth": 32,

        # Benchmark должен быть детерминированным.
        "noise_eps": 0.0,

        "reset_virtual_loss": True,
        "early_stop_top_share": 0.98,
        "early_stop_margin_share": 0.0,
    }

    t0 = time.perf_counter()

    result = mcts_search(
        game,
        engine.model,
        engine.device,
        last_root=None,
        temperature=0.0,
        cfg=cfg,
    )

    dt = time.perf_counter() - t0

    if result is None:
        print("MCTS returned None")
        return

    best_idx, _top_idx, _top_prob, _mask, _new_root, dbg = result

    simulations = int(dbg.get("simulations", 0))
    nn_evals = int(dbg.get("nn_evals", 0))

    print(f"best_idx: {best_idx}")
    print(f"time: {dt:.3f}s")
    print(f"simulations: {simulations}")
    print(f"nn_evals: {nn_evals}")

    if dt > 0:
        print(f"sims/sec: {simulations / dt:.1f}")
        print(f"nn_evals/sec: {nn_evals / dt:.1f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Bot benchmark")

    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--no-detect", action="store_true")
    parser.add_argument("--no-nn", action="store_true")
    parser.add_argument("--no-mcts", action="store_true")

    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--mcts-time", type=float, default=5.0)

    args = parser.parse_args()

    if not args.no_capture:
        try:
            bench_capture()
        except Exception as e:
            print(f"capture benchmark failed: {e}")

    if not args.no_detect:
        try:
            bench_detection()
        except Exception as e:
            print(f"detection benchmark failed: {e}")

    engine = None

    if (not args.no_nn) or (not args.no_mcts):
        from engine_wrapper import EngineWrapper

        print("")
        print("Loading EngineWrapper...")
        engine = EngineWrapper()

    if not args.no_nn and engine is not None:
        try:
            bench_nn(engine)
        except Exception as e:
            print(f"NN benchmark failed: {e}")

    if not args.no_mcts and engine is not None:
        try:
            bench_mcts(
                engine,
                sims=args.sims,
                batch=args.batch,
                max_time=args.mcts_time,
            )
        except Exception as e:
            print(f"MCTS benchmark failed: {e}")


if __name__ == "__main__":
    main()