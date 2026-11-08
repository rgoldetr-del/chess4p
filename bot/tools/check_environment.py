"""
check_environment.py

Диагностика окружения перед обновлением бота.

Проверяет:
- Python / Windows / torch / CUDA;
- chess4p_ai API;
- PyGameState methods;
- board_to_tensor;
- set_board signatures;
- load_pgn_verbose tokens;
- model.pt keys/shapes;
- наличие calibration/templates/timer files.

Запуск:
    cd bot
    python tools/check_environment.py
или:
    python tools/check_environment.py path/to/model.pt

Результат пишется в environment_report.txt.
"""

import os
import sys
import inspect
import traceback

# Чтобы можно было импортировать bot-модули, если понадобится
BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

REPORT = []


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    REPORT.append(msg)


def section(name):
    log("")
    log("=" * 70)
    log(f"=== {name}")
    log("=" * 70)


def try_call(label, fn):
    try:
        v = fn()
        log(f"[OK]   {label} -> {type(v).__name__}: {repr(v)[:180]}")
        return v
    except Exception as e:
        log(f"[FAIL] {label} -> {repr(e)[:300]}")
        return None


def probe(label, fn):
    try:
        res = fn()
        log(f"[OK]   {label} -> {repr(res)[:220]}")
        return True
    except Exception as e:
        log(f"[FAIL] {label} -> {repr(e)[:300]}")
        return False


def find_model_path():
    if len(sys.argv) > 1:
        return sys.argv[1]

    candidates = [
        os.path.join(BOT_DIR, "model.pt"),
        os.path.join(os.getcwd(), "model.pt"),
        "model.pt",
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    return candidates[0]


def main():
    section("PYTHON / OS")
    log("sys.version:", sys.version)
    log("sys.executable:", sys.executable)
    log("os.name:", os.name)
    log("platform:", sys.platform)
    log("BOT_DIR:", BOT_DIR)
    log("CWD:", os.getcwd())

    # ------------------------------------------------------------------
    # Packages
    # ------------------------------------------------------------------
    section("PACKAGES")

    np = None
    try:
        import numpy as _np
        np = _np
        log("[OK]   numpy:", _np.__version__)
    except Exception as e:
        log("[FAIL] numpy:", repr(e))

    cv2 = None
    try:
        import cv2 as _cv2
        cv2 = _cv2
        log("[OK]   cv2:", _cv2.__version__)
    except Exception as e:
        log("[FAIL] cv2:", repr(e))

    mss_ok = False
    try:
        import mss  # noqa: F401
        mss_ok = True
        log("[OK]   mss")
    except Exception as e:
        log("[FAIL] mss:", repr(e))

    pyautogui_ok = False
    try:
        import pyautogui  # noqa: F401
        pyautogui_ok = True
        log("[OK]   pyautogui")
    except Exception as e:
        log("[FAIL] pyautogui:", repr(e))

    torch = None
    try:
        import torch as _torch
        torch = _torch
        log("[OK]   torch:", _torch.__version__)

        try:
            log("torch.cuda.is_available():", _torch.cuda.is_available())
            if _torch.cuda.is_available():
                log("torch.version.cuda:", getattr(_torch.version, "cuda", None))
                log("cuda device count:", _torch.cuda.device_count())
                for i in range(_torch.cuda.device_count()):
                    log(f"cuda:{i}:", _torch.cuda.get_device_name(i))
        except Exception as e:
            log("[FAIL] CUDA info:", repr(e))

        try:
            log("cudnn available:", getattr(_torch.backends, "cudnn", None) is not None)
            if hasattr(_torch.backends, "cudnn"):
                log("cudnn.enabled:", _torch.backends.cudnn.enabled)
                log("cudnn.benchmark:", _torch.backends.cudnn.benchmark)
        except Exception as e:
            log("[FAIL] cudnn info:", repr(e))

        try:
            log("allow_tf32 matmul:", _torch.backends.cuda.matmul.allow_tf32)
            log("allow_tf32 cudnn:", _torch.backends.cudnn.allow_tf32)
        except Exception as e:
            log("[WARN] tf32 info:", repr(e))

    except Exception as e:
        log("[FAIL] torch:", repr(e))

    chess4p_ai = None
    try:
        import chess4p_ai as _chess4p_ai
        chess4p_ai = _chess4p_ai
        log("[OK]   chess4p_ai imported")
        log("chess4p_ai file:", getattr(_chess4p_ai, "__file__", "unknown"))
    except Exception as e:
        log("[FAIL] chess4p_ai:", repr(e))
        log(traceback.format_exc())

    # ------------------------------------------------------------------
    # chess4p_ai constants
    # ------------------------------------------------------------------
    section("CHESS4P_AI CONSTANTS")

    if chess4p_ai is not None:
        for name in ["ACTION_SPACE", "TENSOR_PLANES"]:
            try:
                v = getattr(chess4p_ai, name)
                log(f"[OK]   chess4p_ai.{name} = {v}")
            except Exception as e:
                log(f"[FAIL] chess4p_ai.{name} -> {repr(e)}")
    else:
        log("chess4p_ai unavailable")

    # ------------------------------------------------------------------
    # PyGameState API
    # ------------------------------------------------------------------
    section("PyGameState API")

    game = None
    if chess4p_ai is not None:
        try:
            game = chess4p_ai.PyGameState()
            log("[OK]   PyGameState created")
        except Exception as e:
            log("[FAIL] PyGameState():", repr(e))
            log(traceback.format_exc())

    if game is not None:
        methods = [
            "get_turn",
            "get_ply_count",
            "get_scores",
            "get_dead",
            "get_resigned",
            "get_legal_move_indices",
            "make_move",
            "make_move_fast_idx",
            "board_to_tensor",
            "clone_state",
            "set_seed",
            "set_deterministic_dead_king",
            "is_game_over",
            "is_turn_dead",
            "is_turn_resigned",
            "auto_play_dead",
            "auto_play_dead_fast",
            "get_turn_after_idx",
            "mcts_push_idx",
            "terminal_value_if_no_moves",
            "get_final_rewards",
            "get_final_rewards_expected",
            "get_final_placement_soft",
            "is_training_adjudication_draw",
            "resign_player",
            "set_board",
            "load_pgn_verbose",
        ]

        for m in methods:
            has = hasattr(game, m)
            callable_ok = callable(getattr(game, m, None))
            log(f"{'[OK]  ' if has else '[MISS]'} game.{m}: hasattr={has}, callable={callable_ok}")

        # Safe calls
        section("PyGameState SAFE CALLS")

        try_call("game.get_turn()", lambda: game.get_turn())
        try_call("game.get_ply_count()", lambda: game.get_ply_count())
        try_call("game.get_scores()", lambda: game.get_scores())
        try_call("game.get_dead()", lambda: game.get_dead())
        try_call("game.get_resigned()", lambda: game.get_resigned())
        try_call("game.is_game_over()", lambda: game.is_game_over())
        try_call("game.is_turn_dead()", lambda: game.is_turn_dead())
        try_call("game.is_turn_resigned()", lambda: game.is_turn_resigned())

        legal = try_call("game.get_legal_move_indices()", lambda: list(game.get_legal_move_indices()))
        if legal is not None:
            log("legal moves count:", len(legal))
            if len(legal) > 0:
                idx0 = int(legal[0])
                log("first legal idx:", idx0)

                try_call(
                    "game.get_turn_after_idx(idx0)",
                    lambda: game.get_turn_after_idx(idx0),
                )

                clone = try_call("game.clone_state()", lambda: game.clone_state())
                if clone is not None:
                    try_call(
                        "clone.set_deterministic_dead_king(True)",
                        lambda: clone.set_deterministic_dead_king(True),
                    )
                    try_call(
                        "clone.mcts_push_idx(idx0)",
                        lambda: clone.mcts_push_idx(idx0),
                    )

        try_call(
            "game.is_training_adjudication_draw(350, 5)",
            lambda: game.is_training_adjudication_draw(350, 5),
        )

        try_call(
            "game.terminal_value_if_no_moves()",
            lambda: game.terminal_value_if_no_moves(),
        )

        try_call(
            "game.get_final_rewards()",
            lambda: game.get_final_rewards(),
        )

        try_call(
            "game.get_final_rewards_expected()",
            lambda: game.get_final_rewards_expected(),
        )

        try_call(
            "game.get_final_placement_soft()",
            lambda: game.get_final_placement_soft(),
        )

        # board_to_tensor
        if np is not None:
            section("board_to_tensor")
            planes = 256
            try:
                planes = int(getattr(chess4p_ai, "TENSOR_PLANES", 256))
            except Exception:
                pass

            state = np.zeros((planes, 14, 14), dtype=np.float32)
            try:
                game.board_to_tensor(state)
                nonzero = int((state != 0).sum())
                log(f"[OK]   board_to_tensor shape={state.shape}, nonzero={nonzero}")
                if nonzero == 0:
                    log("[WARN] board_to_tensor returned all zeros for initial state")
            except Exception as e:
                log(f"[FAIL] board_to_tensor -> {repr(e)[:300]}")
                log(traceback.format_exc())

        # set_board signature / doc
        section("set_board SIGNATURE / DOC")

        if hasattr(game, "set_board"):
            try:
                sig = inspect.signature(game.set_board)
                log("inspect.signature:", sig)
            except Exception as e:
                log("inspect.signature failed:", repr(e))

            doc = getattr(game.set_board, "__doc__", None)
            if doc:
                log("doc:")
                for line in str(doc).splitlines():
                    log("   ", line)
            else:
                log("doc: None")
        else:
            log("game.set_board missing")

        # Probe set_board variants
        section("set_board PROBES")

        if hasattr(game, "set_board"):
            # Initial-like entries in SCREEN coordinates:
            # tuple: (x, y, color_idx, piece_idx, promoted)
            # piece: pawn=0, knight=1, bishop=2, rook=3, queen=4, king=5
            entries = []

            # RED bottom
            for x, p in [
                (4, 3), (5, 1), (6, 2), (7, 4),
                (8, 5), (9, 2), (10, 1), (11, 3),
            ]:
                entries.append((x, 14, 0, p, 0))
            for x in range(4, 12):
                entries.append((x, 13, 0, 0, 0))

            # BLUE left
            for y, p in [
                (4, 3), (5, 1), (6, 2), (7, 4),
                (8, 5), (9, 2), (10, 1), (11, 3),
            ]:
                entries.append((1, y, 1, p, 0))
            for y in range(4, 12):
                entries.append((2, y, 1, 0, 0))

            # YELLOW top
            for x, p in [
                (4, 3), (5, 1), (6, 2), (7, 5),
                (8, 4), (9, 2), (10, 1), (11, 3),
            ]:
                entries.append((x, 1, 2, p, 0))
            for x in range(4, 12):
                entries.append((x, 2, 2, 0, 0))

            # GREEN right
            for y, p in [
                (4, 3), (5, 1), (6, 2), (7, 5),
                (8, 4), (9, 2), (10, 1), (11, 3),
            ]:
                entries.append((14, y, 3, p, 0))
            for y in range(4, 12):
                entries.append((13, y, 3, 0, 0))

            log(f"entries count: {len(entries)}")

            def probe_set_board(label, call):
                try:
                    g = chess4p_ai.PyGameState()
                except Exception as e:
                    log(f"[FAIL] {label}: cannot create fresh game: {repr(e)[:200]}")
                    return

                try:
                    res = call(g)
                    log(f"[OK]   {label} -> {repr(res)[:200]}")
                except Exception as e:
                    log(f"[FAIL] {label} -> {repr(e)[:300]}")

            probe_set_board(
                "set_board(entries, turn, scores, dead, resigned)",
                lambda g: g.set_board(
                    entries,
                    0,
                    [0, 0, 0, 0],
                    [False, False, False, False],
                    [False, False, False, False],
                ),
            )

            probe_set_board(
                "set_board(entries, turn, scores, dead, resigned) ints/bools",
                lambda g: g.set_board(
                    entries,
                    0,
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ),
            )

            if np is not None:
                probe_set_board(
                    "set_board(entries, turn, np scores/dead/resigned)",
                    lambda g: g.set_board(
                        entries,
                        0,
                        np.zeros(4, dtype=np.int32),
                        np.zeros(4, dtype=bool),
                        np.zeros(4, dtype=bool),
                    ),
                )

            probe_set_board(
                "set_board(entries, turn)",
                lambda g: g.set_board(entries, 0),
            )

            probe_set_board(
                "set_board(entries)",
                lambda g: g.set_board(entries),
            )

        # load_pgn_verbose probes
        section("load_pgn_verbose PROBES")

        if hasattr(game, "load_pgn_verbose"):
            test_tokens = [
                ("empty", ""),
                ("skip_token", "Z"),
                ("resign_token", "R"),
                ("stalemate_token", "S"),
                ("simple_token", "a2-a3"),
                ("mixed_tokens", "Z R S"),
            ]

            for label, text in test_tokens:
                try:
                    g = chess4p_ai.PyGameState()
                    err = g.load_pgn_verbose(text)
                    log(f"[OK]   load_pgn_verbose({label!r}) -> {repr(err)[:200]}")
                except Exception as e:
                    log(f"[FAIL] load_pgn_verbose({label!r}) -> {repr(e)[:300]}")
        else:
            log("load_pgn_verbose missing")

    # ------------------------------------------------------------------
    # Model checkpoint
    # ------------------------------------------------------------------
    section("MODEL CHECKPOINT")

    model_path = find_model_path()
    log("model path:", model_path)
    log("exists:", os.path.exists(model_path))

    if torch is not None and os.path.exists(model_path):
        sd = None

        try:
            sd = torch.load(model_path, map_location="cpu", weights_only=True)
            log("torch.load(weights_only=True): OK")
        except Exception as e:
            log("torch.load(weights_only=True) failed:", repr(e)[:200])
            try:
                sd = torch.load(model_path, map_location="cpu", weights_only=False)
                log("torch.load(weights_only=False): OK")
            except Exception as e2:
                log("torch.load(weights_only=False) failed:", repr(e2)[:300])

        if sd is not None:
            if hasattr(sd, "state_dict"):
                try:
                    sd = sd.state_dict()
                    log("converted object with .state_dict() to state_dict")
                except Exception as e:
                    log("state_dict() failed:", repr(e))

            if isinstance(sd, dict):
                if "model" in sd and isinstance(sd["model"], dict):
                    sd = sd["model"]
                    log("used checkpoint['model']")
                elif "state_dict" in sd and isinstance(sd["state_dict"], dict):
                    sd = sd["state_dict"]
                    log("used checkpoint['state_dict']")
                else:
                    log("checkpoint is raw state_dict or unknown dict")
                    log("top-level keys example:", list(sd.keys())[:20])
            else:
                log("loaded object is not dict:", type(sd).__name__)

            if isinstance(sd, dict):
                cleaned = {}
                for k, v in sd.items():
                    nk = str(k).replace("module.", "")
                    cleaned[nk] = v
                sd = cleaned

                log("state_dict tensors:", len(sd))

                important_keys = [
                    "conv_init.0.weight",
                    "conv_init.0.bias",
                    "conv_init.1.weight",
                    "conv_init.1.bias",
                    "res_blocks.0.conv1.weight",
                    "res_blocks.0.bn1.weight",
                    "res_blocks.29.conv2.weight",
                    "policy_head.0.weight",
                    "policy_head.1.weight",
                    "policy_out.weight",
                    "policy_out.bias",
                    "value_head.0.weight",
                    "value_head.1.weight",
                    "value_fc.0.weight",
                    "value_fc.0.bias",
                    "value_fc.2.weight",
                    "value_fc.4.weight",
                ]

                for k in important_keys:
                    if k in sd:
                        try:
                            shape = tuple(sd[k].shape)
                        except Exception:
                            shape = "unknown"
                        log(f"[OK]   {k}: {shape}")
                    else:
                        log(f"[MISS] {k}")

                # Count res blocks
                block_ids = set()
                for k in sd.keys():
                    if k.startswith("res_blocks."):
                        try:
                            block_id = int(k.split(".")[1])
                            block_ids.add(block_id)
                        except Exception:
                            pass
                if block_ids:
                    log("res_blocks found:", len(block_ids), "max index:", max(block_ids))
                else:
                    log("res_blocks keys not found")

                # Detect old heads
                old_keys = [k for k in sd.keys() if k.startswith("policy_from") or k.startswith("policy_to")]
                if old_keys:
                    log("WARNING: old low-rank policy keys found:", old_keys[:10])
                else:
                    log("old low-rank policy keys: not found")

                if "policy_out.weight" in sd and "value_fc.0.weight" in sd:
                    log("GOOD: new policy/value heads are present in checkpoint")
                else:
                    log("WARNING: new policy/value heads may be missing")
    else:
        if torch is None:
            log("torch unavailable, cannot inspect model")
        if not os.path.exists(model_path):
            log("model file not found")

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------
    section("FILES")

    files_to_check = [
        os.path.join(BOT_DIR, "config.py"),
        os.path.join(BOT_DIR, "calibration_data.json"),
        os.path.join(BOT_DIR, "timer_data.json"),
        os.path.join(BOT_DIR, "templates"),
        os.path.join(BOT_DIR, "model.pt"),
        os.path.join(BOT_DIR, "board_detector.py"),
        os.path.join(BOT_DIR, "board_detecor.py"),
    ]

    for p in files_to_check:
        log(f"{'[OK]  ' if os.path.exists(p) else '[MISS]'} {p}")

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    section("REPORT FILE")
    out_path = os.path.join(os.getcwd(), "environment_report.txt")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(REPORT))
        log("Report saved to:", out_path)
    except Exception as e:
        log("Failed to save report:", repr(e))


if __name__ == "__main__":
    main()