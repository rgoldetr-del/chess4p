"""main.py — live-бот 4PC под модель 30x256x256.

Изменения:
- режим watch удалён;
- стартовый set_board-resync добавлен;
- рокировка в diff-детекторе исправлена;
- ход бота подтверждается перед make_move_fast_idx;
- при неподтверждении хода делается resync;
- координаты и индексы ходов берутся из coordinates.py;
- можно переопределять sims/time через CLI.
"""

import os
import sys
import time
import argparse


from catchup import run_catchup, burst_detect_move



import chess4p_ai  # type: ignore[import-untyped]

from config import (
    MODES,
    COLOR_ORDER,
    MODEL_PATH,
    LIVE_CFG,
)

from calibration import (
    calibrate,
    load_calibration,
)

from screen_capture import capture_screen
from engine_wrapper import EngineWrapper
from move_executor import execute_move, human_think_delay
from human_input_watcher import TimerDetector, calibrate_timers
from board_detector import BoardDetector, iter_pieces


# Файлы управления из GUI (создаются/удаляются чекбоксом «Играть»)
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAY_FLAG = os.path.join(BOT_DIR, "gui_play.flag")
PAUSE_FLAG = os.path.join(BOT_DIR, "gui_pause.flag")
from state_sync import (
    KingTracker,
    detect_statuses,
    read_scores_ocr,
    scores_diverged,
    GameRecorder,
    try_set_board_resync,
    replay_resync,
)

from coordinates import (
    game_to_screen,
    idx_to_coords,
    coords_to_idx,
)

import os

PGN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_history.pgn")

def dump_pgn(recorder):
    try:
        with open(PGN_FILE, "w", encoding="utf-8") as f:
            f.write(recorder.pgn())
    except Exception:
        pass
# ============================================================
# Diff -> move
# ============================================================

def detect_move_from_diff(
    game,
    last_board,
    curr_board,
    turn_idx,
    verbose=False,
):
    """
    Ищет легальный ход текущего игрока, объясняющий изменения доски.

    Возвращает:
        (fx, fy, tx, ty)
    или None.
    """
    changes = {}

    for y in range(1, 15):
        for x in range(1, 15):
            p = last_board[y][x]
            c = curr_board[y][x]

            if p != c:
                changes[(x, y)] = (p, c)

    if len(changes) < 2:
        return None

    try:
        legal = game.get_legal_move_indices()
    except Exception:
        return None

    for idx in legal:
        fx, fy, tx, ty = idx_to_coords(int(idx))

        f_s = game_to_screen(fx, fy)
        t_s = game_to_screen(tx, ty)

        p_from = last_board[f_s[1]][f_s[0]]
        c_from = curr_board[f_s[1]][f_s[0]]

        p_to_last = last_board[t_s[1]][t_s[0]]
        c_to = curr_board[t_s[1]][t_s[0]]

        # 1. Клетка from в прошлом кадре должна была содержать фигуру
        if p_from is None:
            continue

        # 2. Клетка from теперь пуста (или стала серой/чужой)
        if c_from is not None and c_from[0] == turn_idx and not c_from[3]:
            continue

        # 3. На to теперь должна быть фигура (допускаем, что детектор мог перепутать цвет или пометить как dead)
        if c_to is None:
            continue
            
        # Проверяем тип фигуры: он должен совпадать (или это превращение в ферзя)
        if c_to[1] != p_from[1] and not (c_to[2] and c_to[1] == 4):
            continue
        # Мягкая проверка цвета: если детектор уверен, что это наш цвет - ок.
        # Если нет, но ход легальный и фигура та же - все равно принимаем (защита от багов детектора).
        if c_to[0] != turn_idx and p_from[0] == turn_idx:
            pass

        # Если было взятие, на to не должно быть своей живой фигуры.
        if (
            p_to_last is not None
            and p_to_last[0] == turn_idx
            and not p_to_last[3]
        ):
            continue

        extra = set(changes) - {f_s, t_s}

        if not extra:
            if verbose:
                print(f"  [diff] candidate: ({fx},{fy})->({tx},{ty})")
            return (fx, fy, tx, ty)

        # Предварительно собираем рокировочные изменения.
        vanished_rooks = []
        appeared_rooks = []

        for sq in extra:
            lp, cp = changes[sq]

            if (
                lp is not None
                and cp is None
                and lp[0] == turn_idx
                and lp[1] == 3
            ):
                vanished_rooks.append(sq)

            if (
                lp is None
                and cp is not None
                and cp[0] == turn_idx
                and cp[1] == 3
            ):
                appeared_rooks.append(sq)

        ok = True

        for sq in extra:
            lp, cp = changes[sq]

            # En passant: чужая пешка исчезла.
            if (
                cp is None
                and lp is not None
                and lp[0] != turn_idx
                and lp[1] == 0
            ):
                continue

            # Рокировка: ладья исчезла.
            if (
                lp is not None
                and cp is None
                and lp[0] == turn_idx
                and lp[1] == 3
                and appeared_rooks
            ):
                continue

            # Рокировка: ладья появилась.
            if (
                lp is None
                and cp is not None
                and cp[0] == turn_idx
                and cp[1] == 3
                and vanished_rooks
            ):
                continue

            # Допускаем изменение dead/live флага той же фигуры.
            # Например, армия стала серой между кадрами.
            if (
                lp is not None
                and cp is not None
                and lp[0] == cp[0]
                and lp[1] == cp[1]
                and lp[3] != cp[3]
            ):
                continue

            ok = False
            break

        if not ok:
            continue

        if verbose:
            print(f"  [diff] candidate: ({fx},{fy})->({tx},{ty})")

        return (fx, fy, tx, ty)
    if verbose:
        print(f"  [diff] FAILED to match any legal move for {COLOR_ORDER[turn_idx]}.")
        print(f"  [diff] Total changed cells: {len(changes)}")
        # Печатаем первые 5 изменений, чтобы увидеть, что "моргнуло" на доске
        for sq, (old, new) in list(changes.items())[:5]:
            print(f"    {sq}: {old} -> {new}")
    
    return None

def wait_for_stable_board(detector, timeout=3.0):
    """
    Ждет, пока детекция доски перестанет меняться.
    Возвращает стабильную доску.
    """
    last_board = detector.detect_full_board(capture_screen())
    stable_count = 0
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        time.sleep(0.2)
        curr = detector.detect_full_board(capture_screen())
        
        diff = sum(1 for y in range(1, 15) for x in range(1, 15) if curr[y][x] != last_board[y][x])
        
        if diff == 0:
            stable_count += 1
            if stable_count >= 2: # Доска не менялась 0.4 секунды
                return curr
        else:
            stable_count = 0
            last_board = curr
            
    return last_board # Таймаут, возвращаем последнюю версию

def get_consensus_board(detector, num_shots=3):
    """
    Делает несколько снимков и берет консенсус.
    Убирает шум JPEG, тени и случайные галлюцинации детектора.
    """
    boards = []
    for _ in range(num_shots):
        boards.append(detector.detect_full_board(capture_screen()))
        time.sleep(0.1) # 100мс между кадрами
    
    consensus = [[None]*15 for _ in range(15)]
    for y in range(1, 15):
        for x in range(1, 15):
            votes = {}
            for b in boards:
                cell = b[y][x]
                votes[cell] = votes.get(cell, 0) + 1
            # Берем клетку, которая повторилась минимум 2 раза из 3
            best_cell = max(votes, key=votes.get)
            consensus[y][x] = best_cell
    return consensus

# ============================================================
# Wait action
# ============================================================

def wait_action(timer, use_timer, current, detector, last_board, timeout=300.0):
    """
    Ждёт действия: смену таймера ИЛИ устойчивое изменение доски (что раньше).
    Раньше при откалиброванных таймерах мы ждали ТОЛЬКО таймер, и если детект
    таймера промахивался - висели до timeout (до 300 с).

    Возвращает:
    (acted: bool, active_color: int | None)
    """
    deadline = time.time() + timeout
    prev_diff_squares = None

    while time.time() < deadline:
        if use_timer:
            try:
                act, _ = timer.wait_for_turn_change(current, timeout=0.3)
                if act is not None:
                    return True, act
            except Exception:
                pass

        time.sleep(0.25)
        try:
            curr = detector.detect_full_board(capture_screen())
        except Exception:
            continue

        changed = frozenset(
            (x, y)
            for y in range(1, 15)
            for x in range(1, 15)
            if curr[y][x] != last_board[y][x]
        )

        if len(changed) >= 2:
            # Подтверждаем изменение дважды подряд (защита от анимаций/миганий):
            # набор изменившихся клеток должен повторяться в двух кадрах подряд.
            if changed == prev_diff_squares:
                return True, None
            prev_diff_squares = changed
        else:
            prev_diff_squares = None

    return False, None


# ============================================================
# Status sync helpers
# ============================================================

def sync_statuses(
    game,
    last_board,
    tracker,
    recorder,
    pending_status,
):
    statuses = detect_statuses(last_board, tracker)
        
    marked_dead = game.get_dead()
    marked_res = game.get_resigned()

    for ci, kind in statuses.items():
        if kind == "dead" and marked_dead[ci]:
            continue

        if kind == "resigned" and (marked_res[ci] or marked_dead[ci]):
            continue

        pending_status[ci] = kind
        print(f"  [sync] color {COLOR_ORDER[ci]} -> {kind}")
       

def apply_pending_status(
    game,
    turn,
    pending_status,
    recorder,
    last_board,
    tracker,
):
    """
    Применяет отложенный статус, если сейчас ход этого цвета.

    Возвращает:
        (game, resynced: bool)
    """
    if turn not in pending_status:
        return game, False

    kind = pending_status.pop(turn)

    if kind == "resigned":
        recorder.resign(turn=turn)
        game.resign_player(turn)
        return game, False

    # dead: нужен set_board или replay.
    recorder.stalemate(turn=turn)

    dead = game.get_dead()
    res = game.get_resigned()
    dead[turn] = True

    if try_set_board_resync(
        game,
        last_board,
        tracker,
        dead,
        res,
        turn=turn,
    ):
        return game, True

    g2 = replay_resync(recorder)

    if g2 is not None:
        return g2, True

    # Последний шанс: approximating dead as resign.
    game.resign_player(turn)
    return game, False


# ============================================================
# Initial resync
# ============================================================

# main.py (внутри initial_resync)

def initial_resync(game, detector, tracker, timer):
    print("⏳ Ожидание стабилизации стартовой позиции (анимация chess.com)...")
    time.sleep(2.0)
    
    # Ждем, пока capture_screen() не вернет валидный массив
    screenshot = capture_screen()
    while screenshot is None or screenshot.size == 0:
        print("⚠ Не удалось захватить экран. Проверьте, что chess.com открыт и MONITOR_INDEX верен. Повтор через 1 сек...")
        time.sleep(1.0)
        screenshot = capture_screen()

    last_board = detector.detect_full_board(screenshot)
    
    time.sleep(0.5)
    screenshot2 = capture_screen()
    if screenshot2 is not None and screenshot2.size > 0:
        stable_board = detector.detect_full_board(screenshot2)
       
    
    diff = sum(1 for y in range(1, 15) for x in range(1, 15) if stable_board[y][x] != last_board[y][x])
    if diff > 0:
        print(f"  [sync] Стартовая доска стабилизировалась (изменилось {diff} клеток).")
        last_board = stable_board
        
    tracker.update(last_board)
    
    # Считаем живые фигуры. В стандартной начальной позиции их 64 (16 * 4).
    live_pieces = sum(1 for _, _, cell in iter_pieces(last_board) if not cell[3])
    
    if live_pieces < 60:
        print("⚠ Обнаружена середина партии. Пытаюсь set_board-resync...")
        current_turn = 0
        if timer.is_ready():
            active_timer = timer.read_stable()
            if active_timer is not None:
                current_turn = active_timer
        print(f"  [sync] Таймер показывает ход: {COLOR_ORDER[current_turn]}")
        dead = game.get_dead()
        res = game.get_resigned()
        if try_set_board_resync(game, last_board, tracker, dead, res, turn=current_turn):
            print("✅ Стартовый set_board-resync успешен! Движок синхронизирован с экраном.")
            return last_board
        else:
            print("❌ set_board не удался. Бот запущен в середине чужой партии.")
            print("🛑 ОСТАНОВКА. Пожалуйста, запустите бота в начале НОВОЙ партии.")
            import sys
            sys.exit(1)  # <-- ЖЕСТКИЙ ВЫХОД

    print("✅ Стартовая позиция (64 фигуры на месте). Поехали!")
    return last_board
# ============================================================
# Analysis mode (optional)
# ============================================================

def run_analysis_mode(
    pgn_file,
    model_path=MODEL_PATH,
    simulations=None,
    max_time=None,
):
    print("📖 Режим анализа")
    with open(pgn_file, "r", encoding="utf-8") as f:
        pgn = f.read()

    game = chess4p_ai.PyGameState()

    for f in (PLAY_FLAG, PAUSE_FLAG):
        try:
            os.remove(f)
        except OSError:
            pass
    err = game.load_pgn_verbose(pgn)
    if err:
        print(f"❌ Ошибка разбора PGN: {err}")
        return

    engine = EngineWrapper(model_path)
    move_idx = game.get_move_count()
    print(f"PGN загружен: {move_idx} полуходов. Движок продолжает с текущей позиции.")

    while not game.is_game_over():
        if game.is_turn_dead() or game.is_turn_resigned():
            if not game.auto_play_dead_fast():
                break
            continue

        best = engine.get_best_move(
            game,
            simulations=simulations or 400,
            max_time=max_time,
        )
        turn = game.get_turn()
        print(f"Ход {move_idx}: {COLOR_ORDER[turn]} -> {best}")

        if best is None:
            break
        game.make_move_fast_idx(coords_to_idx(*best))
        move_idx += 1

        if move_idx > 300:
            break

    print("Партия завершена.")
    try:
        print(f"Финальные очки: {game.get_scores()}")
        print(f"Placement: {game.get_final_rewards_expected()}")
    except Exception:
        pass


# ============================================================
# Live mode
# ============================================================

def detect_bottom_color(board):
    """Цвет, чьи фигуры стоят в нижних рядах доски (y=1..2).

    В chess.com доска рисуется так, что цвет пользователя всегда снизу.
    """
    counts = {}
    for y in (1, 2):
        for x in range(1, 15):
            cell = board[y][x]
            if cell is not None and not cell[3]:
                counts[cell[0]] = counts.get(cell[0], 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def run_live_mode(
    mode_id,
    model_path=MODEL_PATH,
    simulations=None,
    max_time=None,
    catchup=False,
    ponder=True,
    standby=False,
):
    mode = MODES[mode_id]

    print(f"🎮 Live режим {mode_id}: {mode}")
    print("⚠️  Убедитесь, что chess.com 4PC открыт!")

    try:
        cal = load_calibration()
        print("✅ Калибровка доски загружена")
    except FileNotFoundError:
        print("🔧 Калибровка не найдена. Запускаю...")
        cal = calibrate()

    detector = BoardDetector(cal)
    engine = EngineWrapper(model_path)

    game = chess4p_ai.PyGameState()

    timer = TimerDetector()
    use_timer = timer.is_ready()

    if use_timer:
        print("⏱️  Таймеры откалиброваны: детект хода по смене таймера")
    else:
        print(
            "⚠ Таймеры не откалиброваны "
            "(python human_input_watcher.py) — детект по диффу доски."
        )

    bot_colors = set(mode["bot"])
    human_colors = set(mode["human"])

    tracker = KingTracker()
    recorder = GameRecorder()
    pending_status = {}

    if standby:
        print("🟡 STANDBY: движок и калибровка готовы. Поставьте галочку «Играть» в окне бота.")
        while not os.path.exists(PLAY_FLAG):
            time.sleep(0.5)
            print(".", end="", flush=True)
        try:
            os.remove(PLAY_FLAG)
        except OSError:
            pass
        print("\n✅ Играть! Синхронизируюсь с доской...")

    time.sleep(2)
    if catchup:
        game, recorder, last_board = run_catchup(
            game, detector, tracker, recorder, timer, engine)
        engine.reset_root()
        dump_pgn(recorder)
    else:
        last_board = initial_resync(game, detector, tracker, timer)

    if mode_id == "auto":
        bottom = detect_bottom_color(last_board)
        if bottom is None:
            print("❌ AUTO: не смог определить цвет снизу. Проверьте калибровку.")
            sys.exit(1)
        print(
            f"🎮 AUTO: снизу {COLOR_ORDER[bottom].upper()} — "
            f"бот играет за него, остальные — люди"
        )
        bot_colors = {bottom}
        human_colors = set(range(4)) - bot_colors
    move_count = 0
    ocr_counter = 0
    no_detection_failures = 0

    animation_sleep = float(LIVE_CFG.get("animation_sleep", 0.5))
    bot_move_timeout = float(LIVE_CFG.get("bot_move_timeout", 15.0))
    human_move_timeout = float(LIVE_CFG.get("human_move_timeout", 300.0))
    resigned_move_timeout = float(LIVE_CFG.get("resigned_move_timeout", 60.0))
    max_bot_move_retries = int(LIVE_CFG.get("max_bot_move_retries", 2))
    ocr_every = int(LIVE_CFG.get("ocr_check_every", 30))

    try:
        while not game.is_game_over():
            # ⏸ Пауза по галочке из GUI (снять галочку = пауза)
            while os.path.exists(PAUSE_FLAG):
                time.sleep(0.5)

            # ---------- -1. Защита от паники Rust-движка (PanicException) ----------
            try:
                # Пробуем сгенерировать ходы. Если состояние битое, движок упадет здесь.
                _ = game.get_legal_move_indices()
            except BaseException as e:
                print(f"\n⚠ Движок упал (состояние сломано после resync): {type(e).__name__}")
                print("Пытаюсь восстановиться через PGN replay...")

                g2 = replay_resync(recorder)
                if g2 is not None:
                    game = g2
                    engine.reset_root()
                    # Обновляем доску, чтобы избежать ложных срабатываний diff на следующем шаге
                    last_board = detector.detect_full_board(capture_screen())
                    tracker.update(last_board)
                    continue
                else:
                    # АВАРИЙНЫЙ ЛЮК: PGN побит, а set_board фундаментально сломан для мидгейма.
                    print("\n❌ Критический десинк: replay_resync не удался.")
                    print("Бот потерял историю ходов или был запущен в середине чужой партии.")
                    print("Пожалуйста, запустите бота в самом начале новой партии.")
                    break  # Выходим из цикла while, завершая работу бота без спама ошибками
            # ------------------------------------------------------------
            # 0. Статусы с доски
            # ------------------------------------------------------------
            sync_statuses(
                game,
                last_board,
                tracker,
                recorder,
                pending_status,
            )

            # Периодический OCR sanity check.
            ocr_counter += 1

            if ocr_every > 0 and ocr_counter % ocr_every == 0:
                ocr = read_scores_ocr(cal)

                if scores_diverged(game, ocr):
                    print(
                        f"  [sync] OCR mismatch: ocr={ocr} "
                        f"internal={game.get_scores()} -> resync"
                    )

                    dead = game.get_dead()
                    res = game.get_resigned()

                    if not try_set_board_resync(
                        game,
                        last_board,
                        tracker,
                        dead,
                        res,
                        turn=int(game.get_turn()),
                    ):
                        g2 = replay_resync(recorder)

                        if g2 is not None:
                            game = g2
                            engine.reset_root()

            turn = game.get_turn()
            color_name = COLOR_ORDER[turn]

            # ------------------------------------------------------------
            # 1. Отложенный статус этого цвета
            # ------------------------------------------------------------
            game, resynced = apply_pending_status(
                game,
                turn,
                pending_status,
                recorder,
                last_board,
                tracker,
            )

            if resynced:
                engine.reset_root()
                continue

            # ------------------------------------------------------------
            # 2. Мёртвый ход: skip
            # ------------------------------------------------------------
            if game.is_turn_dead() and not game.is_turn_resigned():
                game.auto_play_dead_fast()
                recorder.skip(turn=turn)

                time.sleep(animation_sleep)

                last_board = detector.detect_full_board(capture_screen())
                tracker.update(last_board)

                continue

            # ------------------------------------------------------------
            # 3. Resigned: король может ходить сам / человеком
            # ------------------------------------------------------------
            if game.is_turn_resigned():
                acted, _active = wait_action(
                    timer,
                    use_timer,
                    turn,
                    detector,
                    last_board,
                    timeout=resigned_move_timeout,
                )

                time.sleep(animation_sleep)

                curr = wait_for_stable_board(detector, timeout=3.0)
                mv = detect_move_from_diff(game, last_board, curr, turn)

                if mv:
                    game.make_move_fast_idx(coords_to_idx(*mv))
                    recorder.move(mv[:2], mv[2:], turn=turn)
                else:
                    game.auto_play_dead_fast()
                    recorder.skip(turn=turn)

                last_board = curr
                tracker.update(curr)

                continue

            # ------------------------------------------------------------
            # 4. Ход бота
            # ------------------------------------------------------------
            if color_name in bot_colors:
                # Предохранитель: если у цвета нет живых фигур на доске — он мёртв.
                live_now = sum(
                    1 for _, _, cell in iter_pieces(last_board)
                    if cell is not None and not cell[3] and cell[0] == turn
                )

                detected_now = sum(1 for _ in iter_pieces(last_board))

                if live_now == 0:
                    # Если детектор видит слишком мало фигур, это проблема распознавания,
                    # а не реальная смерть цвета.
                    if detected_now < 10:
                        no_detection_failures += 1
                        print(
                            f"  ⚠ Детектор видит всего {detected_now} фигур. "
                            f"Проверь калибровку/шаблоны/монитор. "
                            f"Попытка {no_detection_failures}/5"
                        )

                        if no_detection_failures >= 5:
                            print(
                                "🛑 Детекция не работает. "
                                "Запусти: python tools/check_calibration.py"
                            )
                            sys.exit(1)

                        time.sleep(1.0)
                        last_board = get_consensus_board(detector)
                        tracker.update(last_board)
                        continue

                    no_detection_failures = 0

                    print(f"  ⚠ {color_name}: на доске нет живых фигур -> принудительный resync")
                    dead = game.get_dead()
                    dead[turn] = True
                    res = game.get_resigned()

                    ok = try_set_board_resync(
                        game,
                        last_board,
                        tracker,
                        dead,
                        res,
                        turn=turn,
                    )

                    if ok:
                        engine.reset_root()
                    else:
                        print("🛑 set_board не удался. Останавливаюсь, чтобы не зациклиться.")
                        sys.exit(1)

                    continue

                human_think_delay()
                

                best = engine.get_best_move(
                    game,
                    simulations=simulations,
                    max_time=max_time,
                )
                if best is None:
                    print(f"  ⚠ {color_name}: MCTS не дал хода")
                    time.sleep(1)
                    continue
                    
                fx, fy, tx, ty = best

                # 🛡 ЖЕСТКАЯ ПРОВЕРКА: Убедимся, что ход легален в текущем состоянии движка
                try:
                    current_legal = set(int(x) for x in game.get_legal_move_indices())
                    move_idx = coords_to_idx(fx, fy, tx, ty)
                    if move_idx not in current_legal:
                        print(f"  ⚠ MCTS вернул нелегальный ход! (Десинк состояния). Делаю resync...")
                        dead = game.get_dead()
                        res = game.get_resigned()
                        if try_set_board_resync(game, last_board, tracker, dead, res, turn=turn):
                            engine.reset_root()
                            best = engine.get_best_move(game, simulations=simulations, max_time=max_time)
                            if best is None: continue
                            fx, fy, tx, ty = best
                        else:
                            print("  ❌ Resync не удался. Пропускаю ход.")
                            time.sleep(1)
                            continue
                except Exception as e:
                    print(f"  ⚠ Ошибка проверки легальности: {e}")

                fx_s, fy_s = game_to_screen(fx, fy)
                tx_s, ty_s = game_to_screen(tx, ty)
                move_count += 1
                print(
                    f"  🤖 {move_count}: {color_name.upper():6s} "
                    f"({fx},{fy}) -> ({tx},{ty})"
                )
                execute_move(
                    (fx_s, fy_s),
                    (tx_s, ty_s),
                    cal,
                    method="auto",
                )

                acted, _active = wait_action(
                    timer,
                    use_timer,
                    turn,
                    detector,
                    last_board,
                    timeout=bot_move_timeout,
                )

                time.sleep(animation_sleep)

                curr = wait_for_stable_board(detector, timeout=3.0)

                def _confirmed(board):
                    cell = board[ty_s][tx_s]

                    return (
                        cell is not None
                        and cell[0] == turn
                        and not cell[3]
                    )

                retries = 0
                while not _confirmed(curr) and retries < max_bot_move_retries:
                    print("  ⚠ ход не подтверждён (сервер отклонил). Делаю resync и пересчет...")

                    # 1. Синхронизируем движок с реальной доской (чтобы он увидел шах)
                    dead = game.get_dead()
                    res = game.get_resigned()
                    if try_set_board_resync(game, curr, tracker, dead, res, turn=turn):
                        engine.reset_root()
                        # 2. Пересчитываем ход заново
                        new_best = engine.get_best_move(game, simulations=simulations, max_time=max_time)
                        if new_best:
                            best = new_best
                            fx, fy, tx, ty = best
                            fx_s, fy_s = game_to_screen(fx, fy)
                            tx_s, ty_s = game_to_screen(tx, ty)
                            print(f"  🔄 Пересчитал ход: ({fx},{fy}) -> ({tx},{ty})")

                    # 3. Делаем клик новым ходом
                    execute_move((fx_s, fy_s), (tx_s, ty_s), cal, method="click")
                    wait_action(timer, use_timer, turn, detector, last_board, timeout=bot_move_timeout)
                    time.sleep(animation_sleep)
                    curr = get_consensus_board(detector) # Используем консенсус!
                    retries += 1
        
                if not _confirmed(curr):
                    print("  ⚠ ход бота не подтверждён -> resync")

                    dead = game.get_dead()
                    res = game.get_resigned()

                    if try_set_board_resync(
                        game,
                        curr,
                        tracker,
                        dead,
                        res,
                        turn=turn,
                    ):
                        engine.reset_root()
                    else:
                        g2 = replay_resync(recorder)

                        if g2 is not None:
                            game = g2
                            engine.reset_root()
                        else:
                            engine.reset_root()

                    last_board = curr
                    tracker.update(curr)

                    continue

                game.make_move_fast_idx(coords_to_idx(*best))
                recorder.move(best[:2], best[2:], turn=turn)
                dump_pgn(recorder)
                last_board = curr
                tracker.update(curr)

                continue

            # ------------------------------------------------------------
            # 5. Ход человека / внешнего игрока
            # ------------------------------------------------------------
            if color_name in human_colors:
                sys.stdout.write(
                    f"\r  ⏳ Ход: {color_name.upper()}. "
                    f"Жду смену таймера/изменение доски...   "
                )
                sys.stdout.flush()

                # 🧠 Фоновый поиск текущей позиции, пока соперники думают
                if ponder:
                    engine.start_ponder(game, simulations=simulations or 800)

                acted, active_after = wait_action(
                    timer,
                    use_timer,
                    turn,
                    detector,
                    last_board,
                    timeout=human_move_timeout,
                )

                engine.stop_ponder()

                if not acted:
                    print("\n⚠ timeout ожидания хода")
                    continue

                curr, mv = burst_detect_move(
                    game, detector, turn, last_board,
                    detect_move_from_diff, max_wait=8.0,
                    consensus_fn=lambda: get_consensus_board(detector))

                if mv:
                    print(
                        f"\r  📥 {color_name.upper()}: "
                        f"({mv[0]},{mv[1]}) -> ({mv[2]},{mv[3]})"
                    )

                    game.make_move_fast_idx(coords_to_idx(*mv))
                    recorder.move(mv[:2], mv[2:], turn=turn)
                    dump_pgn(recorder)
                    last_board = curr
                    tracker.update(curr)
                else:
                    print(f"❌ ДИФФ НЕ РАСПОЗНАН для {color_name}!")
                    dead = game.get_dead()
                    res = game.get_resigned()

                    # 🚀 УМНАЯ СИНХРОНИЗАЦИЯ: Читаем актуальный ход из таймера!
                    # Это спасает, если игроки ходят очень быстро и мы пропустили 1-2 хода.
                    current_turn = turn
                    if timer.is_ready():
                        active = timer.read_stable()
                        if active is not None:
                            current_turn = active

                    if try_set_board_resync(game, curr, tracker, dead, res, turn=current_turn):
                        print(f"✅ Аварийный set_board спас ситуацию! (Синхронизировано по таймеру: {COLOR_ORDER[current_turn]})")
                        engine.reset_root()
                        last_board = curr
                    else:
                        print("⚠ set_board отклонен. Пытаюсь PGN replay...")
                        g2 = replay_resync(recorder)
                        if g2 is not None:
                            game = g2
                            engine.reset_root()
                            last_board = curr 
                        else:
                            print("❌ Полный десинк. Остановка.")
                            break
                            continue

            # Если цвет не bot и не human, просто ждём.
            time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\n⏹ Остановлено. Ходов сделано: {move_count}")

    try:
        print(f"🏁 Партия завершена. Очки: {game.get_final_rewards()}")
    except Exception:
        pass


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Chess4P FFA Bot")

    parser.add_argument(
        "--mode",
        type=str,
        default="ever",
        choices=list(MODES.keys()),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
    )

    parser.add_argument(
        "--analysis",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--calibrate",
        action="store_true",
    )

    parser.add_argument(
        "--calibrate-timers",
        action="store_true",
    )

    parser.add_argument(
        "--standby",
        action="store_true",
        help="Standby: загрузиться и ждать галочку «Играть» (gui_play.flag)",
    )

    parser.add_argument(
        "--no-ponder",
        action="store_true",
        help="Отключить фоновый поиск во время ходов соперников",
    )

    parser.add_argument(
        "--sims",
        type=int,
        default=None,
        help="Переопределить MCTS_SIMULATIONS для live",
    )

    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Переопределить MCTS_MAX_TIME для live",
    )
    parser.add_argument(
        "--catchup",
        action="store_true",
        help="Режим catch-up: я быстро пролистаю партию, бот трекает по подсветке",
    )
    args = parser.parse_args()

    if args.calibrate:
        calibrate()
        return

    if args.calibrate_timers:
        calibrate_timers()
        return

    if args.analysis:
        run_analysis_mode(
            args.analysis,
            args.model,
            simulations=args.sims,
            max_time=args.time,
        )
    else:
        run_live_mode(
            args.mode,
            args.model,
            simulations=args.sims,
            max_time=args.time,
            catchup=args.catchup,
            ponder=not args.no_ponder,
            standby=args.standby,
        )


if __name__ == "__main__":
    main()