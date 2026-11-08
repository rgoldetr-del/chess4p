import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import chess4p_ai
from config import COLOR_ORDER
from calibration import load_calibration
from screen_capture import capture_screen
from board_detector import BoardDetector, iter_pieces
from human_input_watcher import TimerDetector
from state_sync import KingTracker, try_set_board_resync, replay_resync, GameRecorder
from main import initial_resync, detect_move_from_diff, wait_action
from coordinates import coords_to_idx

def wait_for_stable_board(detector, timeout=3.0):
    """
    Ждет, пока детекция доски перестанет меняться (пока не закончится анимация).
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

def initial_resync(game, detector, tracker, timer):
    """
    Стартовый resync. Ждет стабилизации анимации chess.com.
    """
    print("⏳ Ожидание стабилизации стартовой позиции (анимация chess.com)...")
    time.sleep(2.0) # Даем UI отрисовать все фигуры
    
    last_board = detector.detect_full_board(capture_screen())
    
    # Делаем повторный снимок через 0.5 сек, чтобы убедиться, что доска не "моргает"
    time.sleep(0.5)
    stable_board = detector.detect_full_board(capture_screen())
    
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
            print("❌ set_board не удался. Бот запущен в середине чужой партии без истории PGN.")
            
    print("✅ Стартовая позиция (или set_board не нужен).")
    return last_board

def main():
    print("=== DIAG 02: Turn Sync Shadow Mode ===")
    print("Запусти партию на chess.com. Скрипт будет следить за ходами, но НЕ БУДЕТ кликать.")
    input("Нажми Enter, когда партия начнется и будет ход Красных...")
    
    cal = load_calibration()
    detector = BoardDetector(cal)
    timer = TimerDetector()
    tracker = KingTracker()
    game = chess4p_ai.PyGameState()
    recorder = GameRecorder()
    
    use_timer = timer.is_ready()
    
    last_board = initial_resync(game, detector, tracker, timer)
    
    move_count = 0
    while move_count < 20 and not game.is_game_over():
        turn = game.get_turn()
        color_name = COLOR_ORDER[turn]
        
        print(f"\n[Ожидание] Ход движка: {color_name} (turn={turn})")
        
        # Ждем изменения доски или смены таймера
        acted, _active = wait_action(
            timer,
            use_timer,
            turn,
            detector,
            last_board,
            timeout=300.0,
        )
        
        if not acted:
            print("⚠ timeout ожидания хода")
            continue
            
        # ⚠ КРИТИЧНО: Ждем окончания анимации и стабилизации пикселей
        curr = wait_for_stable_board(detector, timeout=3.0)
        
        # Защита от паники Rust-движка (index out of bounds)
        try:
            mv = detect_move_from_diff(game, last_board, curr, turn, verbose=True)
        except Exception as e:
            print(f"⚠ Паника или ошибка в detect_move_from_diff: {e}")
            print("Движок сломан из-за предыдущего десинка. Остановка.")
            break
        
        if mv:
            print(f"✅ Ход распознан: {mv}")
            idx = coords_to_idx(*mv)
            recorder.move(mv[:2], mv[2:], turn=turn)
            try:
                game.make_move_fast_idx(idx)
            except Exception as e:
                print(f"⚠ Паника при make_move_fast_idx: {e}")
                break
            last_board = curr
            move_count += 1
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

if __name__ == "__main__":
    main()