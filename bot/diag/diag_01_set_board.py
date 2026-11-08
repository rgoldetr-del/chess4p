import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import chess4p_ai
from state_sync import try_set_board_resync, KingTracker
from board_detector import BoardDetector
from screen_capture import capture_screen
from calibration import load_calibration

def main():
    print("=== DIAG 01: set_board fuzzing ===")
    cal = load_calibration()
    detector = BoardDetector(cal)
    tracker = KingTracker()
    game = chess4p_ai.PyGameState()
    
    print("1. Detecting current board...")
    board = detector.detect_full_board(capture_screen())
    tracker.update(board)
    
    dead = game.get_dead()
    res = game.get_resigned()
    
    print("2. Calling try_set_board_resync...")
    ok = try_set_board_resync(game, board, tracker, dead, res, turn=0)
    
    if ok:
        print("✅ SUCCESS! set_board works perfectly.")
        print(f"   Engine turn: {game.get_turn()}")
        print(f"   Engine legal moves: {len(game.get_legal_move_indices())}")
    else:
        print("❌ FAIL! set_board still broken.")

if __name__ == "__main__":
    main()