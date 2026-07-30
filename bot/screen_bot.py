"""
Chess4P Screen Bot - plays chess by reading screen and controlling mouse.
Integrates with existing chess4p AI (MCTS + Neural Network).
"""

import numpy as np
import cv2
import time
import torch
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import IntEnum


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BotConfig:
    """Configuration for screen bot."""
    
    # Screen capture
    board_top: int = 100
    board_left: int = 200
    board_width: int = 896  # 14 * 64
    board_height: int = 896  # 14 * 64
    cell_size: int = 64
    
    # Recognition
    recognition_method: str = "template"  # "template" or "cnn"
    template_match_threshold: float = 0.85
    
    # Mouse control
    mouse_delay_min: float = 0.1
    mouse_delay_max: float = 0.3
    click_pause: float = 0.3
    use_bezier: bool = True
    
    # MCTS
    simulations: int = 800
    batch_size: int = 16
    temperature: float = 0.5
    
    # Colors (RGB)
    red_color: Tuple[int, int, int] = (200, 50, 50)
    blue_color: Tuple[int, int, int] = (50, 50, 200)
    yellow_color: Tuple[int, int, int] = (200, 200, 50)
    green_color: Tuple[int, int, int] = (50, 200, 50)


# ============================================================================
# Piece Types
# ============================================================================

class PieceType(IntEnum):
    EMPTY = 0
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6


class Color(IntEnum):
    RED = 0
    BLUE = 1
    YELLOW = 2
    GREEN = 3


# ============================================================================
# Screen Capture
# ============================================================================

class ScreenCapture:
    """Captures screen regions efficiently."""
    
    def __init__(self, use_dxcam: bool = False):
        self.use_dxcam = use_dxcam
        self.sct = None
        
        if use_dxcam:
            try:
                import dxcam
                self.camera = dxcam.create()
                print("✓ Using DXCam for fast screen capture")
            except ImportError:
                print("⚠ DXCam not available, falling back to mss")
                self.use_dxcam = False
        
        if not use_dxcam:
            try:
                import mss
                self.sct = mss.mss()
                print("✓ Using MSS for screen capture")
            except ImportError:
                raise ImportError("Please install mss: pip install mss")
    
    def capture(self, top: int, left: int, width: int, height: int) -> np.ndarray:
        """Capture a screen region as RGB numpy array."""
        if self.use_dxcam:
            frame = self.camera.grab(region=(left, top, left + width, top + height))
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            monitor = {"top": top, "left": left, "width": width, "height": height}
            screenshot = self.sct.grab(monitor)
            return np.array(screenshot)[:, :, :3]  # Remove alpha channel


# ============================================================================
# Board Recognition
# ============================================================================

class BoardRecognizer:
    """Recognizes chess board from screen image."""
    
    def __init__(self, config: BotConfig, template_dir: str = "bot/templates"):
        self.config = config
        self.template_dir = template_dir
        self.templates = {}
        
        if config.recognition_method == "template":
            self._load_templates()
        elif config.recognition_method == "cnn":
            self.cnn_model = self._load_cnn_model()
    
    def _load_templates(self):
        """Load piece templates for template matching."""
        colors = ["red", "blue", "yellow", "green"]
        pieces = ["pawn", "knight", "bishop", "rook", "queen", "king"]
        
        for color in colors:
            for piece in pieces:
                name = f"{color}_{piece}"
                path = f"{self.template_dir}/{name}.png"
                try:
                    tmpl = cv2.imread(path, cv2.IMREAD_COLOR)
                    if tmpl is not None:
                        self.templates[name] = cv2.resize(tmpl, (self.config.cell_size, self.config.cell_size))
                        print(f"✓ Loaded template: {name}")
                    else:
                        print(f"⚠ Template not found: {path}")
                except Exception as e:
                    print(f"⚠ Error loading template {path}: {e}")
        
        # Empty cell template
        empty_path = f"{self.template_dir}/empty.png"
        try:
            tmpl = cv2.imread(empty_path, cv2.IMREAD_COLOR)
            if tmpl is not None:
                self.templates["empty"] = cv2.resize(tmpl, (self.config.cell_size, self.config.cell_size))
        except:
            pass
    
    def _load_cnn_model(self):
        """Load CNN model for cell classification."""
        # TODO: Implement CNN-based recognition
        print("⚠ CNN recognition not yet implemented, using template matching")
        return None
    
    def recognize(self, image: np.ndarray) -> np.ndarray:
        """
        Recognize board state from image.
        Returns: 14x14 array with piece information.
                 Each cell: (color, piece_type) or None for empty.
        """
        if self.config.recognition_method == "template":
            return self._recognize_template(image)
        else:
            return self._recognize_cnn(image)
    
    def _recognize_template(self, image: np.ndarray) -> np.ndarray:
        """Template matching approach."""
        board = np.full((14, 14), None, dtype=object)
        
        for y in range(14):
            for x in range(14):
                # Extract cell
                x1 = x * self.config.cell_size
                y1 = y * self.config.cell_size
                x2 = x1 + self.config.cell_size
                y2 = y1 + self.config.cell_size
                
                if x2 > image.shape[1] or y2 > image.shape[0]:
                    continue
                
                cell = image[y1:y2, x1:x2]
                cell_gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
                
                best_match = None
                best_score = 0.0
                
                # Match against all templates
                for name, template in self.templates.items():
                    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
                    
                    result = cv2.matchTemplate(cell_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    score = result.max()
                    
                    if score > best_score and score >= self.config.template_match_threshold:
                        best_score = score
                        best_match = name
                
                if best_match:
                    parts = best_match.split("_")
                    if len(parts) == 2:
                        color_str, piece_str = parts
                        color_map = {"red": Color.RED, "blue": Color.BLUE, 
                                    "yellow": Color.YELLOW, "green": Color.GREEN}
                        piece_map = {"pawn": PieceType.PAWN, "knight": PieceType.KNIGHT,
                                    "bishop": PieceType.BISHOP, "rook": PieceType.ROOK,
                                    "queen": PieceType.QUEEN, "king": PieceType.KING}
                        
                        color = color_map.get(color_str)
                        piece = piece_map.get(piece_str)
                        
                        if color is not None and piece is not None:
                            board[y, x] = (color, piece)
        
        return board
    
    def _recognize_cnn(self, image: np.ndarray) -> np.ndarray:
        """CNN-based recognition (to be implemented)."""
        # TODO: Implement CNN recognition
        return self._recognize_template(image)
    
    def detect_turn(self, image: np.ndarray) -> Optional[Color]:
        """Detect whose turn it is from UI indicators."""
        # This depends on the specific website UI
        # Common approaches:
        # 1. Look for highlighted border around board
        # 2. OCR text indicator
        # 3. Color-coded turn indicator
        
        # Simple heuristic: check for colored indicators at board edges
        # This needs to be customized per website
        
        # Placeholder implementation
        return None


# ============================================================================
# Mouse Control
# ============================================================================

class MouseController:
    """Controls mouse with human-like movements."""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.current_pos = None
        
        try:
            import pyautogui
            self.pyautogui = pyautogui
            pyautogui.FAILSAFE = False  # Disable failsafe for automation
            print("✓ PyAutoGUI initialized")
        except ImportError:
            raise ImportError("Please install pyautogui: pip install pyautogui")
    
    def move_to(self, x: int, y: int, duration: float = None):
        """Move mouse to position with optional smooth movement."""
        if duration is None:
            duration = np.random.uniform(self.config.mouse_delay_min, 
                                        self.config.mouse_delay_max)
        
        if self.config.use_bezier and self.current_pos is not None:
            self._move_bezier(self.current_pos, (x, y), duration)
        else:
            self.pyautogui.moveTo(x, y, duration=duration)
        
        self.current_pos = (x, y)
    
    def _move_bezier(self, start: Tuple[int, int], end: Tuple[int, int], 
                     duration: float, num_points: int = 50):
        """Move along Bezier curve for human-like motion."""
        start_np = np.array(start, dtype=float)
        end_np = np.array(end, dtype=float)
        
        # Control points with random offset
        direction = end_np - start_np
        ctrl1 = start_np + direction * 0.25 + np.random.uniform(-30, 30, 2)
        ctrl2 = start_np + direction * 0.75 + np.random.uniform(-30, 30, 2)
        
        t = np.linspace(0, 1, num_points)
        points = (
            (1-t)**3 * start_np +
            3*(1-t)**2*t * ctrl1 +
            3*(1-t)*t**2 * ctrl2 +
            t**3 * end_np
        )
        
        delay_per_point = duration / num_points
        for point in points:
            self.pyautogui.moveTo(int(point[0]), int(point[1]), _pause=False)
            time.sleep(np.random.uniform(delay_per_point * 0.5, delay_per_point * 1.5))
    
    def click(self):
        """Perform a mouse click."""
        self.pyautogui.click()
        time.sleep(0.05)
    
    def click_cell(self, x: int, y: int, board_origin: Tuple[int, int], 
                   cell_size: int = None):
        """Click on a specific board cell."""
        if cell_size is None:
            cell_size = self.config.cell_size
        
        # Convert board coordinates (1-14, 1-14) to screen pixels
        # Note: y is inverted (1 is top, 14 is bottom in chess, but opposite in screen)
        px = board_origin[0] + (x - 1) * cell_size + cell_size // 2
        py = board_origin[1] + (14 - y) * cell_size + cell_size // 2
        
        self.move_to(px, py)
        time.sleep(0.05)
        self.click()
    
    def make_move(self, from_x: int, from_y: int, to_x: int, to_y: int,
                  board_origin: Tuple[int, int], cell_size: int = None):
        """Execute a move by clicking from and to cells."""
        if cell_size is None:
            cell_size = self.config.cell_size
        
        self.click_cell(from_x, from_y, board_origin, cell_size)
        time.sleep(self.config.click_pause)
        self.click_cell(to_x, to_y, board_origin, cell_size)


# ============================================================================
# Game State Synchronization
# ============================================================================

class GameStateSync:
    """Synchronizes recognized board with chess4p GameState."""
    
    def __init__(self):
        # Import chess4p game module
        try:
            import sys
            sys.path.insert(0, '/workspace/game/src')
            # We'll use Python bindings or recreate state
            print("✓ GameState sync ready")
        except Exception as e:
            print(f"⚠ GameState import issue: {e}")
    
    def sync_board(self, game_state, recognized_board: np.ndarray):
        """
        Sync recognized board with game state.
        recognized_board: 14x14 array of (color, piece_type) or None
        """
        # This requires integration with Rust GameState via PyO3
        # For now, placeholder implementation
        
        for y in range(14):
            for x in range(14):
                cell = recognized_board[y, x]
                if cell is not None:
                    color, piece_type = cell
                    # Set piece in game state
                    # game_state.set_piece(x, y, color, piece_type)
                    pass
    
    def create_tensor(self, recognized_board: np.ndarray, 
                      current_turn: Color) -> torch.Tensor:
        """
        Create input tensor for neural network from recognized board.
        Returns: [1, 73, 14, 14] tensor
        """
        tensor = np.zeros((73, 14, 14), dtype=np.float32)
        
        # Plane encoding (similar to AlphaZero):
        # Planes 0-5: Red pieces (Pawn, Knight, Bishop, Rook, Queen, King)
        # Planes 6-11: Blue pieces
        # Planes 12-17: Yellow pieces
        # Planes 18-23: Green pieces
        # Planes 24-29: Promoted pieces (same order)
        # ... additional planes for castling, en passant, turn, etc.
        
        for y in range(14):
            for x in range(14):
                cell = recognized_board[y, x]
                if cell is not None:
                    color, piece_type = cell
                    
                    # Calculate plane index
                    plane = color * 6 + (piece_type - 1)
                    
                    if 0 <= plane < 24:
                        tensor[plane, y, x] = 1.0
        
        # Add turn indicator planes (planes 24-27 for 4 players)
        tensor[24 + current_turn, :, :] = 1.0
        
        # Add repetition count, move clock, etc. in remaining planes
        
        return torch.from_numpy(tensor).unsqueeze(0)


# ============================================================================
# Main Bot Class
# ============================================================================

class ChessBot:
    """Main bot class integrating all components."""
    
    def __init__(self, config: BotConfig = None):
        self.config = config or BotConfig()
        
        print("\n" + "="*60)
        print("🤖 Chess4P Screen Bot Initializing...")
        print("="*60 + "\n")
        
        # Initialize components
        self.screen_capture = ScreenCapture(use_dxcam=False)  # Use mss by default
        self.recognizer = BoardRecognizer(self.config)
        self.mouse = MouseController(self.config)
        self.state_sync = GameStateSync()
        
        # Load AI model
        self.model = self._load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Device: {self.device}")
        
        # Board origin on screen (to be calibrated)
        self.board_origin = (self.config.board_left, self.config.board_top)
        
        print("\n✓ Bot initialization complete!\n")
    
    def _load_model(self):
        """Load trained neural network model."""
        try:
            from training.model import ChessNet
            
            model = ChessNet(num_blocks=30, channels=256)
            
            # Load weights if available
            model_path = "/workspace/training/model.pt"
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"✓ Model loaded from {model_path}")
            except FileNotFoundError:
                print("⚠ No model weights found, using random initialization")
            
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            return None
    
    def calibrate_board(self):
        """Calibrate board position on screen."""
        print("\n📍 Board Calibration")
        print("Please position the mouse at the TOP-LEFT corner of the board")
        print("and press Ctrl+C within 3 seconds to mark the position...\n")
        
        # Wait for user to position mouse
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        import pyautogui
        pos = pyautogui.position()
        self.board_origin = (pos.x, pos.y)
        
        print(f"\n✓ Board origin set to: {self.board_origin}")
        print("Now position at BOTTOM-RIGHT corner...")
        
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        pos2 = pyautogui.position()
        
        # Calculate cell size
        width = pos2.x - pos.x
        height = pos2.y - pos.y
        cell_size_x = width / 14
        cell_size_y = height / 14
        
        self.config.cell_size = int((cell_size_x + cell_size_y) / 2)
        print(f"✓ Cell size: {self.config.cell_size}px")
        print(f"✓ Board dimensions: {width}x{height}\n")
    
    def think(self, game_state, tensor: torch.Tensor) -> Tuple[Tuple[int, int, int, int], float]:
        """
        Use MCTS to find best move.
        Returns: ((from_x, from_y, to_x, to_y), confidence)
        """
        try:
            from training.mcts import MCTS
            
            mcts = MCTS(
                simulations=self.config.simulations,
                batch_size=self.config.batch_size,
                device=self.device
            )
            
            # Run MCTS search
            root = mcts.search(game_state, tensor, temperature=self.config.temperature)
            
            # Get best move from visit counts
            if root.children:
                visits = np.array([child.visits for child in root.children])
                best_idx = np.argmax(visits)
                best_move = root.legal_moves[best_idx]
                
                total_visits = visits.sum()
                confidence = visits[best_idx] / total_visits if total_visits > 0 else 0.0
                
                # Convert move format (depends on your representation)
                # Assuming (from_x, from_y, to_x, to_y)
                return best_move, confidence
            else:
                print("⚠ No legal moves available")
                return None, 0.0
                
        except Exception as e:
            print(f"⚠ MCTS error: {e}")
            return None, 0.0
    
    def play_loop(self, my_color: Color, delay_between_moves: float = 2.0):
        """
        Main playing loop.
        Continuously monitors screen and makes moves when it's our turn.
        """
        print(f"\n🎮 Starting bot (playing as: {['RED', 'BLUE', 'YELLOW', 'GREEN'][my_color]})")
        print("Press Ctrl+C to stop\n")
        
        last_move_time = 0
        game_state = None  # Initialize from chess4p
        
        try:
            while True:
                # 1. Capture screen
                image = self.screen_capture.capture(
                    self.board_origin[1], 
                    self.board_origin[0],
                    self.config.board_width,
                    self.config.board_height
                )
                
                # 2. Recognize board
                recognized = self.recognizer.recognize(image)
                
                # 3. Detect whose turn it is
                current_turn = self.recognizer.detect_turn(image)
                
                if current_turn is None:
                    # Fallback: assume based on time or other heuristics
                    time.sleep(0.5)
                    continue
                
                # 4. Check if it's our turn
                if current_turn != my_color:
                    time.sleep(0.5)
                    continue
                
                # 5. Sync game state
                # game_state = self.state_sync.sync_board(game_state, recognized)
                
                # 6. Create tensor for model
                tensor = self.state_sync.create_tensor(recognized, current_turn)
                tensor = tensor.to(self.device)
                
                # 7. Think and make move
                print(f"\n🤔 Thinking... (turn: {['RED', 'BLUE', 'YELLOW', 'GREEN'][current_turn]})")
                best_move, confidence = self.think(game_state, tensor)
                
                if best_move:
                    from_x, from_y, to_x, to_y = best_move
                    print(f"✅ Best move: ({from_x},{from_y}) -> ({to_x},{to_y}) [confidence: {confidence:.2f}]")
                    
                    # 8. Execute move with mouse
                    self.mouse.make_move(from_x, from_y, to_x, to_y, self.board_origin)
                    
                    last_move_time = time.time()
                    print("✓ Move executed")
                
                # 9. Wait for opponent
                time.sleep(delay_between_moves)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Bot error: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chess4P Screen Bot")
    parser.add_argument("--color", type=int, default=0, 
                       choices=[0, 1, 2, 3],
                       help="Color to play: 0=Red, 1=Blue, 2=Yellow, 3=Green")
    parser.add_argument("--simulations", type=int, default=800,
                       help="MCTS simulations per move")
    parser.add_argument("--calibrate", action="store_true",
                       help="Run board calibration")
    
    args = parser.parse_args()
    
    config = BotConfig(
        simulations=args.simulations
    )
    
    bot = ChessBot(config)
    
    if args.calibrate:
        bot.calibrate_board()
    else:
        my_color = Color(args.color)
        bot.play_loop(my_color)
