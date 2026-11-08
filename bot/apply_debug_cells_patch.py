import re

p = "board_detector.py"
s = open(p, encoding="utf-8").read()

# 1. Добавляем import time в шапку
if "import time" not in s:
    s = s.replace("import os\nimport cv2\n", "import os\nimport time\nimport cv2\n", 1)
    print("✅ import time added")

# 2. Добавляем поля в __init__ класса BoardDetector
init_old = '''    def __init__(self, cal=None):
        self.cal = cal or load_calibration()
        self.templates = PieceTemplates(self.cal)
        self.templates.prune_empty()'''

init_new = '''    def __init__(self, cal=None):
        self.cal = cal or load_calibration()
        self.templates = PieceTemplates(self.cal)
        self.templates.prune_empty()
        self._dbg_last = {}
        self._dbg_count = 0'''

if init_old in s and "_dbg_last" not in s:
    s = s.replace(init_old, init_new, 1)
    print("✅ BoardDetector debug fields added")

# 3. Добавляем метод _maybe_debug_cell перед detect_full_board
debug_method = '''
    def _maybe_debug_cell(self, cell_img, x, y, cooldown=10.0, max_files=300):
        """Клетка не пустая, но не распознана -> сохранить как дебаг."""
        from config import DEBUG_DIR
        try:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            if _is_empty_cell(gray, DETECTION_CFG):
                return
            now = time.time()
            if now - self._dbg_last.get((x, y), 0.0) < cooldown:
                return
            if self._dbg_count >= max_files:
                return
            self._dbg_last[(x, y)] = now
            self._dbg_count += 1
            out_dir = os.path.join(DEBUG_DIR, "unknown_cells")
            os.makedirs(out_dir, exist_ok=True)
            ts = time.strftime("%H%M%S")
            cv2.imwrite(os.path.join(out_dir, f"cell_x{x}_y{y}_{ts}.png"), cell_img)
        except Exception:
            pass

'''

if "_maybe_debug_cell" not in s:
    s = s.replace(
        "    def detect_full_board(self, screenshot):",
        debug_method + "    def detect_full_board(self, screenshot):",
        1
    )
    print("✅ _maybe_debug_cell method added")

# 4. Добавляем вызов в detect_full_board после result = detect_piece_in_cell(...)
call_old = '''            result = detect_piece_in_cell(
                cell,
                self.templates,
                DETECTION_CFG,
            )
            if result is None:
                continue'''

call_new = '''            result = detect_piece_in_cell(
                cell,
                self.templates,
                DETECTION_CFG,
            )
            if result is None:
                self._maybe_debug_cell(cell, x, y)
                continue'''

if call_old in s and "_maybe_debug_cell(cell, x, y)" not in s:
    s = s.replace(call_old, call_new, 1)
    print("✅ debug call added to detect_full_board")

open(p, "w", encoding="utf-8").write(s)
print("\n🎯 Патч применён. Теперь нераспознанные клетки будут сохраняться в:")
print("   C:\\chess4p\\bot\\debug\\unknown_cells\\")