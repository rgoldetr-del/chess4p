"""config.py — режимы, пути, константы.

Модель:
- 256 плоскостей
- 30 блоков
- 256 каналов

Целевая платформа:
- Windows 10
- CUDA
- GTX 1060 3GB (batch без запаса, поэтому аккуратный)
"""

import os

# ============================================================
# Пути
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
CALIBRATION_FILE = os.path.join(BASE_DIR, "calibration_data.json")
TIMER_CALIB_FILE = os.path.join(BASE_DIR, "timer_data.json")
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")

CALIBRATION_SCREENSHOT = os.path.join(BASE_DIR, "calibration_screenshot.png")
BOARD_CROP_FILE = os.path.join(BASE_DIR, "board_crop.png")

DEBUG_DIR = os.path.join(BASE_DIR, "debug")

# mss monitors[1] обычно основной монитор.
# Если шахматы на другом мониторе, поменяйте индекс.
MONITOR_INDEX = 1


# ============================================================
# Режимы игры
# ============================================================

# Режим watch удалён.
MODES = {
    "red": {
        "bot": ["red"],
        "human": ["blue", "yellow", "green"],
        "observer": False,
    },
    "blue": {
        "bot": ["blue"],
        "human": ["red", "yellow", "green"],
        "observer": False,
    },
    "yellow": {
        "bot": ["yellow"],
        "human": ["red", "blue", "green"],
        "observer": False,
    },
    "green": {
        "bot": ["green"],
        "human": ["red", "blue", "yellow"],
        "observer": False,
    },

    # Бот играет за три цвета, человек за один.
    "pred": {
        "bot": ["blue", "yellow", "green"],
        "human": ["red"],
        "observer": False,
    },
    "pblue": {
        "bot": ["red", "yellow", "green"],
        "human": ["blue"],
        "observer": False,
    },
    "pyellow": {
        "bot": ["red", "blue", "green"],
        "human": ["yellow"],
        "observer": False,
    },
    "pgreen": {
        "bot": ["red", "blue", "yellow"],
        "human": ["green"],
        "observer": False,
    },

    # Бот играет за все четыре цвета.
    "ever": {
        "bot": ["red", "blue", "yellow", "green"],
        "human": [],
        "observer": False,
    },
}

COLOR_ORDER = ["red", "blue", "yellow", "green"]
COLOR_TO_IDX = {"red": 0, "blue": 1, "yellow": 2, "green": 3}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}

BOARD_SIZE = 14
CORNER_CUT = 3
PLAYABLE_SQUARES = 160


# ============================================================
# Human-like задержки / мышь
# ============================================================

THINK_TIME_MIN = 0.5
THINK_TIME_MAX = 1.4

MOUSE_MOVE_DURATION = (0.12, 0.30)
CLICK_PAUSE = (0.03, 0.08)
PREMOVE_DELAY = (0.1, 0.3)

# click / drag / auto
MOVE_METHOD = "auto"


# ============================================================
# Модель
# ============================================================

MODEL_CFG = {
    "planes": 256,
    "blocks": 30,
    "channels": 256,
}

# auto / cuda / cpu
MODEL_DEVICE = "auto"

# Для GTX 1060 3GB:
# - FP16 полезен для памяти;
# - channels_last может помочь;
# - torch.compile на Windows/старых GPU часто нестабилен, поэтому по умолчанию выключен.
USE_FP16 = True
USE_CHANNELS_LAST = True
USE_TORCH_COMPILE = False


# ============================================================
# MCTS для live-игры
# ============================================================

# Пользовательская установка: 800 симуляций.
# Потом можно корректировать по времени и VRAM.
MCTS_SIMULATIONS = 1200

# GTX 1060 3GB:
# 8 - безопасный старт.
# Если будет CUDA OOM, поставить 4.
# Если время на ход слишком маленькое и GPU недогружен, можно попробовать 16.
MCTS_BATCH_SIZE = 4

# Максимальное время на ход в секундах.
MCTS_MAX_TIME = 20.0

# Минимальное число симуляций до остановки по времени/уверенности.
MCTS_MIN_SIMS = 500

# Для live Dirichlet noise лучше выключить.
# Шум нужен в self-play, а не при детерминированной проверке силы.
MCTS_NOISE_EPS = 0.0

MCTS_C_INIT = 1.25
MCTS_C_BASE = 19652.0
MCTS_FPU_REDUCTION = 0.3
MCTS_VIRTUAL_LOSS = 1.0

# Остановка, если лучший ход набрал слишком много посещений.
MCTS_EARLY_STOP_TOP_SHARE = 1.0

# Использовать ли переиспользование дерева MCTS между ходами.
MCTS_REUSE_ROOT = True


# ============================================================
# Детекция доски
# ============================================================

DETECTION_CFG = {
    # Основной порог совпадения цветного шаблона.
    "color_threshold": 0.80,

    # Порог совпадения серого шаблона.
    "gray_threshold": 0.78,

    # Если цветной матч сильнее серого хотя бы на gray_margin,
    # считаем фигуру живой.
    "gray_margin": 0.03,

    "color_min_ratio": 0.3,

    # Эвристики пустой клетки.
    "empty_std": 10.0,
    "canny_low": 90,
    "canny_high": 180,
    "edge_ratio_min": 0.015,

    # Проверка promoted-маркера.
    # Пока лучше выключить.
    "promoted_enabled": True,

    # Если включите, начинайте не с 0.05, а минимум с 0.15-0.20.
    "promoted_white_ratio": 0.15,

        # Разделение живая/мёртвая по насыщенности.
    # Доля "цветных" пикселей в клетке.
    "sat_thresh": 80,       # пиксель считается цветным, если S > 100
    "dead_sat_ratio": 0.12,  # если цветных пикселей меньше 12% -> серая фигура
}


# ============================================================
# Таймеры
# ============================================================

TIMER_CFG = {
    # Минимальный score, чтобы считать таймер активным.
    "min_score": 0.35,

    # Сколько стабильных опросов нужно.
    "stable_polls": 3,
    "stable_interval": 0.12,

    # Пауза между опросами в ожидании смены хода.
    "wait_poll": 0.15,

    # Веса признаков.
    "brightness_weight": 0.45,
    "bright_ratio_weight": 0.30,
    "saturation_weight": 0.25,
}


# ============================================================
# Live-логика
# ============================================================

LIVE_CFG = {
    # Делать set_board-resync при старте, если позиция не начальная.
    "start_resync": True,

    # Как часто проверять OCR счёта.
    "ocr_check_every": 30,

    # Таймауты ожидания.
    "bot_move_timeout": 15.0,
    "human_move_timeout": 300.0,
    "resigned_move_timeout": 60.0,

    # Пауза на анимацию после хода.
    "animation_sleep": 0.5,

    # Подтверждать ход бота по доске перед make_move_fast_idx.
    "confirm_bot_move": True,

    # Сколько раз повторять клик, если ход бота не подтвердился.
    "max_bot_move_retries": 2,

    # Сколько кадров ждать стабилизации доски после изменения.
    "board_stable_polls": 2,
    "board_stable_interval": 0.15,
}