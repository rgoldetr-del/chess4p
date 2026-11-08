import os


# ============================================================
# Paths
# ============================================================

WORKDIR = "/kaggle/working"
LOGDIR = os.path.join(WORKDIR, "logs")

LATEST_MODEL_PATH = os.path.join(WORKDIR, "latest_model.pt")
BEST_MODEL_PATH = os.path.join(WORKDIR, "best_model.pt")
CHECKPOINT_PATH = os.path.join(WORKDIR, "checkpoint.pt")

# Если хочешь продолжить со старой модели:
# - положи model.pt / best_model.pt в input dataset;
# - или скопируй его в /kaggle/working/input_model.pt;
# - или включи скачивание с HF ниже.
INPUT_MODEL_PATTERNS = [
    "/kaggle/working/input_model.pt",
    "/kaggle/input/**/model.pt",
    "/kaggle/input/**/best_model.pt",
    "/kaggle/input/**/latest_model.pt",
    "/kaggle/input/**/*model*.pt",
]


# ============================================================
# Hugging Face
# ============================================================

# Репозиторий, где у тебя лежит best_model.pt
HF_MODEL_REPO = "goldetn/chess4p_data"
HF_MODEL_FILENAME = "best_model.pt"


# ============================================================
# Model config
# ============================================================

# Сейчас сознательно 19x192:
# - можно подхватить старую модель;
# - выше скорость на T4;
# - стабильнее self-play.
#
# Позже можно будет сделать:
#   blocks=30, channels=256
# но это уже будет другая ёмкость и, скорее всего, новый старт.

MODEL_CFG = {
    "planes": 100,
    "blocks": 19,
    "channels": 192,
}


# ============================================================
# MCTS config
# ============================================================

MCTS_CFG = {
    "min_sims": 32,
    "max_sims": 256,
    "batch_size": 16,
    "max_time": 0.08,

    "max_depth": 256,
    "forced_depth": 32,

    "c_init": 1.25,
    "c_base": 19652.0,
    "fpu_reduction": 0.3,
    "virtual_loss": 1.0,
}


# ============================================================
# Self-play worker config
# ============================================================

WORKER_CFG = {
    "base_seed": 12345,

    "game_timeout": 900,
    "emergency_ply": 800,

    "adjudication_ply": 350,
    "adjudication_spread": 5,

    "print_every": 5,
    "policy_k": 192,

    # Если 3 воркера на одной GPU:
    # 3 * 0.30 = 0.90
    #
    # Если OOM, поставь:
    #   0.22-0.25
    # или уменьши num_workers до 2.
    "gpu_memory_fraction": 0.30,

    "model": MODEL_CFG,
    "mcts": MCTS_CFG,
}


# ============================================================
# Training config
# ============================================================

TRAIN_CFG = {
    # GPU split:
    # GPU0: trainer
    # GPU1: self-play workers
    #
    # Если вдруг одна GPU, поставь:
    #   "selfplay_gpus": [0]
    "trainer_gpu": 0,
    "selfplay_gpus": [1],

    # 4 CPU на Kaggle T4x2.
    # 3 воркера — безопасный старт.
    # Если CPU не загружен и очередь пустая, можно попробовать 4.
    "num_workers": 3,

    "queue_maxsize": 1200,
    "buffer_capacity": 40000,

    "batch_size": 512,
    "accum_steps": 1,

    "lr": 1e-3,
    "weight_decay": 5e-4,
    "warmup_steps": 1000,
    "total_steps": 60000,

    "value_mse_weight": 0.1,
    "grad_clip": 1.0,

    # Сколько trainer может занять GPU памяти.
    "trainer_memory_fraction": 0.90,

    # Logging / saving
    "log_every": 20,
    "save_every": 500,

    # Evaluation пока выключена.
    # Когда добавим arena, можно поставить 1000-2000.
    "eval_every": 0,

    # Resume
    "resume": True,

    # TensorBoard
    "tensorboard": True,

    # Пока нет arena, best_model можно обновлять вместе с latest.
    # Позже это нужно заменить на arena-based best update.
    "copy_latest_to_best_each_save": True,

    # Скачать стартовую модель с HF, если локально не найдена.
    "download_best_from_hf": True,

    # HF upload.
    # По умолчанию выключено, чтобы не требовать токен.
    "upload_hf": False,
    "upload_checkpoint": False,

    "hf_repo_id": HF_MODEL_REPO,
    "hf_model_filename": HF_MODEL_FILENAME,
    "hf_secret_name": "HF_TOKEN",
}