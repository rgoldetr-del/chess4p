"""engine_wrapper.py — MCTS через chess4p_ai + модель 30x256x256.

Улучшения:
- правильная загрузка policy/value heads;
- поддержка config.MCTS_NOISE_EPS = 0 для live;
- поддержка torch.compile / channels_last / fp16;
- warmup модели, чтобы первый ход не был аномально долгим.
"""

import os
import threading
import torch
import numpy as np
import chess4p_ai  # type: ignore[import-untyped]

from contextlib import nullcontext

from config import (
    MODEL_CFG,
    MODEL_PATH,
    MODEL_DEVICE,
    USE_FP16,
    USE_CHANNELS_LAST,
    USE_TORCH_COMPILE,

    MCTS_SIMULATIONS,
    MCTS_BATCH_SIZE,
    MCTS_MAX_TIME,
    MCTS_MIN_SIMS,
    MCTS_NOISE_EPS,
    MCTS_C_INIT,
    MCTS_C_BASE,
    MCTS_FPU_REDUCTION,
    MCTS_VIRTUAL_LOSS,
    MCTS_EARLY_STOP_TOP_SHARE,
    MCTS_REUSE_ROOT,
)

from model_utils import (
    create_model,
    load_state_dict_from_path,
    load_model_weights,
)

from selfplay_utils import (
    mcts_search,
    _idx_to_coords,
)


def _setup_torch_backends(device: torch.device):
    """Оптимальные настройки PyTorch для CUDA."""
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    if device.type == "cuda":
        try:
            # GTX 1060 не имеет tensor cores для TF32,
            # но флаг безопасен и может быть полезен на других GPU.
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def game_to_tensor(game) -> np.ndarray:
    """Позиция chess4p_ai -> float32 [256, 14, 14]."""
    state = np.zeros(
        (int(MODEL_CFG["planes"]), 14, 14),
        dtype=np.float32,
    )
    game.board_to_tensor(state)
    return state


class EngineWrapper:
    def __init__(self, model_path=MODEL_PATH, device=None):
        if device is None:
            device = MODEL_DEVICE

        if str(device).lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        _setup_torch_backends(self.device)

        self.use_fp16 = bool(USE_FP16) and self.device.type == "cuda"

        self.model = create_model(
            planes=int(MODEL_CFG["planes"]),
            blocks=int(MODEL_CFG["blocks"]),
            channels=int(MODEL_CFG["channels"]),
        )

        self.model.to(self.device)

        if USE_CHANNELS_LAST and self.device.type == "cuda":
            try:
                self.model = self.model.to(
                    memory_format=torch.channels_last
                )
            except Exception as e:
                print(f"⚠️ channels_last не применён: {e}")

        # ============================================================
        # Загрузка весов
        # ============================================================
        if os.path.exists(model_path):
            sd = load_state_dict_from_path(
                model_path,
                map_location="cpu",
            )
            load_model_weights(self.model, sd, verbose=True)
        else:
            print(f"⚠️  Модель {model_path} не найдена — случайные веса!")

        self.model.eval()

        # torch.compile опционален.
        # На Windows/GTX 1060 может работать нестабильно, поэтому по умолчанию выключен.
        if USE_TORCH_COMPILE:
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                )
                print("[engine] torch.compile enabled")
            except Exception as e:
                print(f"⚠️ torch.compile не удался: {e}")

        self.last_root = None

        # Потокобезопасность + pondering (поиск в чужие ходы)
        self._lock = threading.Lock()
        self._ponder_stop = threading.Event()
        self._ponder_thread = None

        self.warmup()

    # ============================================================
    # Autocast
    # ============================================================

    def _autocast(self):
        if self.use_fp16 and self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    # ============================================================
    # Warmup
    # ============================================================

    def warmup(self):
        """Прогреваем модель, чтобы первый реальный ход был быстрее."""
        try:
            batch = max(1, min(4, int(MCTS_BATCH_SIZE)))
            x = torch.zeros(
                (
                    batch,
                    int(MODEL_CFG["planes"]),
                    14,
                    14,
                ),
                device=self.device,
                dtype=torch.float32,
            )

            with torch.inference_mode(), self._autocast():
                self.model(x)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            print(f"[engine] warmup done on {self.device}")
        except Exception as e:
            print(f"⚠️ engine warmup failed: {e}")

    # ============================================================
    # Main move search
    # ============================================================
    def get_best_move(self, game, simulations=None, max_time=None, verbose=True):
        """Потокобезопасный поиск лучшего хода."""
        with self._lock:
            return self._get_best_move_impl(game, simulations, max_time, verbose)

    def _get_best_move_impl(self, game, simulations, max_time, verbose):
        sims = int(simulations or MCTS_SIMULATIONS)
        min_sims = int(MCTS_MIN_SIMS)
        if min_sims <= 0:
            min_sims = max(64, sims // 4)
            
        cfg = {
            "min_sims": min_sims,
            "max_sims": sims,
            "batch_size": int(MCTS_BATCH_SIZE),
            "max_time": float(max_time or MCTS_MAX_TIME),
            "max_depth": 256,
            "forced_depth": 32,
            "c_init": float(MCTS_C_INIT),
            "c_base": float(MCTS_C_BASE),
            "fpu_reduction": float(MCTS_FPU_REDUCTION),
            "virtual_loss": float(MCTS_VIRTUAL_LOSS),
            "noise_eps": float(MCTS_NOISE_EPS),
            "early_stop_top_share": float(MCTS_EARLY_STOP_TOP_SHARE),
        }
        
        last_root = self.last_root if MCTS_REUSE_ROOT else None

        # Спуск дерева: если соперник только что сделал ход, который уже есть
        # в дереве (после pondering) — переходим в соответствующее поддерево,
        # чтобы не начинать поиск с нуля.
        if last_root is not None:
            try:
                lmi = game.get_last_move_idx()
                if lmi is not None:
                    child = next(
                        (c for c in last_root.children if c.move_idx == int(lmi)),
                        None,
                    )
                    if child is not None:
                        last_root = child
            except Exception:
                pass

        result = mcts_search(
            game,
            self.model,
            self.device,
            last_root=last_root,
            temperature=0.0,
            cfg=cfg,
        )
        
        if result is None:
            return None
            
        best_idx, _top_idx, _top_prob, _legal_packed, new_root, dbg = result
        self.last_root = new_root
        
        if dbg and verbose:
            reused = dbg.get("root_reused", 0)
            print(
                f"  MCTS: sims={dbg.get('simulations', 0)} "
                f"nn={dbg.get('nn_evals', 0)} "
                f"time={dbg.get('time', 0):.2f}s "
                f"root_reused={reused}"
            )
            
        fx, fy, tx, ty = _idx_to_coords(int(best_idx))
        return (fx, fy, tx, ty)

    # ============================================================
    # Root management
    # ============================================================

    def reset_root(self):
        self.last_root = None

    # ============================================================
    # Pondering: фоновый поиск во время ходов соперников
    # ============================================================
    def start_ponder(self, game, simulations=800):
        """Запускает фоновый поиск текущей позиции (небольшими срезами).

        Дерево живёт в self.last_root: когда соперник сделает ход и
        main-поток вызовет get_best_move, произойдёт спуск в поддерево его
        хода — и ход будет готов почти мгновенно.
        """
        if self.device.type != 'cuda':
            return  # на CPU фоновый поиск только мешает
        if self._ponder_thread is not None:
            return
        self._ponder_stop.clear()

        def _loop():
            while not self._ponder_stop.is_set():
                try:
                    g = game.clone_state()
                    self._get_best_move_impl(
                        g,
                        max(128, int(simulations) // 4),
                        1.2,   # срез: чтобы main-поток не ждал дольше ~1.5 с
                        False, # без спама в лог
                    )
                except Exception:
                    return
                self._ponder_stop.wait(0.05)

        self._ponder_thread = threading.Thread(target=_loop, daemon=True)
        self._ponder_thread.start()

    def stop_ponder(self, timeout=2.0):
        """Останавливает фоновый поиск и дожидается текущего среза."""
        if self._ponder_thread is None:
            return
        self._ponder_stop.set()
        try:
            self._ponder_thread.join(timeout=timeout)
        except Exception:
            pass
        self._ponder_thread = None

    # ============================================================
    # Eval only
    # ============================================================

    def evaluate_position(self, game):
        """Возвращает raw policy logits и value logits."""
        state = game_to_tensor(game)
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        with torch.inference_mode(), self._autocast():
            policy, value = self.model(tensor)

        return policy.cpu().numpy(), value.cpu().numpy()