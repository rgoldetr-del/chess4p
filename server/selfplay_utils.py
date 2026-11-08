"""
selfplay_utils.py — часть 1/2

Новый фундамент self-play:
- int actions;
- новый engine API;
- dynamic MCTS;
- deterministic dead king в симуляциях;
- sparse policy target;
- packed legal mask;
- soft value targets через движок.
"""

import os
import sys
import math
import time
import random
import signal
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import chess4p_ai

# Если model_utils лежит рядом.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_utils import ChessNet100, load_model_weights, create_model


# ============================================================
# Constants
# ============================================================

ACTION_SPACE = int(getattr(chess4p_ai, "ACTION_SPACE", 38416))
TENSOR_PLANES = int(getattr(chess4p_ai, "TENSOR_PLANES", 100))
MASK_BYTES = (ACTION_SPACE + 7) // 8

# Сколько ненулевых policy-целей храним.
# 192 обычно хватает даже для 256-400 simulations.
POLICY_K = 192

PLACEMENT_CENTERS = np.array(
    [1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0],
    dtype=np.float32,
)


# ============================================================
# Debug flags
# ============================================================

DEBUG_MCTS = os.getenv("DEBUG_MCTS", "0") == "1"
DEBUG_ENGINE = os.getenv("DEBUG_ENGINE", "0") == "1"
DEBUG_AUG = os.getenv("DEBUG_AUG", "0") == "1"
DEBUG_WORKER = os.getenv("DEBUG_WORKER", "0") == "1"


# ============================================================
# Seeds
# ============================================================

def set_worker_seeds(worker_id: int, base_seed: int = 12345):
    seed = base_seed + worker_id * 7919

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Legal mask pack/unpack
# ============================================================

def pack_legal_indices(indices) -> np.ndarray:
    """
    indices: list/np.array of int action indices.
    returns uint8 packed bits length MASK_BYTES.
    """
    packed = np.zeros(MASK_BYTES, dtype=np.uint8)

    if len(indices) == 0:
        return packed

    idx = np.asarray(indices, dtype=np.int64)

    byte_idx = idx // 8
    bit_idx = (idx % 8).astype(np.uint8)

    bits = np.left_shift(np.uint8(1), bit_idx).astype(np.uint8)

    np.bitwise_or.at(packed, byte_idx, bits)

    return packed


def unpack_legal_packed(packed: np.ndarray) -> np.ndarray:
    """
    returns bool array [ACTION_SPACE].
    """
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))
    return bits[:ACTION_SPACE].astype(bool)


# ============================================================
# Move / channel rotation permutations
# ============================================================

_MOVE_PERM = None       # old -> new
_MOVE_INV_PERM = None   # new -> old
_CHANNEL_PERM = None    # new -> old


def _coords_to_idx(fx: int, fy: int, tx: int, ty: int) -> int:
    return (fx - 1) + (fy - 1) * 14 + (tx - 1) * 196 + (ty - 1) * 2744


def _idx_to_coords(idx: int) -> Tuple[int, int, int, int]:
    fx = (idx % 14) + 1
    fy = ((idx // 14) % 14) + 1
    tx = ((idx // 196) % 14) + 1
    ty = ((idx // 2744) % 14) + 1
    return fx, fy, tx, ty


def _rotate_coords_once(fx: int, fy: int, tx: int, ty: int):
    # 90 degrees, согласованно с np.rot90(k=1)
    return fy, 15 - fx, ty, 15 - tx


def _build_move_permutations():
    global _MOVE_PERM, _MOVE_INV_PERM

    if _MOVE_PERM is not None:
        return

    perms = []
    inv_perms = []

    for k in range(4):
        perm = np.empty(ACTION_SPACE, dtype=np.int64)

        for idx in range(ACTION_SPACE):
            fx, fy, tx, ty = _idx_to_coords(idx)

            for _ in range(k):
                fx, fy, tx, ty = _rotate_coords_once(fx, fy, tx, ty)

            perm[idx] = _coords_to_idx(fx, fy, tx, ty)

        inv_perm = np.argsort(perm)

        perms.append(perm)
        inv_perms.append(inv_perm)

    _MOVE_PERM = perms
    _MOVE_INV_PERM = inv_perms


def _build_channel_permutations():
    global _CHANNEL_PERM

    if _CHANNEL_PERM is not None:
        return

    perms = []

    for k in range(4):
        c = np.arange(TENSOR_PLANES, dtype=np.int64)

        if k != 0:
            # Старые совместимые группы
            c[0:24] = np.roll(c[0:24], k * 6)
            c[24:28] = np.roll(c[24:28], k)
            c[28:32] = np.roll(c[28:32], k)
            c[32:36] = np.roll(c[32:36], k)
            c[36:44] = np.roll(c[36:44], k * 2)
            c[44:48] = np.roll(c[44:48], k)
            c[48:52] = np.roll(c[48:52], k)
            c[52:56] = np.roll(c[52:56], k)

            # 56: halfmove clock — без roll

            c[57:61] = np.roll(c[57:61], k)
            c[61:65] = np.roll(c[61:65], k)
            c[65:69] = np.roll(c[65:69], k)
            c[69:73] = np.roll(c[69:73], k)

            # Новые группы
            c[73:77] = np.roll(c[73:77], k)
            c[77:81] = np.roll(c[77:81], k)
            c[81:85] = np.roll(c[81:85], k)
            c[85:89] = np.roll(c[85:89], k)
            c[89:93] = np.roll(c[89:93], k)
            c[93:97] = np.roll(c[93:97], k)

            # 97: repetition — без roll
            # 98: valid mask — без roll
            # 99: last move signed — spatial only, channel без roll

        perms.append(c)

    _CHANNEL_PERM = perms


def build_rotation_tables():
    _build_move_permutations()
    _build_channel_permutations()


# Сразу строим таблицы при импорте.
build_rotation_tables()


# ============================================================
# MCTS node
# ============================================================

class MctsNode:
    __slots__ = (
        "move_idx",
        "parent",
        "children",
        "visits",
        "total_values",
        "prior",
        "prior_orig",
        "turn",
        "virtual_loss",
    )

    def __init__(
        self,
        move_idx: Optional[int] = None,
        parent: Optional["MctsNode"] = None,
        prior: float = 0.0,
        turn: int = 0,
    ):
        self.move_idx = move_idx
        self.parent = parent
        self.children: List["MctsNode"] = []
        self.visits = 0
        self.total_values = np.zeros(4, dtype=np.float32)
        self.prior = float(prior)
        self.prior_orig = float(prior)
        self.turn = int(turn)
        self.virtual_loss = 0.0


# ============================================================
# PUCT selection
# ============================================================

def select_child(
    node: MctsNode,
    c_init: float = 1.25,
    c_base: float = 19652.0,
    fpu_reduction: float = 0.3,
    virtual_loss: float = 1.0,
) -> Optional[MctsNode]:
    if not node.children:
        return None

    sqrt_visits = math.sqrt(node.visits + 1)

    c_puct = c_init + math.log((node.visits + c_base + 1.0) / c_base)

    if node.visits > 0:
        parent_q = node.total_values[node.turn] / max(1, node.visits)
    else:
        parent_q = 0.0

    best_score = -float("inf")
    best_child = None

    for child in node.children:
        if child.visits == 0:
            q = parent_q - fpu_reduction
        else:
            q = child.total_values[node.turn] / max(1, child.visits)

        q -= virtual_loss * child.virtual_loss

        u = c_puct * child.prior * sqrt_visits / (1.0 + child.visits)

        score = q + u

        if score > best_score:
            best_score = score
            best_child = child

    if best_child is not None:
        best_child.virtual_loss += virtual_loss

    return best_child


# ============================================================
# Root helpers
# ============================================================

def validate_root(
    root: Optional[MctsNode],
    game,
    legal_idx: List[int],
) -> bool:
    if root is None:
        return False

    if not root.children:
        return False

    if root.turn != game.get_turn():
        return False

    legal_set = set(int(x) for x in legal_idx)
    child_set = set(int(c.move_idx) for c in root.children)

    return legal_set == child_set


def add_root_noise(
    root: MctsNode,
    legal_count: int,
    alpha: Optional[float] = None,
    eps: float = 0.25,
):
    if not root.children:
        return

    if alpha is None:
        alpha = 0.3 if legal_count <= 30 else 0.15

    n = len(root.children)

    noise = np.random.dirichlet([alpha] * n)

    for child, nz in zip(root.children, noise):
        base = child.prior_orig
        child.prior = float((1.0 - eps) * base + eps * float(nz))


def expand_root(
    root: MctsNode,
    game,
    model,
    device,
    legal_idx: List[int],
    alpha: Optional[float] = None,
):
    """
    Раскрывает корень:
    - NN eval;
    - создаёт детей;
    - добавляет Dirichlet noise.
    """
    state = np.zeros((TENSOR_PLANES, 14, 14), dtype=np.float32)
    game.board_to_tensor(state)

    tensor = torch.from_numpy(state).to(device, non_blocking=True).unsqueeze(0)

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        policy_logits, _ = model(tensor)

    policy_logits = policy_logits.float()[0]

    idx_np = np.asarray(legal_idx, dtype=np.int64)
    idx_t = torch.from_numpy(idx_np).to(device, non_blocking=True)

    legal_logits = policy_logits.index_select(0, idx_t)
    probs = torch.softmax(legal_logits, dim=0).cpu().numpy()

    for i, idx in enumerate(legal_idx):
        idx = int(idx)

        turn_after = game.get_turn_after_idx(idx)

        if turn_after is None:
            turn_after = (root.turn + 1) % 4

        child = MctsNode(
            move_idx=idx,
            parent=root,
            prior=float(probs[i]),
            turn=int(turn_after),
        )

        child.prior_orig = float(probs[i])
        root.children.append(child)

    add_root_noise(root, len(legal_idx), alpha=alpha)


# ============================================================
# Sparse policy target
# ============================================================

def make_sparse_policy_target(
    root: MctsNode,
    k: int = POLICY_K,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Возвращает:
      top_idx:  uint16 [K]
      top_prob: float16 [K]
      discarded_mass: float
    """
    top_idx = np.zeros(k, dtype=np.uint16)
    top_prob = np.zeros(k, dtype=np.float16)

    if not root.children:
        return top_idx, top_prob, 1.0

    visits = np.array([c.visits for c in root.children], dtype=np.float32)
    total = float(visits.sum())

    if total <= 0.0:
        return top_idx, top_prob, 1.0

    probs = visits / total

    order = np.argsort(-probs)

    kk = min(k, len(order))
    selected = order[:kk]

    idxs = np.array([root.children[i].move_idx for i in selected], dtype=np.uint16)
    ps = probs[selected].astype(np.float32)

    top_idx[:kk] = idxs

    mass = float(ps.sum())

    if mass > 0.0:
        ps = ps / mass

    top_prob[:kk] = ps.astype(np.float16)

    discarded_mass = max(0.0, 1.0 - mass)

    return top_idx, top_prob, discarded_mass


# ============================================================
# Dynamic MCTS search
# ============================================================

def mcts_search(
    game,
    model,
    device,
    last_root: Optional[MctsNode] = None,
    temperature: float = 1.0,
    cfg: Optional[Dict[str, Any]] = None,
):
    """
    Новый MCTS search.

    Возвращает:
      best_move_idx: int
      top_idx: np.ndarray uint16 [POLICY_K]
      top_prob: np.ndarray float16 [POLICY_K]
      legal_packed: np.ndarray uint8 [MASK_BYTES]
      new_root: MctsNode
      debug: dict
    """
    if cfg is None:
        cfg = {}

    min_sims = int(cfg.get("min_sims", 32))
    max_sims = int(cfg.get("max_sims", 256))
    batch_size = int(cfg.get("batch_size", 16))
    max_time = float(cfg.get("max_time", 0.08))
    max_depth = int(cfg.get("max_depth", 256))
    forced_depth = int(cfg.get("forced_depth", 32))

    c_init = float(cfg.get("c_init", 1.25))
    c_base = float(cfg.get("c_base", 19652.0))
    fpu_reduction = float(cfg.get("fpu_reduction", 0.3))
    virtual_loss = float(cfg.get("virtual_loss", 1.0))

    debug = {
        "simulations": 0,
        "nn_evals": 0,
        "terminal_leaves": 0,
        "forced_leaves": 0,
        "root_reused": 0,
        "root_new": 0,
        "discarded_policy_mass": 0.0,
        "time": 0.0,
        "legal_count": 0,
        "root_visits": 0,
    }

    legal_idx = game.get_legal_move_indices()
    debug["legal_count"] = len(legal_idx)

    if not legal_idx:
        return None

    root_turn = game.get_turn()

    if validate_root(last_root, game, legal_idx):
        root = last_root
        root.parent = None
        root.virtual_loss = 0.0
        add_root_noise(root, len(legal_idx))
        debug["root_reused"] = 1
    else:
        root = MctsNode(turn=root_turn)
        expand_root(root, game, model, device, legal_idx)
        debug["root_new"] = 1

    centers_t = torch.tensor(PLACEMENT_CENTERS, device=device, dtype=torch.float32)

    start_time = time.perf_counter()

    while root.visits < max_sims:
        pending_nodes: List[MctsNode] = []
        pending_games = []
        pending_paths: List[List[MctsNode]] = []
        pending_forced: List[bool] = []

        # ----------------------------------------------------
        # 1. Selection + simulation
        # ----------------------------------------------------
        for _ in range(batch_size):
            sim_game = game.clone_state()
            sim_game.set_deterministic_dead_king(True)

            node = root
            path = [root]
            forced = False

            for _ in range(max_depth):
                if sim_game.is_game_over() or not node.children:
                    break

                if sim_game.is_turn_dead() or sim_game.is_turn_resigned():
                    forced = True

                    # Fast-forward forced moves.
                    for _ in range(forced_depth):
                        if sim_game.is_game_over():
                            break

                        if not (sim_game.is_turn_dead() or sim_game.is_turn_resigned()):
                            break

                        if not sim_game.auto_play_dead_fast():
                            break

                    break

                child = select_child(
                    node,
                    c_init=c_init,
                    c_base=c_base,
                    fpu_reduction=fpu_reduction,
                    virtual_loss=virtual_loss,
                )

                if child is None:
                    break

                if not sim_game.mcts_push_idx(int(child.move_idx)):
                    break

                node = child
                path.append(node)

            pending_nodes.append(node)
            pending_games.append(sim_game)
            pending_paths.append(path)
            pending_forced.append(forced)

        # ----------------------------------------------------
        # 2. Leaf evaluation
        # ----------------------------------------------------
        n = len(pending_games)

        values: List[Optional[np.ndarray]] = [None] * n
        expand_leaf: List[bool] = [False] * n
        logits: List[Optional[torch.Tensor]] = [None] * n

        nn_indices: List[int] = []
        nn_states: List[np.ndarray] = []

        for i, g in enumerate(pending_games):
            if g.is_game_over():
                values[i] = np.asarray(g.get_final_rewards_expected(), dtype=np.float32)
                debug["terminal_leaves"] += 1
                continue

            legal = g.get_legal_move_indices()

            if not legal:
                term = g.terminal_value_if_no_moves()

                if term is None:
                    term = g.get_final_rewards_expected()

                values[i] = np.asarray(term, dtype=np.float32)
                debug["terminal_leaves"] += 1
                continue

            if pending_forced[i]:
                # Forced position: evaluate NN, but do not expand old node.
                debug["forced_leaves"] += 1
                expand_leaf[i] = False
            else:
                expand_leaf[i] = True

            s = np.zeros((TENSOR_PLANES, 14, 14), dtype=np.float32)
            g.board_to_tensor(s)

            nn_states.append(s)
            nn_indices.append(i)

        if nn_states:
            batch_tensor = torch.from_numpy(np.stack(nn_states)).to(
                device,
                non_blocking=True,
            )

            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                p_logits, v_logits = model(batch_tensor)

            p_logits = p_logits.float()
            v_logits = v_logits.float()

            val_probs = torch.softmax(v_logits, dim=2)
            expected_vals = (val_probs * centers_t).sum(dim=2).cpu().numpy()

            debug["nn_evals"] += len(nn_states)

            for b, i in enumerate(nn_indices):
                values[i] = expected_vals[b].astype(np.float32)
                logits[i] = p_logits[b]

        # ----------------------------------------------------
        # 3. Expand normal leaves
        # ----------------------------------------------------
        for i in range(n):
            if not expand_leaf[i]:
                continue

            node = pending_nodes[i]
            g = pending_games[i]

            if node.children:
                continue

            legal = g.get_legal_move_indices()

            if not legal:
                continue

            policy_logits_i = logits[i]

            if policy_logits_i is None:
                continue

            idx_np = np.asarray(legal, dtype=np.int64)
            idx_t = torch.from_numpy(idx_np).to(device, non_blocking=True)

            priors = torch.softmax(
                policy_logits_i.index_select(0, idx_t),
                dim=0,
            ).cpu().numpy()

            for j, idx in enumerate(legal):
                idx = int(idx)

                turn_after = g.get_turn_after_idx(idx)

                if turn_after is None:
                    turn_after = (node.turn + 1) % 4

                child = MctsNode(
                    move_idx=idx,
                    parent=node,
                    prior=float(priors[j]),
                    turn=int(turn_after),
                )

                child.prior_orig = float(priors[j])
                node.children.append(child)

        # ----------------------------------------------------
        # 4. Backup
        # ----------------------------------------------------
        for i, path in enumerate(pending_paths):
            v = values[i]

            if v is None:
                continue

            for node in reversed(path):
                node.visits += 1
                node.virtual_loss = max(0.0, node.virtual_loss - virtual_loss)
                node.total_values += v

        debug["simulations"] += len(pending_paths)

        # ----------------------------------------------------
        # 5. Dynamic stop
        # ----------------------------------------------------
        elapsed = time.perf_counter() - start_time

        if root.visits >= min_sims:
            if elapsed >= max_time:
                break

            if root.children:
                top_visits = max(c.visits for c in root.children)

                if root.visits > 0 and top_visits / float(root.visits) > 0.85:
                    break

    # ========================================================
    # Policy target + move selection
    # ========================================================

    if not root.children:
        return None

    visits = np.array([c.visits for c in root.children], dtype=np.float32)
    total_visits = float(visits.sum())

    if total_visits <= 0.0:
        return None

    probs = visits / total_visits

    if temperature <= 0.01:
        best_local = int(np.argmax(visits))
    else:
        tempered = probs ** (1.0 / float(temperature))
        s = float(tempered.sum())

        if s <= 0.0 or not np.isfinite(s):
            best_local = int(np.argmax(visits))
        else:
            tempered = tempered / s
            best_local = int(np.random.choice(len(root.children), p=tempered))

    best_child = root.children[best_local]

    top_idx, top_prob, discarded_mass = make_sparse_policy_target(root, POLICY_K)
    debug["discarded_policy_mass"] = discarded_mass

    legal_packed = pack_legal_indices([c.move_idx for c in root.children])

    debug["time"] = time.perf_counter() - start_time
    debug["root_visits"] = int(root.visits)

    return (
        int(best_child.move_idx),
        top_idx,
        top_prob,
        legal_packed,
        best_child,
        debug,
    )
# ============================================================
# Default configs
# ============================================================

DEFAULT_MODEL_CFG = {
    "planes": TENSOR_PLANES,
    "blocks": 19,
    "channels": 192,
}

DEFAULT_MCTS_CFG = {
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

DEFAULT_WORKER_CFG = {
    "base_seed": 12345,

    "game_timeout": 900,
    "emergency_ply": 800,

    "adjudication_ply": 350,
    "adjudication_spread": 5,

    "print_every": 10,
    "policy_k": POLICY_K,

    "gpu_memory_fraction": None,

    "model": DEFAULT_MODEL_CFG,
    "mcts": DEFAULT_MCTS_CFG,
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)

    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v

    return out


def make_worker_cfg(override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if override is None:
        override = {}

    return deep_update(DEFAULT_WORKER_CFG, override)


# ============================================================
# Temperature schedule
# ============================================================

def temperature_schedule(ply: int) -> float:
    """
    Температура для выбора хода по visits.

    Важно:
    - температура влияет только на выбор хода;
    - policy target остаётся по реальным visits.
    """
    if ply < 12:
        return 1.2
    elif ply < 30:
        return 1.0
    elif ply < 60:
        return 0.6
    elif ply < 100:
        return 0.3
    else:
        return 0.1


# ============================================================
# Model helpers
# ============================================================

def build_model(
    model_cfg: Dict[str, Any],
    device: torch.device,
) -> ChessNet100:
    model = create_model(
        planes=int(model_cfg.get("planes", TENSOR_PLANES)),
        blocks=int(model_cfg.get("blocks", 19)),
        channels=int(model_cfg.get("channels", 192)),
    )

    model.to(device)
    model.eval()

    return model


def reload_model_weights(
    model: nn.Module,
    model_path: str,
    device: torch.device,
    verbose: bool = False,
) -> bool:
    if not os.path.exists(model_path):
        if verbose:
            print(f"[model] file not found: {model_path}")
        return False

    try:
        sd = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        # Локальный файл, можно позволить fallback.
        sd = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    load_model_weights(model, sd, verbose=verbose)

    return True


# ============================================================
# Replay buffer: sparse policy + packed mask + soft value
# ============================================================

class ReplayBufferSparse:
    """
    RAM-эффективный buffer:

    states:
        float16 [capacity, 100, 14, 14]

    policy:
        top-K indices: uint16 [capacity, K]
        top-K probs:   float16 [capacity, K]

    legal mask:
        packed bits: uint8 [capacity, MASK_BYTES]

    value:
        soft placements: float16 [capacity, 4, 4]
    """

    def __init__(
        self,
        capacity: int,
        policy_k: int = POLICY_K,
        planes: int = TENSOR_PLANES,
    ):
        self.capacity = int(capacity)
        self.policy_k = int(policy_k)
        self.planes = int(planes)

        self.states = np.zeros(
            (self.capacity, self.planes, 14, 14),
            dtype=np.float16,
        )

        self.policy_idx = np.zeros(
            (self.capacity, self.policy_k),
            dtype=np.uint16,
        )

        self.policy_prob = np.zeros(
            (self.capacity, self.policy_k),
            dtype=np.float16,
        )

        self.masks_packed = np.zeros(
            (self.capacity, MASK_BYTES),
            dtype=np.uint8,
        )

        self.values_soft = np.zeros(
            (self.capacity, 4, 4),
            dtype=np.float16,
        )

        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        state: np.ndarray,
        policy_idx: np.ndarray,
        policy_prob: np.ndarray,
        legal_packed: np.ndarray,
        value_soft: np.ndarray,
    ):
        # State
        self.states[self.pos] = np.asarray(state, dtype=np.float16)

        # Policy idx/prob with padding/truncation
        idx_arr = np.asarray(policy_idx, dtype=np.uint16).reshape(-1)
        prob_arr = np.asarray(policy_prob, dtype=np.float16).reshape(-1)

        k = min(self.policy_k, idx_arr.size, prob_arr.size)

        self.policy_idx[self.pos, :] = 0
        self.policy_prob[self.pos, :] = 0

        self.policy_idx[self.pos, :k] = idx_arr[:k]
        self.policy_prob[self.pos, :k] = prob_arr[:k]

        # Legal mask
        mask_arr = np.asarray(legal_packed, dtype=np.uint8).reshape(-1)

        if mask_arr.size != MASK_BYTES:
            raise ValueError(
                f"legal_packed must have length {MASK_BYTES}, got {mask_arr.size}"
            )

        self.masks_packed[self.pos] = mask_arr

        # Value soft [4,4]
        val_arr = np.asarray(value_soft, dtype=np.float16)

        if val_arr.size == 16:
            val_arr = val_arr.reshape(4, 4)
        elif val_arr.shape != (4, 4):
            raise ValueError(
                f"value_soft must be shape (4,4) or flat 16, got {val_arr.shape}"
            )

        self.values_soft[self.pos] = val_arr

        # Advance
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size == 0:
            raise RuntimeError("ReplayBufferSparse is empty")

        batch_size = min(batch_size, self.size)

        indices = np.random.randint(0, self.size, size=batch_size)

        states = self.states[indices].astype(np.float32, copy=False)

        policy_idx = self.policy_idx[indices].astype(np.int64, copy=False)
        policy_prob = self.policy_prob[indices].astype(np.float32, copy=False)

        masks = np.unpackbits(
            self.masks_packed[indices],
            axis=1,
        ).astype(bool)

        # На всякий случай обрезаем до ACTION_SPACE.
        masks = masks[:, :ACTION_SPACE]

        values_soft = self.values_soft[indices].astype(np.float32, copy=False)

        return states, policy_idx, policy_prob, masks, values_soft


# ============================================================
# Augmentation for sparse samples
# ============================================================

def rotate_state_np(state: np.ndarray, k: int) -> np.ndarray:
    k %= 4

    if k == 0:
        return state.copy()

    s = np.rot90(state, k, axes=(1, 2)).copy()
    s = s[_CHANNEL_PERM[k]].copy()

    return s


def rotate_policy_idx_np(
    top_idx: np.ndarray,
    top_prob: np.ndarray,
    k: int,
):
    k %= 4

    if k == 0:
        return top_idx.copy(), top_prob.copy()

    perm = _MOVE_PERM[k]

    new_idx = np.zeros_like(top_idx, dtype=np.uint16)

    valid = top_prob > 0

    if np.any(valid):
        new_idx[valid] = perm[top_idx[valid]].astype(np.uint16)

    return new_idx, top_prob.copy()


def rotate_mask_bool_np(mask: np.ndarray, k: int) -> np.ndarray:
    k %= 4

    if k == 0:
        return mask.copy()

    return mask[_MOVE_INV_PERM[k]].copy()


def rotate_soft_value_np(value_soft: np.ndarray, k: int) -> np.ndarray:
    k %= 4

    if k == 0:
        return value_soft.copy()

    # value_soft: [player, place]
    return np.roll(value_soft, k, axis=0).copy()


def rotate_sparse_sample_np(
    state: np.ndarray,
    top_idx: np.ndarray,
    top_prob: np.ndarray,
    mask_bool: np.ndarray,
    value_soft: np.ndarray,
    k: int,
):
    k %= 4

    s = rotate_state_np(state, k)
    idx, prob = rotate_policy_idx_np(top_idx, top_prob, k)
    m = rotate_mask_bool_np(mask_bool, k)
    v = rotate_soft_value_np(value_soft, k)

    return s, idx, prob, m, v


def augment_sparse_batch(
    states: np.ndarray,
    policy_idx: np.ndarray,
    policy_prob: np.ndarray,
    masks: np.ndarray,
    values_soft: np.ndarray,
):
    """
    states:       float32 [B, 100, 14, 14]
    policy_idx:   int64   [B, K]
    policy_prob:  float32 [B, K]
    masks:        bool    [B, ACTION_SPACE]
    values_soft:  float32 [B, 4, 4]

    Возвращает torch tensors.
    """
    B = states.shape[0]

    out_s = np.empty_like(states)
    out_idx = np.empty_like(policy_idx)
    out_prob = np.empty_like(policy_prob)
    out_mask = np.empty_like(masks)
    out_val = np.empty_like(values_soft)

    for i in range(B):
        k = random.randint(0, 3)

        s, idx, prob, m, v = rotate_sparse_sample_np(
            states[i],
            policy_idx[i],
            policy_prob[i],
            masks[i],
            values_soft[i],
            k,
        )

        out_s[i] = s
        out_idx[i] = idx
        out_prob[i] = prob
        out_mask[i] = m
        out_val[i] = v

    return (
        torch.from_numpy(out_s),
        torch.from_numpy(out_idx),
        torch.from_numpy(out_prob),
        torch.from_numpy(out_mask),
        torch.from_numpy(out_val),
    )


# ============================================================
# Loss / metrics helpers
# ============================================================

def sparse_policy_loss(
    pred_logits: torch.Tensor,
    policy_idx: torch.Tensor,
    policy_prob: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """
    pred_logits: [B, ACTION_SPACE]
    policy_idx:  [B, K]
    policy_prob: [B, K]
    legal_mask:  [B, ACTION_SPACE]
    """
    pred = pred_logits.float()
    pred = pred.masked_fill(~legal_mask, -1e9)

    logp = F.log_softmax(pred, dim=1)

    gathered = logp.gather(1, policy_idx.long())

    p = policy_prob.float()
    p_sum = p.sum(dim=1, keepdim=True).clamp_min(1e-6)
    p = p / p_sum

    loss = -(p * gathered).sum(dim=1).mean()

    return loss


def soft_value_loss(
    pred_v: torch.Tensor,
    soft_v: torch.Tensor,
    mse_weight: float = 0.1,
):
    """
    pred_v:   [B, 4, 4]
    soft_v:   [B, 4, 4]

    Возвращает:
      total_loss, ce_loss, mse_loss
    """
    pred = pred_v.float()
    target = soft_v.float()

    log_probs = F.log_softmax(pred, dim=-1)

    loss_ce = -(target * log_probs).sum(dim=-1).mean()

    if mse_weight <= 0.0:
        zero = torch.tensor(0.0, device=pred.device)
        return loss_ce, loss_ce, zero

    centers = torch.tensor(
        PLACEMENT_CENTERS,
        device=pred.device,
        dtype=torch.float32,
    )

    pred_expected = (torch.softmax(pred, dim=-1) * centers).sum(dim=-1)
    target_expected = (target * centers).sum(dim=-1)

    loss_mse = F.mse_loss(pred_expected, target_expected)

    total = loss_ce + mse_weight * loss_mse

    return total, loss_ce, loss_mse


def policy_metrics(
    pred_logits: torch.Tensor,
    policy_idx: torch.Tensor,
    policy_prob: torch.Tensor,
    legal_mask: torch.Tensor,
):
    """
    Возвращает:
      top1_acc, top3_acc, entropy
    """
    pred = pred_logits.float()
    pred = pred.masked_fill(~legal_mask, -1e9)

    # Target top move
    target_pos = policy_prob.float().argmax(dim=1)
    target_top = policy_idx.gather(1, target_pos.unsqueeze(1)).squeeze(1)

    # Net top1
    net_top = pred.argmax(dim=1)

    top1 = (net_top == target_top).float().mean()

    # Net top3
    _, net_top3 = pred.topk(3, dim=1)
    top3 = (net_top3 == target_top.unsqueeze(1)).any(dim=1).float().mean()

    # Entropy
    probs = F.softmax(pred, dim=1)
    log_probs = F.log_softmax(pred, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean()

    return top1.item(), top3.item(), entropy.item()


# ============================================================
# Debug helpers
# ============================================================

def run_engine_debug_tests():
    """
    Быстрая проверка нового engine API.
    """
    game = chess4p_ai.PyGameState()

    assert chess4p_ai.TENSOR_PLANES == 100
    assert chess4p_ai.ACTION_SPACE == 38416

    tensor = np.zeros((TENSOR_PLANES, 14, 14), dtype=np.float32)
    game.board_to_tensor(tensor)

    legal = game.get_legal_move_indices()

    assert len(legal) > 0
    assert len(set(legal)) == len(legal)

    for idx in legal:
        assert 0 <= int(idx) < ACTION_SPACE

    old_turn = game.get_turn()
    old_hash = game.get_zobrist_hash()
    old_scores = game.get_scores()
    old_stack = game.mcts_stack_len()

    assert game.mcts_push_idx(int(legal[0]))
    assert game.mcts_stack_len() == old_stack + 1

    assert game.mcts_pop()
    assert game.mcts_stack_len() == old_stack

    assert game.get_turn() == old_turn
    assert game.get_zobrist_hash() == old_hash
    assert game.get_scores() == old_scores

    print("[DEBUG] engine debug tests: OK", flush=True)


def run_rotation_debug_tests():
    """
    Минимальная проверка rotation tables.
    """
    assert _MOVE_PERM is not None
    assert _MOVE_INV_PERM is not None
    assert _CHANNEL_PERM is not None

    assert len(_MOVE_PERM) == 4
    assert len(_MOVE_INV_PERM) == 4
    assert len(_CHANNEL_PERM) == 4

    for k in range(4):
        assert _MOVE_PERM[k].shape[0] == ACTION_SPACE
        assert _MOVE_INV_PERM[k].shape[0] == ACTION_SPACE
        assert _CHANNEL_PERM[k].shape[0] == TENSOR_PLANES

    # k=0 должен быть identity.
    assert np.array_equal(_MOVE_PERM[0], np.arange(ACTION_SPACE))
    assert np.array_equal(_MOVE_INV_PERM[0], np.arange(ACTION_SPACE))
    assert np.array_equal(_CHANNEL_PERM[0], np.arange(TENSOR_PLANES))

    print("[DEBUG] rotation debug tests: OK", flush=True)


if DEBUG_ENGINE:
    try:
        run_engine_debug_tests()
        run_rotation_debug_tests()
    except Exception as e:
        print(f"[DEBUG] startup debug failed: {e}", flush=True)


# ============================================================
# Self-play worker
# ============================================================

def self_play_worker(
    worker_id: int,
    gpu_id: int,
    model_path: str,
    data_queue,
    num_games: int,
    cfg: Optional[Dict[str, Any]] = None,
):
    cfg = make_worker_cfg(cfg)

    set_worker_seeds(worker_id, base_seed=int(cfg.get("base_seed", 12345)))

    # Ограничиваем CPU-потоки, чтобы воркеры не душили друг друга.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    mem_fraction = cfg.get("gpu_memory_fraction", None)

    if mem_fraction is not None:
        try:
            torch.cuda.set_per_process_memory_fraction(
                float(mem_fraction),
                device,
            )
        except Exception as e:
            print(
                f"W{worker_id}: failed to set memory fraction: {e}",
                flush=True,
            )

    model_cfg = cfg["model"]
    mcts_cfg = cfg["mcts"]

    model = build_model(model_cfg, device)

    last_mod_time = 0.0

    games_played = 0
    positions_total = 0

    total_sims = 0
    total_time = 0.0
    total_nn_evals = 0
    total_terminal = 0
    total_forced = 0
    total_discarded = 0.0

    timeouts = 0
    exceptions = 0
    adjudicated_draw = 0
    emergency_stop = 0
    model_reloads = 0

    print_every = int(cfg.get("print_every", 10))

    def timeout_handler(signum, frame):
        raise TimeoutError("self-play game timeout")

    signal.signal(signal.SIGALRM, timeout_handler)

    while games_played < num_games:
        # ====================================================
        # Reload model if updated
        # ====================================================
        try:
            mod_time = os.path.getmtime(model_path)

            if mod_time > last_mod_time:
                ok = reload_model_weights(
                    model,
                    model_path,
                    device,
                    verbose=False,
                )

                if ok:
                    model.eval()
                    last_mod_time = mod_time
                    model_reloads += 1

                    if DEBUG_WORKER:
                        print(
                            f"W{worker_id}: model reloaded, mtime={mod_time}",
                            flush=True,
                        )
        except Exception as e:
            if DEBUG_WORKER:
                print(
                    f"W{worker_id}: model reload error: {e}",
                    flush=True,
                )

        # ====================================================
        # New game
        # ====================================================
        game = chess4p_ai.PyGameState()

        seed = (
            int(cfg.get("base_seed", 12345))
            + worker_id * 1_000_003
            + games_played * 9973
        ) % (2**63)

        game.set_seed(seed)
        game.set_deterministic_dead_king(False)

        states_data: List[np.ndarray] = []
        policy_idx_data: List[np.ndarray] = []
        policy_prob_data: List[np.ndarray] = []
        mask_data: List[np.ndarray] = []

        last_root = None

        game_ok = True

        g_positions = 0
        g_sims = 0
        g_time = 0.0
        g_nn_evals = 0
        g_terminal = 0
        g_forced = 0
        g_discarded = 0.0

        signal.alarm(int(cfg.get("game_timeout", 900)))

        try:
            while not game.is_game_over():
                ply = game.get_ply_count()

                # Emergency limit.
                if ply >= int(cfg.get("emergency_ply", 800)):
                    emergency_stop += 1
                    break

                # Training adjudication.
                if game.is_training_adjudication_draw(
                    int(cfg.get("adjudication_ply", 350)),
                    int(cfg.get("adjudication_spread", 5)),
                ):
                    adjudicated_draw += 1
                    break

                # Dead / resigned turns.
                if game.is_turn_dead() or game.is_turn_resigned():
                    if not game.auto_play_dead_fast():
                        break

                    last_root = None
                    continue

                temp = temperature_schedule(ply)

                result = mcts_search(
                    game,
                    model,
                    device,
                    last_root=last_root,
                    temperature=temp,
                    cfg=mcts_cfg,
                )

                if result is None:
                    break

                (
                    best_idx,
                    top_idx,
                    top_prob,
                    legal_packed,
                    new_root,
                    dbg,
                ) = result

                # State before move.
                state = np.zeros((TENSOR_PLANES, 14, 14), dtype=np.float32)
                game.board_to_tensor(state)

                states_data.append(state.astype(np.float16))
                policy_idx_data.append(top_idx)
                policy_prob_data.append(top_prob)
                mask_data.append(legal_packed)

                g_positions += 1

                g_sims += int(dbg.get("simulations", 0))
                g_time += float(dbg.get("time", 0.0))
                g_nn_evals += int(dbg.get("nn_evals", 0))
                g_terminal += int(dbg.get("terminal_leaves", 0))
                g_forced += int(dbg.get("forced_leaves", 0))
                g_discarded += float(dbg.get("discarded_policy_mass", 0.0))

                if not game.make_move_fast_idx(int(best_idx)):
                    break

                last_root = new_root

        except TimeoutError:
            timeouts += 1
            game_ok = False

            if DEBUG_WORKER:
                print(
                    f"W{worker_id}: game timeout, discard game",
                    flush=True,
                )

        except Exception as e:
            exceptions += 1
            game_ok = False

            if DEBUG_WORKER:
                import traceback

                print(
                    f"W{worker_id}: exception in game: {e}",
                    flush=True,
                )

                traceback.print_exc()

        finally:
            signal.alarm(0)

        # ====================================================
        # Push game data
        # ====================================================
        if game_ok and len(states_data) > 0:
            try:
                soft_flat = game.get_final_placement_soft()
                soft = np.asarray(soft_flat, dtype=np.float32).reshape(4, 4)
                soft = soft.astype(np.float16)

                for i in range(len(states_data)):
                    data_queue.put(
                        (
                            states_data[i],
                            policy_idx_data[i],
                            policy_prob_data[i],
                            mask_data[i],
                            soft,
                        )
                    )

                positions_total += len(states_data)

                total_sims += g_sims
                total_time += g_time
                total_nn_evals += g_nn_evals
                total_terminal += g_terminal
                total_forced += g_forced
                total_discarded += g_discarded

            except Exception as e:
                exceptions += 1

                if DEBUG_WORKER:
                    print(
                        f"W{worker_id}: queue push error: {e}",
                        flush=True,
                    )

        games_played += 1

        # ====================================================
        # Logging
        # ====================================================
        if games_played % print_every == 0:
            avg_positions = positions_total / max(1, games_played)

            avg_sims = total_sims / max(1, positions_total)
            avg_time = total_time / max(1, positions_total)
            avg_nn = total_nn_evals / max(1, positions_total)
            avg_terminal = total_terminal / max(1, positions_total)
            avg_forced = total_forced / max(1, positions_total)
            avg_discarded = total_discarded / max(1, positions_total)

            print(
                f"W{worker_id} GPU{gpu_id} | "
                f"games={games_played} | "
                f"pos/game={avg_positions:.1f} | "
                f"sims={avg_sims:.1f} | "
                f"mcts_time={avg_time:.4f}s | "
                f"nn={avg_nn:.1f} | "
                f"term={avg_terminal:.1f} | "
                f"forced={avg_forced:.1f} | "
                f"discarded={avg_discarded:.3f} | "
                f"timeout={timeouts} | "
                f"exc={exceptions} | "
                f"adj={adjudicated_draw} | "
                f"emrg={emergency_stop} | "
                f"reloads={model_reloads}",
                flush=True,
            )

        if DEBUG_WORKER and g_positions > 0:
            print(
                f"W{worker_id} game {games_played} | "
                f"pos={g_positions} | "
                f"sims={g_sims} | "
                f"time={g_time:.3f}s | "
                f"nn={g_nn_evals} | "
                f"term={g_terminal} | "
                f"forced={g_forced} | "
                f"discarded={g_discarded:.3f}",
                flush=True,
            )