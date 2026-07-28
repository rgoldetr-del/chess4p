"""
AlphaZero-обучение для 4-Player Chess FFA.
Исправлены: race conditions, device override, queue draining,
torch.compile, assert-защиты, константы плоскостей.
"""

import chess4p_ai
import os
import sys
import time
import math
import random
import hashlib
import queue as queue_module
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as tmp

torch.backends.cudnn.benchmark = True

# ============================================================
# Константы плоскостей тензора (единый источник правды)
# ============================================================
PLANE_PIECES_START = 0        # 0-23:  4 цвета × 6 типов
PLANE_PROMOTED_START = 24     # 24-27
PLANE_EP_START = 28           # 28-31
PLANE_TURN_START = 32         # 32-35
PLANE_CASTLING_START = 36     # 36-43: 4 цвета × 2 (short/long)
PLANE_DEAD_START = 44         # 44-47
PLANE_RESIGNED_START = 48     # 48-51
PLANE_SCORES_START = 52       # 52-55
PLANE_HALFMOVE = 56           # 56 (скаляр — НЕ вращается)
PLANE_CHECK_START = 57        # 57-60
PLANE_PAWN_DIST_START = 61    # 61-64
PLANE_GRAY_START = 65         # 65-68
PLANE_HANGING_START = 69      # 69-72
NUM_PLANES = 73
BOARD_SIZE = 14
POLICY_SIZE = 14 * 14 * 14 * 14  # 38416
PACKED_MASK_SIZE = (POLICY_SIZE + 7) // 8  # 4802

# ============================================================
# Кэш placement centers
# ============================================================
PLACEMENT_CENTERS_CACHE = {}


def get_placement_centers(device):
    if device not in PLACEMENT_CENTERS_CACHE:
        PLACEMENT_CENTERS_CACHE[device] = torch.tensor(
            [1.0, 0.33, -0.33, -1.0],
            device=device,
            dtype=torch.float32,
        )
    return PLACEMENT_CENTERS_CACHE[device]


# ============================================================
# Нейросеть
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = torch.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_init = nn.Sequential(
            nn.Conv2d(NUM_PLANES, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(192) for _ in range(19)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.policy_from = nn.Conv2d(64, 196, kernel_size=1)
        self.policy_to = nn.Conv2d(64, 196, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(32 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4 * 4),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = self.res_blocks(x)
        p = self.policy_head(x)
        p_from = self.policy_from(p).flatten(2)
        p_to = self.policy_to(p).flatten(2)
        policy_logits = torch.bmm(p_to.transpose(1, 2), p_from).flatten(1)
        value = self.value_head(x)
        value = value.flatten(1)
        value = self.value_fc(value)
        value = value.view(-1, 4, 4)
        return policy_logits, value


# ============================================================
# Утилиты
# ============================================================
def move_to_idx(fx, fy, tx, ty):
    return (fx - 1) + (fy - 1) * 14 + (tx - 1) * 196 + (ty - 1) * 2744


def rewards_to_placements(rewards):
    sorted_indices = np.argsort(-rewards)
    placements = np.zeros(4, dtype=np.int64)
    for place, player_idx in enumerate(sorted_indices):
        placements[player_idx] = place
    return placements


def rotate_index(idx, k):
    k %= 4
    if k == 0:
        return int(idx)
    fx = (idx % 14) + 1
    fy = ((idx // 14) % 14) + 1
    tx = ((idx // 196) % 14) + 1
    ty = ((idx // 2744) % 14) + 1
    for _ in range(k):
        nfx = fy
        nfy = 15 - fx
        ntx = ty
        nty = 15 - tx
        fx, fy, tx, ty = nfx, nfy, ntx, nty
    return (fx - 1) + (fy - 1) * 14 + (tx - 1) * 196 + (ty - 1) * 2744


def rotate_sample(s, p, v, m, k):
    """Вращает образец на k×90° по часовой. Плоскость 56 (halfmove) — скаляр, не вращается."""
    k %= 4
    if k == 0:
        return s.copy(), p.copy(), v.copy(), m.copy()

    s_new = np.rot90(s, k, axes=(1, 2)).copy()

    # Вращаем цветовые плоскости (roll по оси 0)
    s_new[PLANE_PIECES_START:PLANE_PIECES_START + 24] = np.roll(
        s_new[PLANE_PIECES_START:PLANE_PIECES_START + 24], k * 6, axis=0
    )
    s_new[PLANE_PROMOTED_START:PLANE_PROMOTED_START + 4] = np.roll(
        s_new[PLANE_PROMOTED_START:PLANE_PROMOTED_START + 4], k, axis=0
    )
    s_new[PLANE_EP_START:PLANE_EP_START + 4] = np.roll(
        s_new[PLANE_EP_START:PLANE_EP_START + 4], k, axis=0
    )
    s_new[PLANE_TURN_START:PLANE_TURN_START + 4] = np.roll(
        s_new[PLANE_TURN_START:PLANE_TURN_START + 4], k, axis=0
    )
    s_new[PLANE_CASTLING_START:PLANE_CASTLING_START + 8] = np.roll(
        s_new[PLANE_CASTLING_START:PLANE_CASTLING_START + 8], k * 2, axis=0
    )
    s_new[PLANE_DEAD_START:PLANE_DEAD_START + 4] = np.roll(
        s_new[PLANE_DEAD_START:PLANE_DEAD_START + 4], k, axis=0
    )
    s_new[PLANE_RESIGNED_START:PLANE_RESIGNED_START + 4] = np.roll(
        s_new[PLANE_RESIGNED_START:PLANE_RESIGNED_START + 4], k, axis=0
    )
    s_new[PLANE_SCORES_START:PLANE_SCORES_START + 4] = np.roll(
        s_new[PLANE_SCORES_START:PLANE_SCORES_START + 4], k, axis=0
    )
    # PLANE_HALFMOVE (56) — НЕ вращается (скаляр)
    s_new[PLANE_CHECK_START:PLANE_CHECK_START + 4] = np.roll(
        s_new[PLANE_CHECK_START:PLANE_CHECK_START + 4], k, axis=0
    )
    s_new[PLANE_PAWN_DIST_START:PLANE_PAWN_DIST_START + 4] = np.roll(
        s_new[PLANE_PAWN_DIST_START:PLANE_PAWN_DIST_START + 4], k, axis=0
    )
    s_new[PLANE_GRAY_START:PLANE_GRAY_START + 4] = np.roll(
        s_new[PLANE_GRAY_START:PLANE_GRAY_START + 4], k, axis=0
    )
    s_new[PLANE_HANGING_START:PLANE_HANGING_START + 4] = np.roll(
        s_new[PLANE_HANGING_START:PLANE_HANGING_START + 4], k, axis=0
    )

    p_new = np.zeros_like(p)
    m_new = np.zeros_like(m)
    v_new = np.roll(v, k).copy()

    for idx in np.nonzero(m)[0]:
        m_new[rotate_index(int(idx), k)] = True
    for idx in np.nonzero(p)[0]:
        p_new[rotate_index(int(idx), k)] = p[idx]

    return s_new, p_new, v_new, m_new


# ============================================================
# MCTS
# ============================================================
class MctsNode:
    __slots__ = (
        "move", "parent", "children", "visits",
        "total_values", "prior", "turn", "virtual_loss",
    )

    def __init__(self, move=None, parent=None, prior=0.0, turn=0):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_values = np.zeros(4, dtype=np.float32)
        self.prior = float(prior)
        self.turn = int(turn)
        self.virtual_loss = 0.0


def select_child(node, c_puct=1.5, virtual_loss=1.0):
    best_score = -1e9
    best_child = None
    sqrt_visits = math.sqrt(node.visits + 1)
    for child in node.children:
        q = child.total_values[node.turn] / max(1, child.visits)
        q -= virtual_loss * child.virtual_loss
        u = c_puct * child.prior * sqrt_visits / (1 + child.visits)
        score = q + u
        if score > best_score:
            best_score = score
            best_child = child
    if best_child is not None:
        best_child.virtual_loss += virtual_loss
    return best_child


def expand_root_children(root, root_game, model, device):
    legal_moves_flat = root_game.get_legal_moves()
    if not legal_moves_flat:
        return []
    legal_moves = [
        (legal_moves_flat[i], legal_moves_flat[i + 1],
         legal_moves_flat[i + 2], legal_moves_flat[i + 3])
        for i in range(0, len(legal_moves_flat), 4)
    ]

    state_np = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    root_game.board_to_tensor(state_np)
    state_tensor = (
        torch.from_numpy(state_np)
        .to(device, non_blocking=True)
        .view(1, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    )

    with torch.no_grad():
        policy_logits, _ = model(state_tensor)

    indices = torch.tensor(
        [move_to_idx(*m) for m in legal_moves],
        device=device, dtype=torch.long,
    )
    legal_logits = policy_logits[0].index_select(0, indices)
    probs = torch.softmax(legal_logits, dim=0).cpu().numpy()

    num_moves = len(legal_moves)
    alpha = 0.15
    noise = np.random.dirichlet([alpha] * num_moves)

    for i, m in enumerate(legal_moves):
        prior = 0.75 * probs[i] + 0.25 * noise[i]
        temp_game = root_game.clone_state()
        temp_game.make_move_unchecked(*m)
        next_turn = temp_game.get_turn()
        root.children.append(
            MctsNode(move=m, parent=root, prior=prior, turn=next_turn)
        )
    return legal_moves


def mcts_search(root_game, model, device, simulations=100, batch_size=16, last_root=None):
    root_turn = root_game.get_turn()

    if last_root is not None and last_root.children:
        root = last_root
        root.parent = None
        root.virtual_loss = 0.0
        root.turn = root_turn
    else:
        root = MctsNode(turn=root_turn)

    if not root.children:
        legal_moves = expand_root_children(root, root_game, model, device)
        if not legal_moves:
            return None, None, None

    placement_centers = get_placement_centers(device)
    num_batches = (simulations + batch_size - 1) // batch_size

    for _ in range(num_batches):
        pending_leaves = []
        pending_games = []
        pending_paths = []

        # Selection
        for _ in range(batch_size):
            node = root
            sim_game = root_game.clone_state()
            path = [node]
            for _ in range(512):
                if sim_game.is_game_over():
                    break
                if sim_game.is_turn_resigned() or sim_game.is_turn_dead():
                    if not sim_game.auto_play_dead():
                        break
                    continue
                if not node.children:
                    break
                node = select_child(node)
                sim_game.make_move_unchecked(*node.move)
                path.append(node)
            pending_leaves.append(node)
            pending_games.append(sim_game)
            pending_paths.append(path)

        # Batch inference
        non_terminal_indices = []
        batch_states = []
        for i, g in enumerate(pending_games):
            if g.is_game_over():
                continue
            moves_flat = g.get_legal_moves()
            if not moves_flat:
                continue
            non_terminal_indices.append(i)
            s = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            g.board_to_tensor(s)
            batch_states.append(s)

        values = [None] * len(pending_games)
        nn_policies = [None] * len(pending_games)

        if batch_states:
            batch_tensor = torch.from_numpy(np.stack(batch_states)).to(device, non_blocking=True)
            with torch.no_grad():
                p_logits, val = model(batch_tensor)
                val_probs = torch.softmax(val, dim=2)
                expected_vals = (val_probs * placement_centers).sum(dim=2)
                expected_vals_np = expected_vals.cpu().numpy()
            for batch_idx, i in enumerate(non_terminal_indices):
                values[i] = expected_vals_np[batch_idx]
                nn_policies[i] = p_logits[batch_idx]

        # Expansion
        expanded_ids = set()
        for i, g in enumerate(pending_games):
            if values[i] is None:
                values[i] = np.array(g.get_final_rewards(), dtype=np.float32)
                continue
            node = pending_leaves[i]
            if node.children:
                continue
            node_id = id(node)
            if node_id in expanded_ids:
                continue
            expanded_ids.add(node_id)

            sim_moves_flat = g.get_legal_moves()
            if not sim_moves_flat:
                continue
            sim_moves = [
                (sim_moves_flat[j], sim_moves_flat[j + 1],
                 sim_moves_flat[j + 2], sim_moves_flat[j + 3])
                for j in range(0, len(sim_moves_flat), 4)
            ]
            policy_logits_i = nn_policies[i]
            if policy_logits_i is None:
                continue
            sim_indices = torch.tensor(
                [move_to_idx(*m) for m in sim_moves],
                device=device, dtype=torch.long,
            )
            sim_legal_logits = policy_logits_i.index_select(0, sim_indices)
            priors = torch.softmax(sim_legal_logits, dim=0).cpu().numpy()

            for j, m in enumerate(sim_moves):
                temp_game = g.clone_state()
                temp_game.make_move_unchecked(*m)
                next_turn = temp_game.get_turn()
                node.children.append(
                    MctsNode(move=m, parent=node, prior=float(priors[j]), turn=next_turn)
                )

        # Backpropagation
        for i, path in enumerate(pending_paths):
            v = values[i]
            if v is None:
                continue
            for node in reversed(path):
                node.visits += 1
                node.virtual_loss = max(0.0, node.virtual_loss - 1.0)
                node.total_values += v

    # Aggregate
    visits_by_move = {}
    for child in root.children:
        visits_by_move[child.move] = visits_by_move.get(child.move, 0) + child.visits
    if not visits_by_move:
        return None, None, None

    total_visits = float(sum(visits_by_move.values()))
    policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
    if total_visits > 0.0:
        for m, visits in visits_by_move.items():
            policy_target[move_to_idx(*m)] = visits / total_visits
    else:
        uniform = 1.0 / float(len(visits_by_move))
        for m in visits_by_move.keys():
            policy_target[move_to_idx(*m)] = uniform

    moves = list(visits_by_move.keys())
    visits = np.array([visits_by_move[m] for m in moves], dtype=np.float32)

    if visits.sum() <= 0:
        best_idx = 0
    elif root_game.get_move_count() < 30:
        probs = visits / visits.sum()
        best_idx = np.random.choice(len(moves), p=probs)
    else:
        best_idx = int(np.argmax(visits))

    best_move = moves[best_idx]
    new_root = None
    best_child_visits = -1
    for child in root.children:
        if child.move == best_move and child.visits > best_child_visits:
            best_child_visits = child.visits
            new_root = child
    if new_root is None:
        new_root = root.children[0]

    return best_move, policy_target, new_root


# ============================================================
# Безопасная загрузка модели (защита от race condition)
# ============================================================
def safe_load_model(model_path, model, device, last_hash):
    """Загружает модель только если файл изменился (по MD5-хешу)."""
    try:
        with open(model_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash != last_hash:
            state_dict = torch.load(
                model_path, map_location=device, weights_only=True
            )
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict)
            model.eval()
            return file_hash
    except (FileNotFoundError, RuntimeError, OSError):
        pass
    return last_hash


# ============================================================
# Self-play worker
# ============================================================
def self_play_worker(worker_id, model_path, device, data_queue, num_games):
    # ИСПРАВЛЕНО: НЕ перезаписываем device — используем переданный
    model = ChessNet().to(device)
    last_hash = ""
    games_played = 0

    while games_played < num_games:
        last_hash = safe_load_model(model_path, model, device, last_hash)

        game = chess4p_ai.PyGameState()
        move_count = 0
        states_data = []
        policies_data = []
        masks_data = []
        last_root = None

        while not game.is_game_over() and move_count < 200:
            move_count += 1
            if game.is_turn_dead() or game.is_turn_resigned():
                if not game.auto_play_dead():
                    break
                last_root = None
                continue
            if game.is_game_over():
                break

            best_move, policy_target, last_root = mcts_search(
                game, model, device,
                simulations=100, batch_size=16, last_root=last_root,
            )
            if not best_move:
                break

            state_np = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            game.board_to_tensor(state_np)

            mask_np = np.zeros(POLICY_SIZE, dtype=np.bool_)
            legal_moves_flat = game.get_legal_moves()
            for i in range(0, len(legal_moves_flat), 4):
                fx, fy = legal_moves_flat[i], legal_moves_flat[i + 1]
                tx, ty = legal_moves_flat[i + 2], legal_moves_flat[i + 3]
                mask_np[move_to_idx(fx, fy, tx, ty)] = True

            states_data.append(state_np)
            policies_data.append(policy_target)
            masks_data.append(mask_np)
            game.make_move(*best_move)

        if len(states_data) > 0:
            final_rewards = np.array(game.get_final_rewards(), dtype=np.float32)
            v_placements = rewards_to_placements(final_rewards)
            for i in range(len(states_data)):
                s_tensor = torch.from_numpy(states_data[i]).share_memory_()
                p_tensor = torch.from_numpy(policies_data[i]).share_memory_()
                v_tensor = torch.from_numpy(v_placements).share_memory_()
                m_tensor = torch.from_numpy(masks_data[i]).share_memory_()
                data_queue.put((s_tensor, p_tensor, v_tensor, m_tensor))

        games_played += 1
        if games_played % 10 == 0:
            print(f"Worker {worker_id}: Сыграно {games_played}/{num_games} партий.")


# ============================================================
# Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = torch.zeros(
            (capacity, NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32
        )
        self.policies = torch.zeros((capacity, POLICY_SIZE), dtype=torch.float32)
        self.values = torch.zeros((capacity, 4), dtype=torch.int64)
        self.masks_packed = np.zeros((capacity, PACKED_MASK_SIZE), dtype=np.uint8)
        self.pos = 0
        self.size = 0

    def push(self, state, policy, value_bins, mask):
        assert mask.shape[0] == POLICY_SIZE, (
            f"Mask size {mask.shape[0]} != {POLICY_SIZE}"
        )
        self.states[self.pos] = torch.as_tensor(state, dtype=torch.float32)
        self.policies[self.pos] = torch.as_tensor(policy, dtype=torch.float32)
        self.values[self.pos] = torch.as_tensor(value_bins, dtype=torch.int64)
        self.masks_packed[self.pos] = np.packbits(mask)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,)).numpy()
        states = self.states[torch.from_numpy(indices)]
        policies = self.policies[torch.from_numpy(indices)]
        values = self.values[torch.from_numpy(indices)]
        masks_packed = self.masks_packed[indices]
        masks = np.unpackbits(masks_packed, axis=1)[:, :POLICY_SIZE].astype(bool)
        masks = torch.from_numpy(masks)
        return states, policies, values, masks

    def __len__(self):
        return self.size


# ============================================================
# Безопасное чтение из очереди (ИСПРАВЛЕНИЕ race condition)
# ============================================================
def drain_queue(data_queue, replay_buffer, max_items=500):
    """Вычитывает до max_items элементов без race condition."""
    count = 0
    while count < max_items:
        try:
            s, p, v, m = data_queue.get_nowait()
            replay_buffer.push(s.numpy(), p.numpy(), v.numpy(), m.numpy())
            count += 1
        except queue_module.Empty:
            break
    return count


# ============================================================
# Главный цикл обучения
# ============================================================
def alpha_zero_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство обучения: {device}")

    model = ChessNet().to(device)
    if os.path.exists("model.pt"):
        state_dict = torch.load("model.pt", map_location=device, weights_only=True)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("Модель загружена.")
    else:
        torch.save(model.state_dict(), "model.pt")
        print("Создана новая модель.")

    # torch.compile для ускорения (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile активирован.")
        except Exception as e:
            print(f"torch.compile недоступен: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    total_iters = 100000
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 - min(step, total_iters) / total_iters,
    )

    replay_buffer = ReplayBuffer(capacity=10000)
    batch_size = 64
    iteration = 1

    tmp.set_start_method("spawn", force=True)
    data_queue = tmp.Queue(maxsize=10000)
    num_workers = 2
    games_per_worker = 1_000_000
    workers = []
    for i in range(num_workers):
        p = tmp.Process(
            target=self_play_worker,
            args=(i, "model.pt", device, data_queue, games_per_worker),
        )
        p.start()
        workers.append(p)

    try:
        while True:
            # ИСПРАВЛЕНО: безопасное чтение из очереди
            drained = drain_queue(data_queue, replay_buffer)
            if drained > 0:
                print(
                    f"Итерация {iteration} | Буфер: {len(replay_buffer)} | "
                    f"Добавлено: {drained}",
                    end="\r",
                )

            if len(replay_buffer) >= batch_size:
                model.train()
                s, p, v, m = replay_buffer.sample(batch_size)

                # Аугментация вращения
                s_aug, p_aug, v_aug, m_aug = [], [], [], []
                for i in range(batch_size):
                    k = random.randint(0, 3)
                    s_n, p_n, v_n, m_n = rotate_sample(
                        s[i].numpy(), p[i].numpy(), v[i].numpy(), m[i].numpy(), k,
                    )
                    s_aug.append(torch.from_numpy(s_n.copy()))
                    p_aug.append(torch.from_numpy(p_n.copy()))
                    v_aug.append(torch.from_numpy(v_n.copy()))
                    m_aug.append(torch.from_numpy(m_n.copy()))

                s = torch.stack(s_aug).to(device, non_blocking=True)
                p = torch.stack(p_aug).to(device, non_blocking=True)
                v = torch.stack(v_aug).to(device, non_blocking=True)
                m = torch.stack(m_aug).to(device, non_blocking=True)

                optimizer.zero_grad()
                pred_p, pred_v = model(s)
                pred_p = pred_p.masked_fill(~m, -1e9)

                loss_p = torch.mean(
                    -torch.sum(p * torch.log_softmax(pred_p, dim=1), dim=1)
                )
                loss_v = 0.0
                for i in range(4):
                    loss_v += F.cross_entropy(pred_v[:, i, :], v[:, i])
                loss_v /= 4.0

                loss = loss_p + loss_v
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.eval()

                if iteration % 50 == 0:
                    torch.save(model.state_dict(), "model_tmp.pt")
                    for attempt in range(10):
                        try:
                            os.replace("model_tmp.pt", "model.pt")
                            break
                        except PermissionError:
                            time.sleep(0.2)
                    print(
                        f"\nОбучение. Loss: {loss.item():.4f} "
                        f"(P: {loss_p.item():.4f}, V: {loss_v.item():.4f}) | "
                        f"LR: {scheduler.get_last_lr()[0]:.6f}"
                    )
                iteration += 1
            else:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nОстановка...")
        for p in workers:
            p.terminate()
        for p in workers:
            p.join(timeout=5)


# ============================================================
# Экспорт ONNX
# ============================================================
def export_onnx(weights_path="model.pt", onnx_path="model.onnx"):
    device = torch.device("cpu")
    model = ChessNet().to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, NUM_PLANES, BOARD_SIZE, BOARD_SIZE, device=device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"ONNX экспортирован: {onnx_path}")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        export_onnx()
    else:
        alpha_zero_loop()