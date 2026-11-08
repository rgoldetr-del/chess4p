import os
import sys

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    sys.path.insert(0, "/kaggle/working")

sys.path.insert(0, "/kaggle/working")

import numpy as np
import torch

import chess4p_ai

from config import MODEL_CFG

from model_utils import create_model, load_model_weights

from selfplay_utils import (
    run_engine_debug_tests,
    run_rotation_debug_tests,
    ReplayBufferSparse,
    augment_sparse_batch,
    pack_legal_indices,
    mcts_search,
    TENSOR_PLANES,
    ACTION_SPACE,
    POLICY_K,
)


def test_engine_basic():
    assert chess4p_ai.TENSOR_PLANES == 100
    assert chess4p_ai.ACTION_SPACE == 38416

    game = chess4p_ai.PyGameState()

    tensor = np.zeros((TENSOR_PLANES, 14, 14), dtype=np.float32)
    game.board_to_tensor(tensor)

    legal = game.get_legal_move_indices()

    assert len(legal) > 0
    assert len(set(legal)) == len(legal)

    for idx in legal:
        assert 0 <= int(idx) < ACTION_SPACE

    print("[debug] engine basic OK", flush=True)


def test_buffer_and_augment():
    buf = ReplayBufferSparse(capacity=8)

    state = np.random.rand(TENSOR_PLANES, 14, 14).astype(np.float32)

    idx = np.random.randint(0, ACTION_SPACE, size=POLICY_K).astype(np.uint16)
    prob = np.random.rand(POLICY_K).astype(np.float16)
    prob = prob / max(1e-6, float(prob.sum()))

    legal = np.random.randint(0, ACTION_SPACE, size=25).astype(np.int64)
    packed = pack_legal_indices(legal)

    soft = np.eye(4, dtype=np.float32).astype(np.float16)

    for _ in range(8):
        buf.push(state, idx, prob, packed, soft)

    s, pi, pp, m, v = buf.sample(4)

    ts, ti, tp, tm, tv = augment_sparse_batch(s, pi, pp, m, v)

    assert ts.shape == (4, TENSOR_PLANES, 14, 14)
    assert ti.shape == (4, POLICY_K)
    assert tp.shape == (4, POLICY_K)
    assert tm.shape == (4, ACTION_SPACE)
    assert tv.shape == (4, 4, 4)

    print("[debug] buffer/augment OK", flush=True)


def test_model_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(
        planes=MODEL_CFG["planes"],
        blocks=MODEL_CFG["blocks"],
        channels=MODEL_CFG["channels"],
    ).to(device)

    model.eval()

    x = torch.randn(2, TENSOR_PLANES, 14, 14, device=device)

    with torch.no_grad():
        p, v = model(x)

    assert p.shape == (2, ACTION_SPACE)
    assert v.shape == (2, 4, 4)
    assert torch.isfinite(p).all()
    assert torch.isfinite(v).all()

    print("[debug] model forward OK", flush=True)


def test_model_upgrade_73_to_100():
    device = torch.device("cpu")

    model = create_model(planes=100, blocks=19, channels=192).to(device)

    sd = model.state_dict()

    # Эмулируем старый conv_init с 73 каналами.
    sd["conv_init.0.weight"] = torch.randn(192, 73, 3, 3)

    load_model_weights(model, sd, verbose=False)

    x = torch.randn(1, 100, 14, 14)

    with torch.no_grad():
        p, v = model(x)

    assert torch.isfinite(p).all()
    assert torch.isfinite(v).all()

    print("[debug] model upgrade 73->100 OK", flush=True)


def test_mcts_one_move():
    if not torch.cuda.is_available():
        print("[debug] CUDA not available, skipping MCTS test", flush=True)
        return

    device = torch.device("cuda")

    model = create_model(
        planes=MODEL_CFG["planes"],
        blocks=MODEL_CFG["blocks"],
        channels=MODEL_CFG["channels"],
    ).to(device)

    model.eval()

    game = chess4p_ai.PyGameState()
    game.set_deterministic_dead_king(True)

    cfg = {
        "min_sims": 4,
        "max_sims": 8,
        "batch_size": 4,
        "max_time": 10.0,
        "max_depth": 64,
        "forced_depth": 8,
    }

    result = mcts_search(
        game,
        model,
        device,
        last_root=None,
        temperature=1.0,
        cfg=cfg,
    )

    assert result is not None

    best_idx, top_idx, top_prob, legal_packed, new_root, dbg = result

    assert 0 <= int(best_idx) < ACTION_SPACE
    assert top_idx.shape == (POLICY_K,)
    assert top_prob.shape == (POLICY_K,)

    print("[debug] MCTS one move OK", flush=True)
    print("[debug] MCTS debug:", dbg, flush=True)


if __name__ == "__main__":
    print("=== DEBUG TESTS START ===", flush=True)

    test_engine_basic()
    run_engine_debug_tests()
    run_rotation_debug_tests()

    test_buffer_and_augment()
    test_model_forward()
    test_model_upgrade_73_to_100()
    test_mcts_one_move()

    print("=== ALL DEBUG TESTS OK ===", flush=True)