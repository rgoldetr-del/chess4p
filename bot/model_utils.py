"""model_utils.py — модель 30x256x256 и загрузка весов.

Важно:
- policy_out и value_fc обязаны загружаться, если они есть в checkpoint.
- старый low-rank policy head (policy_from/policy_to) игнорируется,
  потому что в новой модели его нет.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Blocks
# ============================================================

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        out += residual
        return F.relu(out)


# ============================================================
# Model
# ============================================================

class ChessNet256(nn.Module):
    """
    Новая архитектура:
    - 256 плоскостей
    - 30 блоков
    - 256 каналов
    - full-rank policy head
    - value head [B, 4, 4]
    """

    def __init__(
        self,
        planes: int = 256,
        blocks: int = 30,
        channels: int = 256,
        policy_channels: int = 256,
        value_channels: int = 64,
    ):
        super().__init__()

        self.planes = planes
        self.blocks = blocks
        self.channels = channels

        self.conv_init = nn.Sequential(
            nn.Conv2d(planes, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(blocks)]
        )

        # Policy head:
        # [B, 196, 14, 14] -> flatten -> [B, 38416]
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, policy_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(),
        )
        self.policy_out = nn.Conv2d(policy_channels, 196, kernel_size=1)

        nn.init.normal_(self.policy_out.weight, std=0.01)
        nn.init.zeros_(self.policy_out.bias)

        # Value head:
        # [B, 64, 4, 4] -> [B, 1024] -> fc -> [B, 16] -> [B, 4, 4]
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, value_channels, kernel_size=1),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(),
        )

        self.value_fc = nn.Sequential(
            nn.Linear(value_channels * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4 * 4),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = self.res_blocks(x)

        # Policy
        p = self.policy_head(x)
        p = self.policy_out(p)
        policy_logits = p.flatten(1)

        # Value
        v = self.value_head(x)
        v = F.adaptive_avg_pool2d(v, 4)
        v = v.flatten(1)
        v = self.value_fc(v)
        v = v.view(-1, 4, 4)

        return policy_logits, v


def create_model(
    planes: int = 256,
    blocks: int = 30,
    channels: int = 256,
) -> ChessNet256:
    return ChessNet256(
        planes=planes,
        blocks=blocks,
        channels=channels,
    )


# ============================================================
# State dict helpers
# ============================================================

def clean_state_dict(state_dict) -> dict:
    """Убирает module. и оставляет только dict."""
    if state_dict is None:
        return {}

    if hasattr(state_dict, "state_dict"):
        try:
            state_dict = state_dict.state_dict()
        except Exception:
            return {}

    if not isinstance(state_dict, dict):
        return {}

    out = {}
    for k, v in state_dict.items():
        nk = str(k).replace("module.", "")
        out[nk] = v

    return out


def load_state_dict_from_path(path: str, map_location="cpu") -> dict:
    """Загружает state_dict из checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"model file not found: {path}")

    try:
        sd = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        sd = torch.load(path, map_location=map_location, weights_only=False)

    if hasattr(sd, "state_dict"):
        try:
            sd = sd.state_dict()
        except Exception:
            pass

    if isinstance(sd, dict):
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        elif "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]

    return clean_state_dict(sd)


def _overlap_slices(old_shape, new_shape):
    slices_old = []
    slices_new = []

    for o, n in zip(old_shape, new_shape):
        m = min(o, n)
        slices_old.append(slice(0, m))
        slices_new.append(slice(0, m))

    return tuple(slices_old), tuple(slices_new)


def _upgrade_tensor(
    old_w: torch.Tensor,
    new_w: torch.Tensor,
    key: str,
) -> Optional[torch.Tensor]:
    """
    Апгрейд старых весов 19x192x100 -> 30x256x256.

    Используется только если форма не совпадает.
    Для полностью новой модели 30x256x256 обычно ничего не апгрейдится.
    """
    old_shape = tuple(old_w.shape)
    new_shape = tuple(new_w.shape)

    if old_shape == new_shape:
        return old_w.clone()

    if len(old_shape) != len(new_shape):
        return None

    # Scalars: например num_batches_tracked.
    if len(old_shape) == 0:
        if old_shape == new_shape:
            return old_w.clone()
        return torch.zeros(new_shape, dtype=old_w.dtype)

    # 1D tensors: BN / bias / running stats.
    if len(old_shape) == 1:
        if "running_var" in key:
            new = torch.ones(new_shape, dtype=old_w.dtype)
        elif "weight" in key and ("bn" in key or "norm" in key):
            new = torch.ones(new_shape, dtype=old_w.dtype)
        else:
            new = torch.zeros(new_shape, dtype=old_w.dtype)

        m = min(old_shape[0], new_shape[0])
        new[:m] = old_w[:m]
        return new

    # Linear weights лучше не пытаться частично копировать,
    # если размерности отличаются.
    if len(old_shape) == 2:
        return None

    # Conv weights.
    if len(old_shape) == 4 and old_shape[2:] != new_shape[2:]:
        return None

    new = torch.empty(new_shape, dtype=old_w.dtype)

    if len(new_shape) >= 2:
        nn.init.kaiming_normal_(new, nonlinearity="relu")
    else:
        nn.init.normal_(new, std=0.02)

    # Новые расширенные части делаем маленькими,
    # чтобы не сломать старый trunk.
    new.mul_(0.02)

    so, sn = _overlap_slices(old_shape, new_shape)
    new[sn] = old_w[so]

    return new


def load_model_weights(
    model: nn.Module,
    state_dict: dict,
    verbose: bool = True,
    allow_upgrades: bool = True,
) -> dict:
    """
    Загрузка весов.

    Важно:
    - policy_out и value_fc НЕ пропускаются.
    - если checkpoint новый, они должны попасть в loaded.
    """
    sd = clean_state_dict(state_dict)
    model_sd = model.state_dict()

    loaded = []
    upgraded = []
    skipped = []

    for k, new_w in model_sd.items():
        if k not in sd:
            skipped.append((k, "missing_in_checkpoint"))
            continue

        old_w = sd[k]

        if tuple(old_w.shape) == tuple(new_w.shape):
            model_sd[k] = old_w
            loaded.append(k)
            continue

        if not allow_upgrades:
            skipped.append(
                (
                    k,
                    f"shape {tuple(old_w.shape)} != {tuple(new_w.shape)}",
                )
            )
            continue

        up = _upgrade_tensor(old_w, new_w, k)
        if up is None:
            skipped.append(
                (
                    k,
                    f"shape {tuple(old_w.shape)} != {tuple(new_w.shape)}",
                )
            )
        else:
            model_sd[k] = up
            upgraded.append(k)

    model.load_state_dict(model_sd, strict=True)

    if verbose:
        print(f"[model_utils] loaded tensors:   {len(loaded)}")
        print(f"[model_utils] upgraded tensors: {len(upgraded)}")
        print(f"[model_utils] skipped tensors:  {len(skipped)}")

        important_keys = [
            "policy_out.weight",
            "policy_out.bias",
            "value_fc.0.weight",
            "value_fc.0.bias",
            "value_fc.2.weight",
            "value_fc.4.weight",
        ]

        print("[model_utils] important heads:")
        for k in important_keys:
            if k in loaded:
                print(f"    [LOADED]   {k}")
            elif k in upgraded:
                print(f"    [UPGRADED] {k}")
            else:
                print(f"    [WARNING]  {k} not loaded")

        if upgraded:
            print("[model_utils] upgraded examples:")
            for k in upgraded[:12]:
                print("   ", k)

        if skipped:
            print("[model_utils] skipped examples:")
            for k, reason in skipped[:20]:
                print(f"    {k}: {reason}")

    return {
        "loaded": loaded,
        "upgraded": upgraded,
        "skipped": skipped,
    }


def load_model_from_path(
    path: str,
    device: torch.device,
    model_cfg: Optional[dict] = None,
    verbose: bool = True,
) -> ChessNet256:
    """
    Удобный helper:
    - создаёт модель;
    - переносит на device;
    - загружает веса.
    """
    if model_cfg is None:
        model_cfg = {}

    model = create_model(
        planes=int(model_cfg.get("planes", 256)),
        blocks=int(model_cfg.get("blocks", 30)),
        channels=int(model_cfg.get("channels", 256)),
    )

    model.to(device)

    sd = load_state_dict_from_path(path, map_location=device)
    load_model_weights(model, sd, verbose=verbose)

    return model